import os
os.environ.pop("FORCE_SPMMCR", None)
os.environ.pop("SPMMCRCMD", None)

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import nibabel as nib
import logging
import tempfile
import shutil
import sys
import re
import scipy.io as spio
from tqdm import tqdm
from glob import glob

from nipype import Node, Workflow, IdentityInterface, Function
from nipype.algorithms import modelgen
from nipype.interfaces import spm
from nipype.interfaces.spm.model import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces.base import Bunch
import nipype.interfaces.io as nio
from nipype.interfaces.matlab import MatlabCommand

from models.models import GLMModel
from models.trial_filters import TrialFilter, TrainingFilter

logger = logging.getLogger(__name__)

def loadmat(filename):
    """
    Load MATLAB .mat file with proper handling of nested structures
    
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls _check_keys to cure all entries
    which are still mat-objects.
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict_):
    """
    Check if entries in dictionary are mat-objects. If yes, convert to nested dict
    """
    for key in dict_:
        if isinstance(dict_[key], spio.matlab.mio5_params.mat_struct):
            dict_[key] = _todict(dict_[key])
    return dict_

def _todict(matobj):
    """
    Recursively convert mat-object to nested dictionary
    """
    dict_ = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict_[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            dict_[strg] = elem
        else:
            dict_[strg] = elem
    return dict_


class SPMGLMRunner:
    """
    SPM/Nipype-based GLM runner that replicates the original workflow approach
    """
    
    def __init__(self, 
                 tr: float = 1.0,
                 high_pass: float = 128.0,  # SPM uses cutoff in seconds, not Hz
                 matlab_cmd: Optional[str] = '/opt/matlab/R2022b/bin/matlab -nojvm -nodisplay',
                 spm_path: Optional[str] = '/home/common/matlab/spm12',
                 working_dir: Optional[Union[str, Path]] = None):
        """
        Parameters:
        -----------
        tr : float
            Repetition time in seconds
        high_pass : float  
            High-pass filter cutoff in seconds (SPM convention: 128s default)
        matlab_cmd : str, optional
            Path to MATLAB executable
        spm_path : str, optional
            Path to SPM installation
        working_dir : str, optional
            Working directory for Nipype (if None, creates temporary)
        """
        self.tr = tr
        self.high_pass = high_pass
        self.working_dir = working_dir
        
        # Set up MATLAB/SPM - CRITICAL: Must set both MatlabCommand AND SPMCommand paths
        if matlab_cmd:
            MatlabCommand.set_default_matlab_cmd(matlab_cmd)
        if spm_path:
            MatlabCommand.set_default_paths(spm_path)
            spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd)
            
    def run_subject_glm(self,
                       subject_id: str,
                       model: GLMModel,
                       func_files: List[Union[str, Path]],
                       events_files: List[Union[str, Path]],
                       confounds_files: Optional[List[Union[str, Path]]] = None,
                       behavior_file: Optional[Union[str, Path]] = None,
                       mask_img: Optional[Union[str, Path]] = None,
                       contrasts: Optional[Dict[str, Union[str, List, np.ndarray]]] = None,
                       output_dir: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, List[nib.Nifti1Image]], Dict[str, Dict[str, nib.Nifti1Image]], str]:
        """
        Run SPM GLM for a single subject using Nipype workflow
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        model : GLMModel
            Configured GLM model with filters
        func_files : list
            List of functional run files
        events_files : list  
            List of events files (one per run)
        confounds_files : list, optional
            List of confounds files (motion parameters, etc.)
        behavior_file : str, optional
            Behavioral data file (needed for test task)
        mask_img : str, optional
            Brain mask image
        contrasts : dict, optional
            Dictionary defining contrasts to compute
        output_dir : str, optional
            Directory to save outputs (if None, returns in-memory only)
            
        Returns:
        --------
        beta_maps : dict
            Dictionary mapping condition names to lists of beta images (one per run)
        contrast_maps : dict
            Dictionary mapping contrast names to contrast results
        spm_mat_file : str
            Path to SPM.mat file
        """
        logger.info(f"Running SPM GLM for {subject_id} with model {model.name}")
        
        # Create working directory
        if self.working_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"spm_glm_{subject_id}_")
        else:
            work_dir = Path(self.working_dir) / f"spm_glm_{subject_id}"
            work_dir.mkdir(parents=True, exist_ok=True)
            work_dir = str(work_dir)
            
        try:
            # Load behavioral data if needed
            behavior = None
            if behavior_file:
                behavior = pd.read_csv(behavior_file, sep='\t')
            
            # Create workflow
            workflow_name = f"spm_glm_{subject_id}_{model.name}"
            wf = Workflow(name=workflow_name)
            wf.base_dir = work_dir
            
            # Create subject info for each run
            subj_info = self._create_subject_info(
                func_files, events_files, confounds_files, model, behavior
            )
            
            # Set up workflow nodes
            beta_maps, contrast_maps, spm_mat_file = self._setup_and_run_workflow(
                wf, subj_info, func_files, contrasts, output_dir, subject_id, model.name
            )
            
            logger.info(f"SPM GLM completed for {subject_id}")
            return beta_maps, contrast_maps, spm_mat_file
            
        finally:
            # Clean up working directory if temporary
            if self.working_dir is None:
                shutil.rmtree(work_dir, ignore_errors=True)
                
    def _create_subject_info(self,
                            func_files: List[Union[str, Path]],
                            events_files: List[Union[str, Path]],
                            confounds_files: Optional[List[Union[str, Path]]],
                            model: GLMModel,
                            behavior: Optional[pd.DataFrame]) -> List[Bunch]:
        """Create subject info for each run"""
        
        subj_info = []
        
        for run_idx, (func_file, events_file) in enumerate(zip(func_files, events_files)):
            # Get session info from model
            session_info = model.specify_model(events_file, behavior)
            
            # Convert to Bunch format expected by SPM
            bunch = Bunch(
                conditions=session_info.conditions,
                onsets=session_info.onsets,
                durations=session_info.durations
            )
            
            # Add motion regressors if provided
            if confounds_files and len(confounds_files) > run_idx:
                motion_regressors, motion_names = self._load_motion_regressors(
                    confounds_files[run_idx]
                )
                bunch.regressor_names = motion_names
                bunch.regressors = motion_regressors
            
            subj_info.append(bunch)
        
        logger.info(f"Created subject info for {len(subj_info)} runs")
        return subj_info
    
    def _load_motion_regressors(self, confounds_file: Union[str, Path]) -> Tuple[List[List[float]], List[str]]:
        """Load motion parameters in SPM format"""
        confounds_file = Path(confounds_file)
        
        if confounds_file.suffix == '.txt':
            # SPM realignment parameters (6 columns: tx, ty, tz, rx, ry, rz)
            motion_data = np.loadtxt(confounds_file)
        else:
            # TSV format - extract motion columns
            confounds = pd.read_csv(confounds_file, sep='\t')
            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            
            # Try alternative column names if standard ones not found
            if not all(col in confounds.columns for col in motion_cols):
                alt_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
                if all(col in confounds.columns for col in alt_cols):
                    motion_cols = alt_cols
                else:
                    raise ValueError(f"Could not find motion columns in {confounds_file}")
            
            motion_data = confounds[motion_cols].values
        
        # Convert to list of lists (one per regressor)
        motion_regressors = [motion_data[:, i].tolist() for i in range(motion_data.shape[1])]
        motion_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        
        return motion_regressors, motion_names
    
    def _setup_and_run_workflow(self,
                               wf: Workflow,
                               subj_info: List[Bunch],
                               func_files: List[Union[str, Path]],
                               contrasts: Optional[Dict],
                               output_dir: Optional[Union[str, Path]],
                               subject_id: str,
                               model_name: str) -> Tuple[Dict, Dict, str]:
        """Set up and run the SPM workflow"""
        
        # Input node
        inputnode = Node(IdentityInterface(fields=['subj_info', 'func_files']),
                         name='inputnode')
        inputnode.inputs.subj_info = subj_info
        inputnode.inputs.func_files = func_files
        
        # SPM model specification
        modelspec = Node(modelgen.SpecifySPMModel(), name='modelspec')
        modelspec.inputs.high_pass_filter_cutoff = self.high_pass
        modelspec.inputs.concatenate_runs = False  # Keep runs separate like original
        modelspec.inputs.input_units = 'secs'
        modelspec.inputs.output_units = 'secs'
        modelspec.inputs.time_repetition = self.tr
        
        # Level 1 design - with potential version checking workaround
        level1design = Node(Level1Design(), name='level1design')
                
        level1design.inputs.timing_units = 'secs'
        level1design.inputs.interscan_interval = self.tr
        level1design.inputs.flags = {'mthresh': 0.8}
        level1design.inputs.microtime_onset = 6.0
        level1design.inputs.microtime_resolution = 11
        level1design.inputs.model_serial_correlations = 'AR(1)'
        level1design.inputs.volterra_expansion_order = 1
        
        # Set basis functions - regular HRF for base class
        level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        
        # Model estimation
        modelest = Node(EstimateModel(), name='modelest')
        modelest.inputs.estimation_method = {'Classical': 1}
        modelest.inputs.write_residuals = False
        
        # Output node
        outputnode = Node(IdentityInterface(fields=['beta_images', 'spm_mat_file', 
                                                   'con_images', 'spmT_images']), 
                         name='outputnode')
        
        # Connect workflow
        wf.connect([
            (inputnode, modelspec, [('subj_info', 'subject_info'),
                                   ('func_files', 'functional_runs')]),
            (modelspec, level1design, [('session_info', 'session_info')]),
            (level1design, modelest, [('spm_mat_file', 'spm_mat_file')]),
            (modelest, outputnode, [('beta_images', 'beta_images'),
                                   ('spm_mat_file', 'spm_mat_file')])
        ])
        
        # Add contrast estimation if requested
        if contrasts:
            contrest = self._add_contrast_node(wf, modelest, outputnode, contrasts)
        
        # Add data sink if output directory specified
        if output_dir:
            datasink = self._add_datasink(wf, outputnode, output_dir, subject_id, model_name)
        
        # Run workflow
        logger.info("Running SPM workflow...")
        #wf.run()
        
        # Get SPM.mat file path for parsing
        spm_mat_file = None
        try:
            spm_mat_file = datasink.inputs.base_directory + '/betas/SPM.mat'
        except:
            logger.warning("Could not get SPM.mat file path")
            
        # Get beta files
        beta_files = sorted(glob(datasink.inputs.base_directory+'/betas/beta_*.nii'))
        
        # Extract results with proper condition names
        beta_maps = self._extract_beta_maps(beta_files, spm_mat_file)
        contrast_maps = self._extract_contrast_maps(outputnode) if contrasts else {}
        
        return beta_maps, contrast_maps, spm_mat_file
    
    def _add_contrast_node(self, wf, modelest, outputnode, contrasts):
        """Add contrast estimation node to workflow"""
        
        # Create contrast specification function
        def specify_contrasts():
            contrast_list = []
            for name, spec in contrasts.items():
                if isinstance(spec, tuple) and len(spec) == 4:
                    # SPM format: (name, type, conditions, weights)
                    contrast_list.append(spec)
                else:
                    # Convert other formats to SPM format
                    # This would need more sophisticated parsing
                    logger.warning(f"Contrast {name} format not fully supported in SPM mode")
            return contrast_list
        
        contspec = Node(Function(input_names=[], 
                               output_names=['contrasts'],
                               function=specify_contrasts), 
                       name='contspec')
        
        contrest = Node(EstimateContrast(), name='contrest')
        
        wf.connect([
            (modelest, contrest, [('spm_mat_file', 'spm_mat_file'),
                                 ('beta_images', 'beta_images'),
                                 ('residual_image', 'residual_image')]),
            (contspec, contrest, [('contrasts', 'contrasts')]),
            (contrest, outputnode, [('con_images', 'con_images'),
                                   ('spmT_images', 'spmT_images')])
        ])
        
        return contrest
    
    def _add_datasink(self, wf, outputnode, output_dir, subject_id, model_name):
        """Add data sink node to save outputs"""
        
        datasink = Node(nio.DataSink(parameterization=True), name='datasink')
        datasink.inputs.base_directory = str(Path(output_dir) / subject_id / f"model_{model_name}")
        
        # Set up substitutions for cleaner filenames
        subs = [('_modelest0', ''), ('_contrest0', '')]
        subs.append(('_bases_hrfderivs0.0', ''))
        
        datasink.inputs.substitutions = subs
        
        wf.connect([
            (outputnode, datasink, [('beta_images', 'betas'),
                                   ('spm_mat_file', 'betas.@spm_mat')])
        ])
        
        # Add contrast outputs if available
        try:
            wf.connect([
                (outputnode, datasink, [('con_images', 'contrasts'),
                                       ('spmT_images', 'contrasts.@spmT')])
            ])
        except:
            pass  # No contrast outputs to connect
        
        return datasink
    
    def _parse_spm_mat(self, spm_mat_file: str, is_fir: bool = False) -> Tuple[List[str], List[str], List[str]]:
        """
        Parse SPM.mat file to extract condition names and beta file paths
        
        Parameters:
        -----------
        spm_mat_file : str
            Path to SPM.mat file
        is_fir : bool
            Whether this is a FIR model
            
        Returns:
        --------
        regressor_names : list
            List of all regressor names from SPM
        condition_names : list
            List of condition names (excluding nuisance regressors)
        beta_files : list
            List of beta file paths
        """
        
        try:
            SPM = loadmat(spm_mat_file)
            
            # Extract regressor names
            if is_fir:
                regressor_names = [n[6:] for n in SPM['SPM']['xX']['name']]
            else:
                regressor_names = [n[6:-6] if '*bf(1)' in n else n[6:] for n in SPM['SPM']['xX']['name']]
            
            # Get beta file paths
            beta_files = [os.path.join(os.path.dirname(spm_mat_file), b.fname) 
                         for b in SPM['SPM']['Vbeta']]
            
            # Filter out nuisance regressors
            exclude_patterns = ['buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 
                              'trans_', 'rot_', 'drift_', 'instruction']
            
            condition_names = []
            for name in regressor_names:
                if not any(exclude in name.lower() for exclude in exclude_patterns):
                    condition_names.append(name)
            
            logger.info(f"Parsed SPM.mat: {len(regressor_names)} total regressors, "
                       f"{len(condition_names)} conditions after filtering")
            
            return regressor_names, condition_names, beta_files
            
        except Exception as e:
            logger.error(f"Failed to parse SPM.mat file {spm_mat_file}: {e}")
            return [], [], []
        
    def _extract_beta_maps(self, beta_files: List[Union[str, Path]], spm_mat_file: str = None) -> Dict[str, List[nib.Nifti1Image]]:
        """Extract beta maps from SPM output with proper condition names"""
        
        beta_dict = {}
        toremove_files = []
        
        try:
            # If we have SPM.mat file, parse it for proper names
            if spm_mat_file and os.path.exists(spm_mat_file):
                regressor_names, condition_names, expected_beta_files = self._parse_spm_mat(
                    spm_mat_file, is_fir=False
                )
                
                for i, (regressor_name, beta_file) in enumerate(zip(regressor_names, expected_beta_files)):
                    # Skip nuisance regressors
                    exclude_patterns = ['buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 
                                      'trans_', 'rot_', 'drift_', 'instruction']
                    
                    if any(exclude in regressor_name.lower() for exclude in exclude_patterns):
                        toremove_files.append(beta_file)
                        continue
                    
                    # Clean condition name - remove *bf(1) suffix if present
                    if '*bf(1)' in regressor_name:
                        condition_key = regressor_name.replace('*bf(1)', '')
                    else:
                        condition_key = regressor_name
                    
                    # Track run number for this condition
                    if condition_key not in beta_dict:
                        beta_dict[condition_key] = []
                    
                    # Load beta image if file exists
                    if os.path.exists(beta_file):
                        beta_dict[condition_key].append(beta_file)
            
            else:
                logger.error("SPM.mat not found!")
        
        except Exception as e:
            logger.error(f"Failed to extract beta maps: {e}")
        
        # Log summary
        total_betas = sum(len(runs) for runs in beta_dict.values())
        logger.info(f"Extracted {total_betas} beta maps across {len(beta_dict)} conditions:")
        for condition, runs in beta_dict.items():
            logger.info(f"  {condition}: {len(runs)} runs")
            
        self._rename_beta_files(beta_dict)
        for f in toremove_files:
            os.remove(f)
        os.remove(spm_mat_file)
        
        breakpoint()
        return beta_dict
    
    def _rename_beta_files(self, beta_dict: Dict):
        
        for cond, filelist in beta_dict.items():
            for i, f in enumerate(filelist):
                newfilename = os.path.join(os.path.split(f)[0], f'beta-{cond}-run{i+1:02d}.nii')
                os.rename(f, newfilename)
    
    def _extract_contrast_maps(self, outputnode) -> Dict[str, Dict[str, nib.Nifti1Image]]:
        """Extract contrast maps from SPM output"""
        contrast_maps = {}
        
        try:
            con_files = getattr(outputnode.inputs, 'con_images', [])
            spmT_files = getattr(outputnode.inputs, 'spmT_images', [])
            
            for i, (con_file, spmT_file) in enumerate(zip(con_files, spmT_files)):
                contrast_name = f"contrast_{i+1}"
                contrast_maps[contrast_name] = {
                    'effect_size': nib.load(con_file),
                    't_stat': nib.load(spmT_file)
                }
        except:
            logger.warning("Could not extract contrast maps from SPM output")
        
        return contrast_maps