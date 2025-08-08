import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
import tempfile
import shutil
import os
import re
import scipy.io as spio
from tqdm import tqdm

from nipype import Node, Workflow, IdentityInterface
from nipype.algorithms import modelgen
from nipype.interfaces.spm.model import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces.base import Bunch
import nipype.interfaces.io as nio
from nipype.interfaces.matlab import MatlabCommand

from models.models import GLMModel

logger = logging.getLogger(__name__)

def loadmat(filename):
    """Load MATLAB .mat file with proper handling of nested structures"""
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict_):
    """Check if entries in dictionary are mat-objects. If yes, convert to nested dict"""
    for key in dict_:
        if isinstance(dict_[key], spio.matlab.mio5_params.mat_struct):
            dict_[key] = _todict(dict_[key])
    return dict_

def _todict(matobj):
    """Recursively convert mat-object to nested dictionary"""
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
    """Simplified SPM/Nipype-based GLM runner that handles both HRF and FIR models"""
    
    def __init__(self, 
                 tr: float = 1.0,
                 high_pass: float = 128.0,
                 use_fir: bool = False,
                 fir_length: int = 20,
                 fir_order: int = 10,
                 matlab_cmd: Optional[str] = None,
                 spm_path: Optional[str] = None,
                 working_dir: Optional[Union[str, Path]] = None):
        """
        Parameters:
        -----------
        tr : float
            Repetition time in seconds
        high_pass : float  
            High-pass filter cutoff in seconds (SPM convention: 128s default)
        use_fir : bool
            Whether to use FIR basis functions instead of canonical HRF
        fir_length : int
            FIR length in seconds (only used if use_fir=True)
        fir_order : int
            Number of FIR time bins (only used if use_fir=True)
        matlab_cmd : str, optional
            Path to MATLAB executable
        spm_path : str, optional
            Path to SPM installation
        working_dir : str, optional
            Working directory for Nipype (if None, creates temporary)
        """
        self.tr = tr
        self.high_pass = high_pass
        self.use_fir = use_fir
        self.fir_length = fir_length
        self.fir_order = fir_order
        self.working_dir = working_dir
        
        # Set up MATLAB/SPM
        if matlab_cmd:
            MatlabCommand.set_default_matlab_cmd(matlab_cmd)
        if spm_path:
            MatlabCommand.set_default_paths(spm_path)
        
        logger.info(f"SPM GLM Runner initialized: {'FIR' if use_fir else 'HRF'} model")
        if use_fir:
            logger.info(f"FIR parameters: {fir_length}s length, {fir_order} time bins")
            
    def run_subject_glm(self,
                       subject_id: str,
                       model: GLMModel,
                       func_files: List[Union[str, Path]],
                       events_files: List[Union[str, Path]],
                       confounds_files: Optional[List[Union[str, Path]]] = None,
                       behavior_file: Optional[Union[str, Path]] = None,
                       mask_img: Optional[Union[str, Path]] = None,
                       contrasts: Optional[Dict] = None,
                       output_dir: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
        """
        Run SPM GLM for a single subject
        
        Returns:
        --------
        beta_dict : Dict[str, List[str]]
            Dictionary mapping condition names to lists of beta file paths
        """
        logger.info(f"Running {'FIR ' if self.use_fir else ''}SPM GLM for {subject_id} with model {model.name}")
        
        # Create working directory
        if self.working_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"spm_glm_{subject_id}_")
            cleanup_work_dir = True
        else:
            work_dir = Path(self.working_dir) / f"spm_glm_{subject_id}"
            work_dir.mkdir(parents=True, exist_ok=True)
            work_dir = str(work_dir)
            cleanup_work_dir = False
        
        try:
            # Load behavioral data if needed
            behavior = None
            if behavior_file:
                behavior = pd.read_csv(behavior_file, sep='\t')
            
            # Create and run SPM workflow
            spm_mat_file = self._run_spm_workflow(
                work_dir, subject_id, model, func_files, events_files, 
                confounds_files, behavior, contrasts
            )
            
            # Parse SPM.mat to get beta file mapping
            beta_dict = self._parse_and_organize_betas(spm_mat_file)
            
            # Rename and organize files
            if output_dir:
                beta_dict = self._rename_and_move_files(
                    beta_dict, subject_id, model.name, output_dir, spm_mat_file
                )
            else:
                # Just rename in place
                self._rename_beta_files(beta_dict)
            
            # Clean up unnecessary files
            self._cleanup_spm_files(os.path.dirname(spm_mat_file))
            
            logger.info(f"{'FIR ' if self.use_fir else ''}SPM GLM completed for {subject_id}")
            return beta_dict
            
        finally:
            # Clean up working directory if temporary
            if cleanup_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _run_spm_workflow(self, work_dir: str, subject_id: str, model: GLMModel,
                         func_files: List, events_files: List, confounds_files: Optional[List],
                         behavior: Optional[pd.DataFrame], contrasts: Optional[Dict]) -> str:
        """Run the SPM workflow and return path to SPM.mat file"""
        
        # Create subject info for each run
        subj_info = self._create_subject_info(
            func_files, events_files, confounds_files, model, behavior
        )
        
        # Create workflow
        model_suffix = "_FIR" if self.use_fir else ""
        workflow_name = f"spm_glm_{subject_id}_{model.name}{model_suffix}"
        wf = Workflow(name=workflow_name)
        wf.base_dir = work_dir
        
        # Input node
        inputnode = Node(IdentityInterface(fields=['subj_info', 'func_files']), 
                        name='inputnode')
        inputnode.inputs.subj_info = subj_info
        inputnode.inputs.func_files = func_files
        
        # SPM model specification
        modelspec = Node(modelgen.SpecifySPMModel(), name='modelspec')
        modelspec.inputs.high_pass_filter_cutoff = self.high_pass
        modelspec.inputs.concatenate_runs = False
        modelspec.inputs.input_units = 'secs'
        modelspec.inputs.output_units = 'secs'
        modelspec.inputs.time_repetition = self.tr
        
        # Level 1 design
        level1design = Node(Level1Design(), name='level1design')
        level1design.inputs.timing_units = 'secs'
        level1design.inputs.interscan_interval = self.tr
        level1design.inputs.flags = {'mthresh': 0.8}
        level1design.inputs.microtime_onset = 6.0
        level1design.inputs.microtime_resolution = 11
        level1design.inputs.model_serial_correlations = 'AR(1)'
        level1design.inputs.volterra_expansion_order = 1
        
        # Set basis functions
        if self.use_fir:
            level1design.inputs.bases = {'fir': {'length': self.fir_length, 'order': self.fir_order}}
        else:
            level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
        
        # Model estimation
        modelest = Node(EstimateModel(), name='modelest')
        modelest.inputs.estimation_method = {'Classical': 1}
        modelest.inputs.write_residuals = False
        
        # Connect workflow
        wf.connect([
            (inputnode, modelspec, [('subj_info', 'subject_info'),
                                   ('func_files', 'functional_runs')]),
            (modelspec, level1design, [('session_info', 'session_info')]),
            (level1design, modelest, [('spm_mat_file', 'spm_mat_file')])
        ])
        
        # Add contrast estimation if requested (only for HRF models)
        if contrasts and not self.use_fir:
            contrest = Node(EstimateContrast(), name='contrest')
            # Set up contrasts (simplified - would need proper contrast specification)
            wf.connect([
                (modelest, contrest, [('spm_mat_file', 'spm_mat_file'),
                                     ('beta_images', 'beta_images')])
            ])
        
        # Run workflow
        logger.info(f"Running {'FIR ' if self.use_fir else ''}SPM workflow...")
        wf.run()
        
        # Find SPM.mat file
        spm_mat_files = list(Path(work_dir).rglob("SPM.mat"))
        if not spm_mat_files:
            raise RuntimeError("SPM.mat file not found after workflow execution")
        
        return str(spm_mat_files[0])
    
    def _create_subject_info(self, func_files: List, events_files: List,
                           confounds_files: Optional[List], model: GLMModel,
                           behavior: Optional[pd.DataFrame]) -> List[Bunch]:
        """Create subject info for each run"""
        
        subj_info = []
        
        for run_idx, (func_file, events_file) in enumerate(zip(func_files, events_files)):
            # Get session info from model
            session_info = model.specify_model(events_file, behavior)
            
            # Convert to Bunch format
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
        
        return subj_info
    
    def _load_motion_regressors(self, confounds_file: Union[str, Path]) -> Tuple[List[List[float]], List[str]]:
        """Load motion parameters"""
        confounds_file = Path(confounds_file)
        
        if confounds_file.suffix == '.txt':
            motion_data = np.loadtxt(confounds_file)
        else:
            confounds = pd.read_csv(confounds_file, sep='\t')
            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            
            if not all(col in confounds.columns for col in motion_cols):
                alt_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
                if all(col in confounds.columns for col in alt_cols):
                    motion_cols = alt_cols
                else:
                    raise ValueError(f"Could not find motion columns in {confounds_file}")
            
            motion_data = confounds[motion_cols].values
        
        motion_regressors = [motion_data[:, i].tolist() for i in range(motion_data.shape[1])]
        motion_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        
        return motion_regressors, motion_names
    
    def _parse_and_organize_betas(self, spm_mat_file: str) -> Dict[str, List[str]]:
        """Parse SPM.mat file and organize beta files by condition"""
        
        try:
            SPM = loadmat(spm_mat_file)
            
            # Get regressor names and beta files
            regressor_names = [name[6:] for name in SPM['SPM']['xX']['name']]  # Remove 'Sn(1) ' prefix
            beta_files = [os.path.join(os.path.dirname(spm_mat_file), b.fname) 
                         for b in SPM['SPM']['Vbeta']]
            
            # Organize betas by condition, excluding nuisance regressors
            beta_dict = {}
            exclude_patterns = ['buttonpress', 'constant', 'trans_', 'rot_', 'drift_', 'instruction']
            
            for regressor_name, beta_file in zip(regressor_names, beta_files):
                # Skip nuisance regressors
                if any(exclude in regressor_name.lower() for exclude in exclude_patterns):
                    continue
                
                # Clean condition name
                if self.use_fir:
                    # For FIR: extract condition and delay info
                    if '*bf(' in regressor_name:
                        base_condition = regressor_name[:regressor_name.find('*bf')]
                        delay_match = re.search(r'\*bf\((\d+)\)', regressor_name)
                        delay = int(delay_match.group(1)) - 1 if delay_match else 0
                        condition_key = f"{base_condition}_delay-{delay:02d}"
                    else:
                        condition_key = regressor_name
                else:
                    # For HRF: remove *bf(1) suffix
                    condition_key = regressor_name.replace('*bf(1)', '') if '*bf(1)' in regressor_name else regressor_name
                
                if condition_key not in beta_dict:
                    beta_dict[condition_key] = []
                beta_dict[condition_key].append(beta_file)
            
            logger.info(f"Parsed SPM.mat: found {len(beta_dict)} conditions")
            for condition, files in beta_dict.items():
                logger.info(f"  {condition}: {len(files)} runs")
            
            return beta_dict
            
        except Exception as e:
            logger.error(f"Failed to parse SPM.mat file {spm_mat_file}: {e}")
            return {}
    
    def _rename_beta_files(self, beta_dict: Dict[str, List[str]]) -> None:
        """Rename beta files with meaningful names"""
        
        for condition, filelist in beta_dict.items():
            for run_idx, old_file in enumerate(filelist):
                # Create new filename
                file_dir = os.path.dirname(old_file)
                if self.use_fir:
                    new_filename = os.path.join(file_dir, f'beta_{condition}_run_{run_idx+1:02d}.nii')
                else:
                    new_filename = os.path.join(file_dir, f'beta_{condition}_run_{run_idx+1:02d}.nii')
                
                # Rename file
                if os.path.exists(old_file):
                    os.rename(old_file, new_filename)
                    # Update the list with new filename
                    filelist[run_idx] = new_filename
                    logger.debug(f"Renamed: {os.path.basename(old_file)} -> {os.path.basename(new_filename)}")
    
    def _rename_and_move_files(self, beta_dict: Dict[str, List[str]], subject_id: str,
                              model_name: str, output_dir: Union[str, Path], spm_mat_file: str) -> Dict[str, List[str]]:
        """Rename and move files to output directory"""
        
        # Create output directory
        model_suffix = "_FIR" if self.use_fir else ""
        out_dir = Path(output_dir) / subject_id / f"model_{model_name}{model_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Track metadata
        metadata_rows = []
        new_beta_dict = {}
        
        for condition, filelist in beta_dict.items():
            new_beta_dict[condition] = []
            
            for run_idx, old_file in enumerate(filelist):
                # Create new filename in output directory
                if self.use_fir:
                    new_filename = out_dir / f'beta_{condition}_run-{run_idx+1:02d}.nii'
                else:
                    new_filename = out_dir / f'beta_{condition}_run-{run_idx+1:02d}.nii'
                
                # Copy and rename file
                if os.path.exists(old_file):
                    shutil.copy2(old_file, new_filename)
                    new_beta_dict[condition].append(str(new_filename))
                    
                    # Add to metadata
                    if self.use_fir and '_delay_' in condition:
                        base_condition, delay_part = condition.split('_delay_')
                        delay_tr = int(delay_part)
                        metadata_rows.append({
                            'condition': base_condition,
                            'delay_tr': delay_tr,
                            'delay_seconds': delay_tr * self.tr,
                            'run': run_idx + 1,
                            'filename': new_filename.name
                        })
                    else:
                        metadata_rows.append({
                            'condition': condition,
                            'run': run_idx + 1,
                            'filename': new_filename.name
                        })
        
        # Save metadata
        if metadata_rows:
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_file = "fir_beta_metadata.csv" if self.use_fir else "beta_metadata.csv"
            metadata_df.to_csv(out_dir / metadata_file, index=False)
        
        # Save model parameters
        if self.use_fir:
            params = {
                'runner_type': 'SPM_FIR',
                'fir_length': self.fir_length,
                'fir_order': self.fir_order,
                'tr_seconds': self.tr,
                'use_fir': True
            }
        else:
            params = {
                'runner_type': 'SPM',
                'tr': self.tr,
                'high_pass': self.high_pass,
                'use_fir': False
            }
        params_file = "model_parameters.csv"
        
        pd.DataFrame([params]).to_csv(out_dir / params_file, index=False)
        
        # Copy SPM.mat for reference
        if os.path.exists(spm_mat_file):
            shutil.copy2(spm_mat_file, out_dir / "SPM.mat")
        
        logger.info(f"Moved {sum(len(files) for files in new_beta_dict.values())} files to {out_dir}")
        return new_beta_dict
    
    def _cleanup_spm_files(self, spm_dir: str) -> None:
        """Clean up unnecessary SPM files"""
        
        # Files to remove (keep only renamed betas)
        cleanup_patterns = [
            'beta_*.nii',  # Original beta files (now renamed)
            'mask.nii',
            'ResMS.nii', 
            'RPV.nii',
            'SPM.mat'  # Only if not moved to output
        ]
        
        spm_path = Path(spm_dir)
        for pattern in cleanup_patterns:
            for file_path in spm_path.glob(pattern):
                try:
                    file_path.unlink()
                    logger.debug(f"Removed: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")


# Convenience factory functions
def create_spm_glm_runner(**kwargs) -> SPMGLMRunner:
    """Create standard HRF SPM GLM runner"""
    return SPMGLMRunner(use_fir=False, **kwargs)

def create_fir_spm_glm_runner(fir_length: int = 20, fir_order: int = 10, **kwargs) -> SPMGLMRunner:
    """Create FIR SPM GLM runner"""
    return SPMGLMRunner(use_fir=True, fir_length=fir_length, fir_order=fir_order, **kwargs)