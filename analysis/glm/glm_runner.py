import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import first_level
from nilearn.image import concat_imgs, mean_img
import logging

from models.models import GLMModel
from models.trial_filters import TrialFilter, TrainingFilter

logger = logging.getLogger(__name__)

class NilearnGLMRunner:
    """
    Nilearn-based GLM runner that replaces SPM/Nipype workflow
    """
    
    def __init__(self, 
                 tr: float = 1.0,
                 high_pass: float = 1/128,
                 hrf_model: str = 'spm',
                 standardize: bool = False,
                 smoothing_fwhm: Optional[float] = None,
                 n_jobs: int = 1):
        """
        Parameters:
        -----------
        tr : float
            Repetition time in seconds
        high_pass : float  
            High-pass filter cutoff in Hz (1/128 = 0.0078 Hz default)
        hrf_model : str
            HRF model ('spm', 'glover', 'fir')
        standardize : bool
            Whether to standardize data
        smoothing_fwhm : float, optional
            Spatial smoothing FWHM in mm (if None, assumes already smoothed)
        n_jobs : int
            Number of parallel jobs
        """
        self.tr = tr
        self.high_pass = high_pass  
        self.hrf_model = hrf_model
        self.standardize = standardize
        self.smoothing_fwhm = smoothing_fwhm
        self.n_jobs = n_jobs
        
    def run_subject_glm(self,
                       subject_id: str,
                       model: GLMModel,
                       func_files: List[Union[str, Path]],
                       events_files: List[Union[str, Path]],
                       confounds_files: Optional[List[Union[str, Path]]] = None,
                       behavior_file: Optional[Union[str, Path]] = None,
                       mask_img: Optional[Union[str, Path]] = None,
                       output_dir: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, nib.Nifti1Image], FirstLevelModel]:
        """
        Run GLM for a single subject
        
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
        output_dir : str, optional
            Directory to save outputs (if None, returns in-memory only)
            
        Returns:
        --------
        beta_maps : dict
            Dictionary mapping condition names to beta images
        glm_model : FirstLevelModel
            Fitted GLM model
        """
        logger.info(f"Running GLM for {subject_id} with model {model.name}")
        
        # Load behavioral data if needed
        behavior = None
        if behavior_file:
            behavior = pd.read_csv(behavior_file, sep='\t')
            
        # Create concatenated design matrix
        all_events, all_confounds = self._create_concatenated_design_info(
            func_files, events_files, confounds_files, model, behavior
        )
        
        # Get total number of scans across all runs
        total_scans = sum(nib.load(f).shape[-1] for f in func_files)
        frame_times = np.arange(total_scans) * self.tr
        
        # Create single design matrix for all runs
        # This automatically creates separate regressors per run-condition
        design_matrix = first_level.make_first_level_design_matrix(
            frame_times=frame_times,
            events=all_events,
            hrf_model=self.hrf_model,
            high_pass=self.high_pass,
            add_regs=all_confounds,
            add_reg_names=all_confounds.columns.tolist() if all_confounds is not None else None
        )
        
        logger.info(f"Created design matric: {design_matrix.shape}")
        condition_regressors = [col for col in design_matrix.columns 
                                if not col.startswith(('drift', 'trans_', 'rot_', 'constant'))]
        logger.info(f"Condition regressors: {condition_regressors}")

        confound_regressors = [col for col in design_matrix.columns 
                               if col.startswith(('trans_', 'rot_'))]
        logger.info(f"Motion confounds: {confound_regressors}")
        
        # Fit single GLM across all runs (like in SPM)
        glm = FirstLevelModel(
            t_r=self.tr,
            high_pass=self.high_pass,
            hrf_model=self.hrf_model,
            standardize=self.standardize,
            smoothing_fwhm=self.smoothing_fwhm,
            n_jobs=self.n_jobs,
            mask_img=mask_img
        )
        
        logger.info("Fitting GLM...")
        glm.fit(func_files, [design_matrix])
        
        # Extract run-wise betas from the GLM
        beta_maps = self._extract_run_wise_betas(
            glm, design_matrix, len(func_files)
        )
        
        # Save outputs if requested
        if output_dir:
            self._save_outputs(beta_maps, glm, subject_id, model.name, output_dir)
            
        logger.info(f"GLM completed for {subject_id}")
        
        return beta_maps
    
    def _create_concatenated_design_info(self,
                                         func_files: List[Union[str, Path]],
                                         events_files: List[Union[str, Path]],
                                         confounds_files: Optional[List[Union[str, Path]]],
                                         model,
                                         behavior: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Create concatenated events and confounds (like SPM concatenation)
        """
        
        all_events_list = []
        all_confounds_list = []
        
        cumulative_time = 0.0
        
        for run_idx, (func_file, events_file) in enumerate(zip(func_files, events_files)):
            
            # Get session info from your model
            session_info = model.specify_model(events_file, behavior)
            
            # Convert to events DataFrame
            run_events = self._convert_session_info_to_events(session_info, run_idx)
            
            # Add run information to condition names
            run_events['trial_type']  = run_events['trial_type'] + f'_run{run_idx+1:02d}'
            
            # Adjust timing for concatenation
            run_events['onset'] += cumulative_time
            
            all_events_list.append(run_events)
            
            # Handle confounds
            if confounds_files:
                confounds = self._load_confounds(confounds_files[run_idx])
                all_confounds_list.append(confounds)
                
            # Update cumulative time
            func_img = nib.load(func_file)
            cumulative_time += func_img.shape[-1] * self.tr
            
        # Concatenate everything
        all_events = pd.concat(all_events_list, ignore_index=True)
        
        # Concatenate confounds
        all_confounds = None
        if all_confounds_list:
            all_confounds = pd.concat(all_confounds_list, ignore_index=True)
            
        return all_events, all_confounds
    
    def _load_confounds(self, confounds_file: Union[str, Path]) -> pd.DataFrame:
        """Load motion parameters and other confounds"""
        confounds_file = Path(confounds_file)
        
        if confounds_file.suffix == '.txt':
            # SPM realignment parameters
            confounds = pd.read_csv(confounds_file, sep='\s+', header=None)
            confounds.columns = ['trans_x', 'trans_y', 'trans_z', 
                               'rot_x', 'rot_y', 'rot_z']
        else:
            # TSV format with headers
            confounds = pd.read_csv(confounds_file, sep='\t')
            
        return confounds
    
    def _convert_session_info_to_events(self, session_info, run_idx: int) -> pd.DataFrame:
        """Convert session info to events DataFrame"""
        events_list = []
        
        for i, condition in enumerate(session_info.conditions):
            onsets = session_info.onsets[i]
            durations = session_info.durations[i]
            
            for onset, duration in zip(onsets, durations):
                events_list.append({
                    'onset': onset,
                    'duration': duration, 
                    'trial_type': condition,
                    'run': run_idx
                })
                
        return pd.DataFrame(events_list).sort_values('onset')
    
    def _extract_run_wise_betas(self, 
                               glm: FirstLevelModel,
                               design_matrix: pd.DataFrame,
                               n_runs: int) -> Dict[str, List[nib.Nifti1Image]]:
        """Extract run-wise betas from the single GLM"""
        
        run_wise_betas = {}
        
        # Get condition regressors
        condition_regressors = [col for col in design_matrix.columns 
                              if not col.startswith(('drift', 'trans_', 'rot_', 'constant'))]
        
        # Group by base condition
        conditions = set()
        for regressor in condition_regressors:
            if '_run' in regressor:
                condition = regressor.split('_run')[0]
                conditions.add(condition)
        
        # Extract betas
        for condition in conditions:
            run_wise_betas[condition] = []
            
            for run_idx in range(n_runs):
                regressor_name = f"{condition}_run{run_idx+1:02d}"
                
                try:
                    beta_map = glm.compute_contrast(regressor_name, output_type='effect_size')
                    run_wise_betas[condition].append(beta_map)
                    logger.info(f"  Extracted: {regressor_name}")
                except Exception as e:
                    logger.warning(f"  Could not extract {regressor_name}: {e}")
        
        return run_wise_betas
    
    def _save_outputs(self, 
                     run_wise_betas: Dict[str, List[nib.Nifti1Image]],
                     subject_id: str, 
                     model_name: str,
                     output_dir: Union[str, Path]) -> None:
        """Save run-wise outputs"""
        output_dir = Path(output_dir) / subject_id / f"model_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for condition, beta_list in run_wise_betas.items():
            for run_idx, beta_map in enumerate(beta_list):
                filename = output_dir / f"beta_{condition}_run-{run_idx+1:02d}.nii.gz"
                beta_map.to_filename(filename)
        
        # Save metadata
        all_conditions = []
        all_runs = []
        for condition, beta_list in run_wise_betas.items():
            for run_idx in range(len(beta_list)):
                all_conditions.append(condition)
                all_runs.append(run_idx + 1)
        
        metadata_df = pd.DataFrame({'condition': all_conditions, 'run': all_runs})
        metadata_df.to_csv(output_dir / "run_wise_conditions.csv", index=False)
        
        logger.info(f"Saved {len(all_conditions)} run-wise beta maps")

        
class FIRGLMRunner(NilearnGLMRunner):
    """GLM runner using FIR response function for time-resolved betas"""
    def __init__(self, fir_length: int = 20, fir_order: int = 10, **kwargs):
        kwargs['hrf_model'] = ('fir', fir_length, fir_order)
        super().__init__(**kwargs)
        self.fir_length = fir_length
        self.fir_order = fir_order