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
                       contrasts: Optional[Dict[str, Union[str, List, np.ndarray]]] = None,
                       output_dir: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, nib.Nifti1Image], Dict[str, Dict[str, nib.Nifti1Image]], FirstLevelModel]:
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
        contrasts : dict, optional
            Dictionary defining contrasts to compute. Keys are contrast names,
            values can be:
            - str: simple contrast like 'condition_A - condition_B'
            - list: [condition_names] for each condition vs implicit baseline
            - np.array: explicit contrast vector
        output_dir : str, optional
            Directory to save outputs (if None, returns in-memory only)
            
        Returns:
        --------
        beta_maps : dict
            Dictionary mapping condition names to beta images
        contrast_maps : dict
            Dictionary mapping contrast names to contrast results
            Each contrast result is a dict with 'effect_size', 't_stat', 'p_value' maps
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
        design_matrix = first_level.make_first_level_design_matrix(
            frame_times=frame_times,
            events=all_events,
            hrf_model=self.hrf_model,
            high_pass=self.high_pass,
            add_regs=all_confounds,
            add_reg_names=all_confounds.columns.tolist() if all_confounds is not None else None
        )
        
        logger.info(f"Created design matrix: {design_matrix.shape}")
        condition_regressors = [col for col in design_matrix.columns 
                                if not col.startswith(('drift', 'trans_', 'rot_', 'constant'))]
        logger.info(f"Condition regressors: {condition_regressors}")

        confound_regressors = [col for col in design_matrix.columns 
                               if col.startswith(('trans_', 'rot_'))]
        logger.info(f"Motion confounds: {confound_regressors}")
        
        # Fit single GLM across all runs
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
        
        # Compute contrasts if specified
        contrast_maps = {}
        if contrasts:
            contrast_maps = self._compute_contrasts(
                glm, design_matrix, contrasts, len(func_files)
            )
        
        # Save outputs if requested
        if output_dir:
            self._save_outputs(beta_maps, contrast_maps, glm, subject_id, model.name, output_dir)
            
        logger.info(f"GLM completed for {subject_id}")
        
        return beta_maps, contrast_maps, glm
    
    def _compute_contrasts(self,
                          glm: FirstLevelModel,
                          design_matrix: pd.DataFrame,
                          contrasts: Dict[str, Union[str, List, np.ndarray]],
                          n_runs: int) -> Dict[str, Dict[str, nib.Nifti1Image]]:
        """
        Compute specified contrasts
        
        Parameters:
        -----------
        glm : FirstLevelModel
            Fitted GLM model
        design_matrix : pd.DataFrame
            Design matrix
        contrasts : dict
            Contrast specifications
        n_runs : int
            Number of runs
            
        Returns:
        --------
        contrast_maps : dict
            Dictionary mapping contrast names to contrast results
        """
        logger.info("Computing contrasts...")
        contrast_maps = {}
        
        # Get available condition regressors
        condition_regressors = [col for col in design_matrix.columns 
                              if not col.startswith(('drift', 'trans_', 'rot_', 'constant'))]
        
        for contrast_name, contrast_spec in contrasts.items():
            logger.info(f"Computing contrast: {contrast_name}")
            
            try:
                if isinstance(contrast_spec, str):
                    # String-based contrast (e.g., "wide - narrow")
                    contrast_vector = self._parse_string_contrast(
                        contrast_spec, condition_regressors, n_runs
                    )
                elif isinstance(contrast_spec, list):
                    # List of conditions (each vs implicit baseline)
                    contrast_vector = self._create_baseline_contrast(
                        contrast_spec, condition_regressors, n_runs
                    )
                elif isinstance(contrast_spec, np.ndarray):
                    # Explicit contrast vector
                    contrast_vector = contrast_spec
                else:
                    raise ValueError(f"Invalid contrast specification: {contrast_spec}")
                
                # Compute contrast maps
                effect_size = glm.compute_contrast(contrast_vector, output_type='effect_size')
                t_stat = glm.compute_contrast(contrast_vector, output_type='stat')
                p_value = glm.compute_contrast(contrast_vector, output_type='p_value')
                
                contrast_maps[contrast_name] = {
                    'effect_size': effect_size,
                    't_stat': t_stat, 
                    'p_value': p_value
                }
                
                logger.info(f"  Successfully computed contrast: {contrast_name}")
                
            except Exception as e:
                logger.error(f"  Failed to compute contrast {contrast_name}: {e}")
                
        return contrast_maps
    
    
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
                     contrast_maps: Dict[str, Dict[str, nib.Nifti1Image]],
                     glm: FirstLevelModel,
                     subject_id: str, 
                     model_name: str,
                     output_dir: Union[str, Path]) -> None:
        """Save run-wise and contrast outputs"""
        output_dir = Path(output_dir) / subject_id / f"model_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run-wise betas
        for condition, beta_list in run_wise_betas.items():
            for run_idx, beta_map in enumerate(beta_list):
                filename = output_dir / f"beta_{condition}_run-{run_idx+1:02d}.nii.gz"
                beta_map.to_filename(filename)
        
        # Save contrast maps
        contrast_dir = output_dir / "contrasts"
        contrast_dir.mkdir(exist_ok=True)
        
        for contrast_name, contrast_results in contrast_maps.items():
            for map_type, contrast_map in contrast_results.items():
                filename = contrast_dir / f"contrast_{contrast_name}_{map_type}.nii.gz"
                contrast_map.to_filename(filename)
        
        # Save metadata
        all_conditions = []
        all_runs = []
        for condition, beta_list in run_wise_betas.items():
            for run_idx in range(len(beta_list)):
                all_conditions.append(condition)
                all_runs.append(run_idx + 1)
        
        metadata_df = pd.DataFrame({'condition': all_conditions, 'run': all_runs})
        metadata_df.to_csv(output_dir / "run_wise_conditions.csv", index=False)
        
        # Save contrast metadata
        if contrast_maps:
            contrast_metadata = pd.DataFrame({
                'contrast_name': list(contrast_maps.keys()),
                'maps_saved': [list(results.keys()) for results in contrast_maps.values()]
            })
            contrast_metadata.to_csv(contrast_dir / "contrast_metadata.csv", index=False)
        
        logger.info(f"Saved {len(all_conditions)} run-wise beta maps and {len(contrast_maps)} contrasts")

        
class FIRGLMRunner(NilearnGLMRunner):
    """GLM runner using FIR response function for time-resolved betas"""
    
    def __init__(self, 
                 fir_length: int = 10,
                 tr: float = 1.0,
                 high_pass: float = 1/128,
                 standardize: bool = False,
                 smoothing_fwhm: Optional[float] = None,
                 n_jobs: int = 1):
        """
        Parameters:
        -----------
        fir_length : int
            Number of time bins for FIR model (in TRs)
            e.g., 10 for modeling 0-9 TRs post-stimulus
        tr : float
            Repetition time in seconds
        high_pass : float  
            High-pass filter cutoff in Hz
        standardize : bool
            Whether to standardize data
        smoothing_fwhm : float, optional
            Spatial smoothing FWHM in mm
        n_jobs : int
            Number of parallel jobs
        """
        self.fir_length = fir_length
        
        # Initialize parent with FIR HRF model
        super().__init__(
            tr=tr,
            high_pass=high_pass,
            hrf_model=('fir', fir_length),
            standardize=standardize,
            smoothing_fwhm=smoothing_fwhm,
            n_jobs=n_jobs
        )
        
        logger.info(f"FIR setup: {fir_length} time bins (0-{fir_length-1} TRs)")
        
    def run_subject_glm(self,
                       subject_id: str,
                       model,
                       func_files: List[Union[str, Path]],
                       events_files: List[Union[str, Path]],
                       confounds_files: Optional[List[Union[str, Path]]] = None,
                       behavior_file: Optional[Union[str, Path]] = None,
                       mask_img: Optional[Union[str, Path]] = None,
                       output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, List[nib.Nifti1Image]]]:
        """
        Run FIR GLM for a single subject
        
        Returns:
        --------
        fir_beta_maps : dict
            Nested dictionary: {condition: {delay: [beta_maps_per_run]}}
        """
        logger.info(f"Running FIR GLM for {subject_id}")
        
        # Load behavioral data if needed
        behavior = None
        if behavior_file:
            behavior = pd.read_csv(behavior_file, sep='\t')
            
        # Reuse parent class logic for creating concatenated design
        all_events, all_confounds = self._create_concatenated_design_info(
            func_files, events_files, confounds_files, model, behavior
        )
        
        # Get total number of scans across all runs
        total_scans = sum(nib.load(f).shape[-1] for f in func_files)
        frame_times = np.arange(total_scans) * self.tr
        
        # Create FIR design matrix
        design_matrix = first_level.make_first_level_design_matrix(
            frame_times=frame_times,
            events=all_events,
            hrf_model=self.hrf_model,
            high_pass=self.high_pass,
            add_regs=all_confounds,
            add_reg_names=all_confounds.columns.tolist() if all_confounds is not None else None
        )
        
        logger.info(f"Created FIR design matrix: {design_matrix.shape}")
        
        # Fit GLM using parent class parameters
        glm = FirstLevelModel(
            t_r=self.tr,
            high_pass=self.high_pass,
            hrf_model=self.hrf_model,
            standardize=self.standardize,
            smoothing_fwhm=self.smoothing_fwhm,
            n_jobs=self.n_jobs,
            mask_img=mask_img
        )
        
        logger.info("Fitting FIR GLM...")
        glm.fit(func_files, [design_matrix])
        
        # Extract FIR betas with delay information
        fir_beta_maps = self._extract_fir_betas(
            glm, design_matrix, len(func_files)
        )
        
        # Save outputs if requested
        if output_dir:
            self._save_fir_outputs(fir_beta_maps, subject_id, output_dir)
            
        logger.info(f"FIR GLM completed for {subject_id}")
        
        return fir_beta_maps
    
    def _extract_fir_betas(self, 
                          glm: FirstLevelModel,
                          design_matrix: pd.DataFrame,
                          n_runs: int) -> Dict[str, Dict[str, List[nib.Nifti1Image]]]:
        """
        Extract FIR betas organized by condition and delay
        
        Returns:
        --------
        fir_betas : dict
            {condition: {delay_label: [beta_maps_per_run]}}
        """
        logger.info("Extracting FIR betas...")
        
        # Get FIR regressors (reuse parent class logic)
        fir_regressors = [col for col in design_matrix.columns 
                         if not col.startswith(('drift', 'trans_', 'rot_', 'constant'))]
        
        # Parse FIR regressor names
        fir_betas = {}
        
        for regressor in fir_regressors:
            # Parse regressor name: condition_run01_delay_XX
            condition, delay_label = self._parse_fir_regressor_name(regressor)
            
            if condition not in fir_betas:
                fir_betas[condition] = {}
            
            if delay_label not in fir_betas[condition]:
                fir_betas[condition][delay_label] = []
            
            # Extract beta map
            try:
                beta_map = glm.compute_contrast(regressor, output_type='effect_size')
                fir_betas[condition][delay_label].append(beta_map)
                logger.debug(f"  Extracted: {regressor} -> {condition}_{delay_label}")
            except Exception as e:
                logger.warning(f"  Could not extract {regressor}: {e}")
        
        # Sort delays numerically within each condition
        organized_betas = {}
        for condition in fir_betas:
            organized_betas[condition] = {}
            
            # Sort delays numerically
            delays = sorted(fir_betas[condition].keys(), 
                          key=lambda x: int(x) if x.isdigit() else 0)
            
            for delay in delays:
                organized_betas[condition][delay] = fir_betas[condition][delay]
        
        logger.info(f"Extracted FIR betas for {len(organized_betas)} conditions")
        for condition, delays in organized_betas.items():
            logger.info(f"  {condition}: delays {list(delays.keys())}")
        
        return organized_betas
    
    def _parse_fir_regressor_name(self, regressor: str) -> Tuple[str, str]:
        """
        Parse FIR regressor name to extract condition and delay
        
        Parameters:
        -----------
        regressor : str
            FIR regressor name (e.g., 'wide_run01_delay_03')
            
        Returns:
        --------
        condition : str
            Condition name
        delay_label : str
            Simple delay label (e.g., '03')
        """
        parts = regressor.split('_')
        
        # Find run and delay parts
        condition_parts = []
        delay_label = '00'  # Default
        
        for i, part in enumerate(parts):
            if part.startswith('run'):
                # Everything after run should contain delay info
                if i + 2 < len(parts) and parts[i + 1] == 'delay':
                    delay_label = parts[i + 2]
                elif i + 1 < len(parts):
                    # Try to extract delay from remaining parts
                    remaining = '_'.join(parts[i + 1:])
                    if 'delay' in remaining:
                        delay_parts = remaining.split('_')
                        if len(delay_parts) >= 2:
                            delay_label = delay_parts[-1]
                break
            else:
                condition_parts.append(part)
        
        condition = '_'.join(condition_parts)
        
        # Ensure delay label is zero-padded
        try:
            delay_num = int(delay_label)
            delay_label = f"{delay_num:02d}"
        except ValueError:
            delay_label = "00"
        
        return condition, delay_label
    
    def _save_fir_outputs(self, 
                         fir_beta_maps: Dict[str, Dict[str, List[nib.Nifti1Image]]],
                         subject_id: str,
                         output_dir: Union[str, Path]) -> None:
        """
        Save FIR outputs with simple delay labels in filenames
        """
        output_dir = Path(output_dir) / subject_id / "fir_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_rows = []
        
        for condition, delay_dict in fir_beta_maps.items():
            for delay_label, beta_list in delay_dict.items():
                for run_idx, beta_map in enumerate(beta_list):
                    # Simple filename with delay label
                    filename = output_dir / f"beta_{condition}_delay-{delay_label}_run-{run_idx+1:02d}.nii.gz"
                    beta_map.to_filename(filename)
                    
                    # Track metadata
                    metadata_rows.append({
                        'condition': condition,
                        'delay': delay_label,
                        'delay_tr': int(delay_label),
                        'delay_seconds': int(delay_label) * self.tr,
                        'run': run_idx + 1,
                        'filename': filename.name
                    })
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_csv(output_dir / "fir_beta_metadata.csv", index=False)
        
        # Save FIR parameters
        fir_params = {
            'fir_length': self.fir_length,
            'tr_seconds': self.tr,
            'max_delay_tr': self.fir_length - 1,
            'max_delay_seconds': (self.fir_length - 1) * self.tr
        }
        
        fir_params_df = pd.DataFrame([fir_params])
        fir_params_df.to_csv(output_dir / "fir_parameters.csv", index=False)
        
        logger.info(f"Saved {len(metadata_rows)} FIR beta maps")
        logger.info(f"Output directory: {output_dir}")