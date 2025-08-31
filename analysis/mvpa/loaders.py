import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from pathlib import Path
import nibabel as nib
from dataclasses import dataclass
import logging
import re
import ipdb

logger = logging.getLogger(__name__)

@dataclass
class MVPADataset:
    # Core data
    data: np.ndarray  # (n_samples, n_features) - the actual fMRI data
    labels: np.ndarray  # (n_samples,) - condition labels
    
    # Metadata
    subject_id: str
    task: str # 'train' or 'test'
    roi: str
    
    # Additional attributes for cross-validation and grouping
    runs: Optional[np.ndarray] = None  # (n_samples,) - run numbers for CV
    splits: Optional[np.ndarray] = None  # (n_samples,) - trial split numbers
    trial_indices: Optional[np.ndarray] = None  # (n_samples,) - original trial indices
    delays: Optional[np.ndarray] = None # (n_samples,) delays for time-resolved FIR
    
    # Coordinate information (optional)
    voxel_coords: Optional[np.ndarray] = None  # (n_features, 3) - voxel coordinates
    affine: Optional[np.ndarray] = None  # Affine transformation matrix
    
    def __post_init__(self):
        """Validate data consistency"""
        if self.data.shape[0] != len(self.labels):
            raise ValueError("Data and labels must have same number of samples")
        
        if self.runs is not None and len(self.runs) != len(self.labels):
            raise ValueError("Runs must have same length as labels")
        
        if self.splits is not None and len(self.splits) != len(self.labels):
            raise ValueError("Splits must have same length as labels")
        
        if self.delays is not None and len(self.delays) != len(self.labels):
            raise ValueError("Delays must have same length as labels")
        
    @property
    def n_samples(self):
        return self.data.shape[0]
    
    @property
    def n_features(self):
        return self.data.shape[1]
    
    @property
    def unique_labels(self):
        return np.unique(self.labels)
    
    def select_voxels(self, voxel_mask):
        """Return new dataset with selected voxels"""
        if isinstance(voxel_mask, np.ndarray) and voxel_mask.dtype == bool:
            # Boolean mask
            new_data = self.data[:, voxel_mask]
            new_coords = self.voxel_coords[voxel_mask] if self.voxel_coords is not None else None
        else:
            # Indices
            new_data = self.data[:, voxel_mask]
            new_coords = self.voxel_coords[voxel_mask] if self.voxel_coords is not None else None
        
        return MVPADataset(
            data=new_data,
            labels=self.labels.copy(),
            subject_id=self.subject_id,
            task=self.task,
            roi=self.roi,
            runs=self.runs.copy() if self.runs is not None else None,
            delays=self.delays.copy() if self.delays is not None else None,
            splits=self.splits.copy() if self.splits is not None else None,
            trial_indices=self.trial_indices.copy() if self.trial_indices is not None else None,
            voxel_coords=new_coords,
            affine=self.affine.copy() if self.affine is not None else None
        )
        
    def select_samples(self, sample_mask):
        """Return new dataset with selected samples"""
        return MVPADataset(
            data=self.data[sample_mask],
            labels=self.labels[sample_mask],
            subject_id=self.subject_id,
            task=self.task,
            roi=self.roi,
            runs=self.runs[sample_mask] if self.runs is not None else None,
            delays=self.delays[sample_mask] if self.delays is not None else None,
            splits=self.splits[sample_mask] if self.splits is not None else None,
            trial_indices=self.trial_indices[sample_mask] if self.trial_indices is not None else None,
            voxel_coords=self.voxel_coords,
            affine=self.affine
        )
        
    def filter_by_condition(self, conditions):
        """Return dataset with only specified conditions"""
        if isinstance(conditions, str):
            conditions = [conditions]
        
        mask = np.isin(self.labels, conditions)
        return self.select_samples(mask)
    
    def filter_by_delay(self, delay):
        if self.delays is not None:
            mask = self.delays==delay
            return self.select_samples(mask)
        else:
            raise ValueError("Dataset does not have delay information!")           
    
    def get_binary_data(self, positive_label, negative_label):
        """
        Get data for binary classification
        
        Returns:
        --------
        X : np.ndarray
            Data for the two conditions
        y : np.ndarray
            Binary labels (0 for negative, 1 for positive)
        """
        mask = np.isin(self.labels, [positive_label, negative_label])
        X = self.data[mask]
        y = (self.labels[mask] == positive_label).astype(int)
        return X, y
    
@dataclass
class BetaFileInfo:
    """Information about a beta's condition"""
    path: Path
    condition: str
    run: int
    shape: Optional[str] = None # 'wide' or 'narrow'
    view: Optional[str] = None # '30' or '90'
    congruency: Optional[str] = None # 'congruent' or 'incongruent'
    delay: Optional[int] = None # for FIR models

class BetaFilenameParser:
    def __init__(self):
        self.patterns = {
            # FIR pattern: beta_{condition}_delay-{delay}_run-{run}.nii
            'fir': re.compile(r'beta_(.+)_delay(\d+)_run(\d+)\.nii(?:\.gz)?$'),
            
            # Standard pattern: beta_{condition}_run-{run}.nii
            'standard': re.compile(r'beta_(.+)_run(\d+)\.nii(?:\.gz)?$')
        }
        
        # Condition component parsers
        self.shape_pattern = re.compile(r'(?:^|_)(wide|narrow)(?=_|\.|$)', re.IGNORECASE)
        self.view_pattern = re.compile(r'(?:^|_)(30|90)(?=_|\.|$)')
        self.congruency_pattern = re.compile(r'(?:^|_)(congruent|incongruent)(?=_|\.|$)', re.IGNORECASE)
    
    def parse_filename(self, filepath: Union[str, Path]) -> BetaFileInfo:
        """Parse beta filename to extract condition and run information"""
        filepath = Path(filepath)
        filename = filepath.name
        
        # Try different patterns
        for pattern_name, pattern in self.patterns.items():
            match = pattern.match(filename)
            if match:
                if pattern_name == 'fir':
                    condition = match.group(1)
                    delay = int(match.group(2))
                    run = int(match.group(3))
                else:
                    condition = match.group(1)
                    run = int(match.group(2))
                    delay = None
                    
                # Parse condition components
                shape = self._extract_shape(condition)
                view = self._extract_view(condition)
                congruency = self._extract_congruency(condition)
                
                return BetaFileInfo(
                    path=filepath,
                    condition=condition,
                    run=run,
                    shape=shape,
                    view=view,
                    congruency=congruency,
                    delay=delay
                )
        
        raise ValueError(f"Could not parse filename: {filename}")
    
    def _extract_shape(self, condition: str) -> Optional[str]:
        """Extract shape (wide/narrow) from condition string"""
        match = self.shape_pattern.search(condition)
        return match.group(1) if match else None
    
    def _extract_view(self, condition: str) -> Optional[str]:
        """Extract view (A/B) from condition string"""
        match = self.view_pattern.search(condition)
        return match.group(1) if match else None
    
    def _extract_congruency(self, condition: str) -> Optional[str]:
        """Extract congruency from condition string"""
        match = self.congruency_pattern.search(condition)
        return match.group(1) if match else None

class LocalizerLoader:
    """
    Loader for localizer contrast maps
    """
    
    def __init__(self, data_dir: Union[str, Path], 
                 task: str, model_name: str,
                 contrast_name: str):
        self.data_dir = Path(data_dir)
        self.task = task
        self.model_name = model_name
        self.contrast_name = contrast_name

    def load_localizer_values(self, exp_no: int, subject_id: str, roi_mask: np.ndarray) -> np.ndarray:
        """
        Load localizer contrast values within ROI
        """
        contrast_path = (self.data_dir / f'experiment_{exp_no}' / 'derivatives' / 'spm-preproc' / 
                         'derivatives' / 'spm-stats' / 'contrasts' / 
                         subject_id / self.task / self.model_name / f'con_{self.contrast_name}.nii')
        
        if not contrast_path.exists():
            raise FileNotFoundError(f"Localizer contrast not found: {contrast_path}")
        
        # Load and extract values within ROI
        contrast_img = nib.load(contrast_path)
        contrast_data = contrast_img.get_fdata()
        
        return contrast_data[roi_mask]

class BetaLoader:
    """
    Loader for beta maps from GLM results
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.parser = BetaFilenameParser()
        
    def load_glm_results(self, exp_no: int, subject_id: str, task: str, model_name: str,
                         roi_name: str, localizer: Optional[LocalizerLoader] = None,
                         n_voxels: Optional[int] = None, fir: bool = False) -> MVPADataset:
        """
        Load GLM results for a subject
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        task : str
            Task name ('train' or 'test')
        model_name : str
            GLM model name
        roi_name : str
            ROI name
        localizer : LocalizerLoader, optional
            Localizer for voxel selection
        n_voxels: int, optional
            How many top voxels to select from the localizer contrast.
            
        Returns:
        --------
        MVPADataset : Loaded dataset
        """
        # Construct paths
        exp_dir = f'experiment_{exp_no}'
        glm_path = self.data_dir / exp_dir / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas' / subject_id / task / model_name
        if fir: glm_path = glm_path / 'FIR'
        roi_path = self.data_dir / 'roi_masks' / f'{roi_name}.nii'
        
        beta_files = self._find_beta_files(glm_path)
        if not beta_files:
            raise FileNotFoundError(f"No beta files found in {glm_path}")
        
        # Parse filenames to get condition info
        file_info_list = []
        for beta_file in beta_files:
            try:
                file_info = self.parser.parse_filename(beta_file)
                file_info_list.append(file_info)
            except ValueError as e:
                logger.warning(f"Skipping file {beta_file}: {e}")
        
        if not file_info_list:
            raise ValueError("No valid beta files found")
        
        # Load beta maps and organize
        beta_maps, labels, runs, delays, file_paths = self._load_and_organize_betas(file_info_list)
        
        # Load ROI mask
        roi_mask = self._load_roi_mask(roi_path)
        
        # Extract ROI data
        roi_data = self._extract_roi_data(beta_maps, roi_mask)
        
        # Get voxel coordinates and affine
        voxel_coords = self._get_voxel_coordinates(roi_mask)
        affine = self._get_affine(file_paths[0])
        
        # Create runs array if multiple runs
        runs_array = np.array(runs) if len(set(runs)) > 1 else None
        
        delays_array = np.array(delays) if delays else None
        
        dataset = MVPADataset(
            data=roi_data,
            labels=np.array(labels),
            subject_id=subject_id,
            task=task,
            roi=roi_name,
            runs=runs_array,
            delays=delays_array,
            voxel_coords=voxel_coords,
            affine=affine
        )
        
        # Apply voxel selection if requested
        if localizer is not None and n_voxels is not None:
            dataset = self._select_top_voxels(
                dataset, roi_mask, localizer, n_voxels, exp_no, subject_id
            )
            
        return dataset
        
    def _find_beta_files(self, glm_path: Path) -> List[Path]:
        """Find all beta files in GLM directory"""
        patterns = ['beta_*.nii', 'beta_*.nii.gz']
        beta_files = []
        for pattern in patterns:
            beta_files.extend(list(glm_path.glob(pattern)))
        
        return sorted(beta_files)
    
    def _load_and_organize_betas(self, file_info_list: List[BetaFileInfo]) -> Tuple[np.ndarray, List[str], List[int], List[Path]]:
        """Load beta maps and organize by filename info"""
        beta_maps = []
        labels = []
        runs = []
        delays = []
        file_paths = []
        
        for file_info in sorted(file_info_list, key=lambda x: (x.condition, x.run)):
            # Load beta map
            img = nib.load(file_info.path)
            beta_maps.append(img.get_fdata())
            
            # Store condition and run info
            labels.append(file_info.condition)
            runs.append(file_info.run)
            delay = file_info.delay
            if delay:
                delays.append(delay)
            file_paths.append(file_info.path)
            
        return np.array(beta_maps), labels, runs, delays, file_paths
    
    def _load_roi_mask(self, roi_path: Path) -> np.ndarray:
        """Load ROI mask"""
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI mask not found: {roi_path}")
        
        roi_img = nib.load(roi_path)
        return roi_img.get_fdata() > 0
    
    def _extract_roi_data(self, beta_maps: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """Extract data within ROI"""
        n_betas = beta_maps.shape[0]
        n_voxels = np.sum(roi_mask)
        
        roi_data = np.zeros((n_betas, n_voxels))
        
        for i in range(n_betas):
            roi_data[i] = beta_maps[i][roi_mask]
        
        return roi_data
    
    def _select_top_voxels(self, dataset: MVPADataset, roi_mask: np.ndarray,
                           localizer: LocalizerLoader, n_voxels: int,
                           exp_no: int, subject_id: str) -> MVPADataset:
        """Apply voxel selection using localizer contrast"""
        
        # Load localizer values
        localizer_values = localizer.load_localizer_values(exp_no, subject_id, roi_mask)
        
        # Select top voxels
        if n_voxels >= len(localizer_values):
            return dataset # Use all voxels
        
        # Exclude NaN values before computing top_indices
        valid_mask = ~np.isnan(localizer_values)
        if np.sum(valid_mask) < n_voxels:
            n_voxels = np.sum(valid_mask) # Use all available valid voxels
            
        valid_indices = np.where(valid_mask)[0]
        valid_values = localizer_values[valid_mask]
        sorted_indices = np.argsort(valid_values)[-n_voxels:]
        top_indices = valid_indices[sorted_indices]
        
        voxel_mask = np.zeros(len(localizer_values), dtype=bool)
        voxel_mask[top_indices] = True
        
        return dataset.select_voxels(voxel_mask)
    
    def _get_voxel_coordinates(self, roi_mask: np.ndarray) -> np.ndarray:
        """Get voxel coordinates within ROI"""
        return np.array(np.where(roi_mask)).T
    
    def _get_affine(self, beta_path: Path) -> np.ndarray:
        """Get affine transformation from example beta file"""
        img = nib.load(beta_path)
        return img.affine
        

class ExperimentDataLoader:
    """
    High-level loader for experiment-specific data organization
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.beta_loader = BetaLoader(data_dir)
        self.parser = BetaFilenameParser()
        
    def load_experiment_1_data(self, subject_id: str, roi: str, 
                               localizer: Optional[LocalizerLoader] = None,
                               n_voxels: Optional[int] = None,
                               fir: bool = False) -> Tuple['MVPADataset', 'MVPADataset']:
        """
        Load data for Experiment 1: view-specific training, congruency, split in 3
        
        Returns:
        --------
        train_dataset, test_dataset : Tuple[MVPADataset, MVPADataset]
        """
        train_dataset = self.beta_loader.load_glm_results(
            exp_no=1, subject_id=subject_id, task='train', 
            model_name='exp1_viewspec_training', roi_name=roi,
            localizer=localizer, n_voxels=n_voxels
        )
        
        test_dataset = self.beta_loader.load_glm_results(
            exp_no=1, subject_id=subject_id, task='test', 
            model_name='exp1_full_model', roi_name=roi,
            localizer=localizer, n_voxels=n_voxels, fir=fir
        )
        
        return train_dataset, test_dataset
    
    def load_experiment_1_testonly(self, subject_id: str, roi: str, 
                               localizer: Optional[LocalizerLoader] = None,
                               n_voxels: Optional[int] = None,
                               fir: bool = False) -> MVPADataset:
        """
        Load data for Experiment 1, test only for info. coupling analysis
        
        Returns:
        --------
        test_dataset : MVPADataset
        """
        
        test_dataset = self.beta_loader.load_glm_results(
            exp_no=1, subject_id=subject_id, task='test', 
            model_name='exp1_full_model', roi_name=roi,
            localizer=localizer, n_voxels=n_voxels, fir=fir
        )
        
        return test_dataset
    
    def load_experiment_2_data(self, subject_id: str, roi: str, 
                               localizer: Optional[LocalizerLoader] = None,
                               n_voxels: Optional[int] = None) -> Tuple[MVPADataset, MVPADataset]:
        """
        Load data for Experiment 2: shape-based training, omission test
        
        Returns:
        --------
        train_dataset, test_dataset : Tuple[MVPADataset, MVPADataset]
        """
        train_dataset = self.beta_loader.load_glm_results(
            exp_no=2, subject_id=subject_id, task='train', 
            model_name='exp2_widenarr_training', roi_name=roi,
            localizer=localizer, n_voxels=n_voxels
        )
        test_dataset = self.beta_loader.load_glm_results(
            exp_no=2, subject_id=subject_id, task='test', 
            model_name='exp2_wide_narrow', roi_name=roi,
            localizer=localizer, n_voxels=n_voxels
        )
        
        return train_dataset, test_dataset
    
    def create_shape_labels(self, dataset: MVPADataset) -> np.ndarray:
        """
        Create binary shape labels (0=narrow, 1=wide) from condition names
        
        Parameters:
        -----------
        dataset : MVPADataset
            Dataset with condition labels
            
        Returns:
        --------
        np.ndarray : Binary shape labels
        """
        shape_labels = []
        
        for condition in dataset.labels:
            # Re-parse to extract shape
            shape = self.parser._extract_shape(condition)
            
            if shape == 'wide':
                shape_labels.append(1)
            elif shape == 'narrow':
                shape_labels.append(0)
            else:
                raise ValueError(f"Could not determine shape for condition: {condition}")
        
        return np.array(shape_labels)
    
    def split_by_view(self, dataset: MVPADataset) -> Tuple[MVPADataset, MVPADataset]:
        """
        Split dataset by view (30 and 90) based on condition names
        
        Parameters:
        -----------
        dataset : MVPADataset
            Input dataset
            
        Returns:
        --------
        view_30_dataset, view_90_dataset : Tuple[MVPADataset, MVPADataset]
        """
        view_30_mask = []
        view_90_mask = []
        
        for condition in dataset.labels:
            view = self.parser._extract_view(condition)
            
            if view == '30':
                view_30_mask.append(True)
                view_90_mask.append(False)
            elif view == '90':
                view_30_mask.append(False) 
                view_90_mask.append(True)
            else:
                raise ValueError(f"Could not determine view for condition: {condition}")
        
        view_30_mask = np.array(view_30_mask)
        view_90_mask = np.array(view_90_mask)
        
        view_30_dataset = dataset.select_samples(view_30_mask)
        view_90_dataset = dataset.select_samples(view_90_mask)
        
        return view_30_dataset, view_90_dataset
    
    def create_congruency_labels(self, dataset: 'MVPADataset') -> np.ndarray:
        """
        Create congruency labels from condition names
        
        Parameters:
        -----------
        dataset : MVPADataset
            Dataset with condition labels
            
        Returns:
        --------
        np.ndarray : Congruency labels ('congruent' or 'incongruent')
        """
        congruency_labels = []
        
        for condition in dataset.labels:
            congruency = self.parser._extract_congruency(condition)
            
            if congruency in ['congruent', 'incongruent']:
                congruency_labels.append(congruency)
            else:
                raise ValueError(f"Could not determine congruency for condition: {condition}")
        
        return np.array(congruency_labels)
    
    # BILATERAL SUPPORT METHODS
    
    def parse_roi_name(self, roi_name: str) -> Tuple[str, Optional[str]]:
        """Parse ROI name to extract base name and hemisphere"""
        if roi_name.endswith('_L') or roi_name.endswith('_R'):
            return roi_name[:-2], roi_name[-1]
        else:
            return roi_name, None

    def has_bilateral_roi(self, base_roi_name: str) -> bool:
        """Check if both L and R versions of an ROI exist"""
        roi_dir = self.data_dir / 'roi_masks'
        return (roi_dir / f'{base_roi_name}_L.nii').exists() and (roi_dir / f'{base_roi_name}_R.nii').exists()

    def load_bilateral_data(self, subject_id: str, base_roi_name: str, experiment: int,
                           localizer: Optional[LocalizerLoader] = None,
                           n_voxels: Optional[int] = None) -> Tuple[Tuple[MVPADataset, MVPADataset], 
                                                                   Tuple[MVPADataset, MVPADataset]]:
        """Load data for both hemispheres"""
        load_func = self.load_experiment_1_data if experiment == 1 else self.load_experiment_2_data
        
        # Load both hemispheres
        train_L, test_L = load_func(subject_id, f"{base_roi_name}_L", localizer, n_voxels)
        train_R, test_R = load_func(subject_id, f"{base_roi_name}_R", localizer, n_voxels)
        
        return (train_L, test_L), (train_R, test_R)