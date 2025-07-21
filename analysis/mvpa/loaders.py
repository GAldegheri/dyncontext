import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from pathlib import Path
import nibabel as nib
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MVPADataset:
    # Core data
    data: np.ndarray  # (n_samples, n_features) - the actual neural data
    labels: np.ndarray  # (n_samples,) - condition labels
    
    # Metadata
    subject_id: str
    task: str # 'train' or 'test'
    roi: str
    
    # Additional attributes for cross-validation and grouping
    runs: Optional[np.ndarray] = None  # (n_samples,) - run numbers for CV
    splits: Optional[np.ndarray] = None  # (n_samples,) - trial split numbers
    trial_indices: Optional[np.ndarray] = None  # (n_samples,) - original trial indices
    
    # Behavioral data (optional)
    behavior: Optional[pd.DataFrame] = None
    
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
            splits=self.splits.copy() if self.splits is not None else None,
            trial_indices=self.trial_indices.copy() if self.trial_indices is not None else None,
            behavior=self.behavior.copy() if self.behavior is not None else None,
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
            splits=self.splits[sample_mask] if self.splits is not None else None,
            trial_indices=self.trial_indices[sample_mask] if self.trial_indices is not None else None,
            behavior=self.behavior.iloc[sample_mask].reset_index(drop=True) if self.behavior is not None else None,
            voxel_coords=self.voxel_coords,
            affine=self.affine
        )
        
    def filter_by_condition(self, conditions):
        """Return dataset with only specified conditions"""
        if isinstance(conditions, str):
            conditions = [conditions]
        
        mask = np.isin(self.labels, conditions)
        return self.select_samples(mask)
    
    def filter_by_split(self, split_number):
        """Return dataset with only specified split"""
        if self.splits is None:
            raise ValueError("No split information available")
        
        mask = self.splits == split_number
        return self.select_samples(mask)
    
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
    

class BetaLoader:
    """
    Loader for beta maps from GLM results
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        
    def load_glm_results(self, subject_id: str, task: str, model_name: str,
                         roi_name: str, localizer_contrast: Optional[str] = None) -> MVPADataset:
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
        localizer_contrast : str
            Contrast used for voxel selection
            
        Returns:
        --------
        MVPADataset : Loaded dataset
        """
        # Construct paths
        glm_path = self.data_dir / subject_id / 'glm_results' / f'task-{task}_{model_name}'
        roi_path = self.data_dir /subject_id / 'roi_masks' / f'{roi_name}.nii'
        
        # Load beta maps
        beta_maps, conditions = self._load_beta_maps(glm_path)
        
        # Load ROI mask
        roi_mask = self._load_roi_mask(roi_path)
        
        # Extract ROI data
        roi_data = self._extract_roi_data(beta_maps, roi_mask)
        
        # Load behavioral data if available
        behavior_data = self._load_behavior_data(subject_id, task)
        
        # Create labels array
        labels = np.array(conditions)
        
        # Get voxel coordinates
        voxel_coords = self._get_voxel_coordinates(roi_mask)
        
        # Load affine transformation
        example_beta = nib.load(list(glm_path.glob('beta_*.nii.gz'))[0])
        affine = example_beta.affine
        
        return MVPADataset(
            data=roi_data,
            labels=labels,
            subject_id=subject_id,
            task=task,
            roi=roi_name,
            behavior=behavior_data,
            voxel_coords=voxel_coords,
            affine=affine
        )
        
    def _load_beta_maps(self, glm_path: Path) -> Tuple[np.ndarray, List[str]]:
        """Load beta maps and condition names"""
        beta_files = sorted(list(glm_path.glob('beta_*.nii')))
        conditions_file = glm_path / 'conditions.txt'
        
        if not beta_files:
            raise FileNotFoundError(f"No beta files found in {glm_path}")
        
        # Load conditions
        if conditions_file.exists():
            with open(conditions_file, 'r') as f:
                conditions = [line.strip() for line in f.readlines()]
        else:
            # Fallback: extract from filenames
            conditions = [f.stem.replace('beta_', '') for f in beta_files]
        
        # Load beta maps
        beta_maps = []
        for beta_file in beta_files:
            img = nib.load(beta_file)
            beta_maps.append(img.get_fdata())
        
        return np.array(beta_maps), conditions
    
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
    
    def _get_voxel_coordinates(self, roi_mask: np.ndarray) -> np.ndarray:
        """Get voxel coordinates within ROI"""
        return np.array(np.where(roi_mask)).T
    
    def _load_behavior_data(self, subject_id: str, task: str) -> Optional[pd.DataFrame]:
        """Load behavioral data if available"""
        behavior_file = self.data_dir / subject_id / 'behavior' / f'task-{task}_behavior.csv'
        
        if behavior_file.exists():
            return pd.read_csv(behavior_file)
        else:
            logger.warning(f"No behavioral data found for {subject_id}, task {task}")
            return None
        

class LocalizerLoader:
    """
    Loader for localizer contrasts used in voxel selection
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
    
    def load_localizer_map(self, subject_id: str, contrast_name: str, 
                           roi_mask: np.ndarray) -> np.ndarray:
        """
        Load localizer contrast map
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        contrast_name : str
            Contrast name (e.g., 'stimulus>baseline')
        roi_mask : np.ndarray
            ROI mask for extracting relevant voxels
            
        Returns:
        --------
        np.ndarray : Localizer values within ROI
        """
        localizer_path = self.data_dir / subject_id / 'glm_results' / 'localizer'
        contrast_file = localizer_path / f'con_{contrast_name.replace(">", "_gt_")}.nii.gz'
        
        if not contrast_file.exists():
            raise FileNotFoundError(f"Localizer contrast not found: {contrast_file}")
        
        contrast_img = nib.load(contrast_file)
        contrast_data = contrast_img.get_fdata()
        
        return contrast_data[roi_mask]


class VoxelSelector:
    """
    Handles voxel selection based on localizer contrasts
    """
    
    def __init__(self, localizer_values: np.ndarray):
        """
        Parameters:
        -----------
        localizer_values : np.ndarray
            Localizer contrast values for voxels in ROI
        """
        self.localizer_values = localizer_values
    
    def select_top_voxels(self, n_voxels: int) -> np.ndarray:
        """
        Select top N voxels based on localizer values
        
        Parameters:
        -----------
        n_voxels : int
            Number of voxels to select
            
        Returns:
        --------
        np.ndarray : Boolean mask for selected voxels
        """
        if n_voxels >= len(self.localizer_values):
            return np.ones(len(self.localizer_values), dtype=bool)
        
        # Get indices of top voxels
        top_indices = np.argsort(self.localizer_values)[-n_voxels:]
        
        # Create boolean mask
        voxel_mask = np.zeros(len(self.localizer_values), dtype=bool)
        voxel_mask[top_indices] = True
        
        return voxel_mask
    
    def get_voxel_range(self, min_voxels: int = 100, max_voxels: int = 6000, 
                        step: int = 100) -> List[int]:
        """Get range of voxel counts for systematic analysis"""
        max_available = len(self.localizer_values)
        actual_max = min(max_voxels, max_available)
        
        return list(range(min_voxels, actual_max + 1, step))
    

class TrialFilterDataLoader:
    """
    Loader that integrates with trial filters to create appropriate datasets
    """
    
    def __init__(self, beta_loader: BetaLoader):
        self.beta_loader = beta_loader
    
    def load_filtered_dataset(self, subject_id: str, task: str, model_name: str,
                              roi_name: str, trial_filter, 
                              behavior_data: Optional[pd.DataFrame] = None) -> MVPADataset:
        """
        Load dataset filtered by trial filter
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        task : str
            Task name
        model_name : str
            GLM model name
        roi_name : str
            ROI name
        trial_filter : TrialFilter
            Trial filter to apply
        behavior_data : pd.DataFrame, optional
            Behavioral data (if not provided, will try to load)
            
        Returns:
        --------
        MVPADataset : Filtered dataset with appropriate labels
        """
        # Load base dataset
        dataset = self.beta_loader.load_glm_results(subject_id, task, model_name, roi_name)
        
        # Use provided behavior data or dataset's behavior data
        if behavior_data is not None:
            behavior = behavior_data
        elif dataset.behavior is not None:
            behavior = dataset.behavior
        else:
            raise ValueError("No behavioral data available for trial filtering")
        
        # Apply trial filter
        trial_assignments = trial_filter.filter_trials(behavior)
        
        # Create new labels and data based on trial assignments
        new_data_list = []
        new_labels_list = []
        new_trial_indices_list = []
        
        for condition, trial_indices in trial_assignments.items():
            if len(trial_indices) > 0:
                condition_data = dataset.data[trial_indices]
                condition_labels = [condition] * len(trial_indices)
                
                new_data_list.append(condition_data)
                new_labels_list.extend(condition_labels)
                new_trial_indices_list.extend(trial_indices)
        
        if not new_data_list:
            raise ValueError("No trials selected by filter")
        
        # Combine data
        new_data = np.vstack(new_data_list)
        new_labels = np.array(new_labels_list)
        new_trial_indices = np.array(new_trial_indices_list)
        
        # Create new dataset
        return MVPADataset(
            data=new_data,
            labels=new_labels,
            subject_id=subject_id,
            task=task,
            roi=roi_name,
            trial_indices=new_trial_indices,
            behavior=behavior,
            voxel_coords=dataset.voxel_coords,
            affine=dataset.affine
        )
        
# Convenience functions

def load_experiment_data(subject_id: str, data_dir: Union[str, Path], 
                         experiment: int, roi: str) -> Tuple[MVPADataset, MVPADataset]:
    """
    Convenience function to load train and test data for an experiment
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    data_dir : str or Path
        Data directory
    experiment : int
        Experiment number (1 or 2)
    roi : str
        ROI name
    
    Returns:
    --------
    train_dataset, test_dataset : Tuple[MVPADataset, MVPADataset]
        Training and test datasets
    """
    loader = BetaLoader(data_dir)
    
    if experiment == 1:
        # Experiment 1: view-specific training, congruency-based test
        train_model = 'view_specific_miniblocks'
        test_model = 'wide_narrow_congruency_splits'
    elif experiment == 2:
        # Experiment 2: shape-based training, omission test
        train_model = 'wide_narrow_training'
        test_model = 'wide_narrow'
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")
    
    train_dataset = loader.load_glm_results(subject_id, 'train', train_model, roi)
    test_dataset = loader.load_glm_results(subject_id, 'test', test_model, roi)
    
    return train_dataset, test_dataset


def load_with_voxel_selection(subject_id: str, data_dir: Union[str, Path],
                              task: str, model_name: str, roi: str,
                              n_voxels: int, localizer_contrast: str = 'stimulus>baseline') -> MVPADataset:
    """
    Load dataset with automatic voxel selection
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    data_dir : str or Path
        Data directory
    task : str
        Task name ('train' or 'test')
    model_name : str
        GLM model name
    roi : str
        ROI name
    n_voxels : int
        Number of voxels to select
    localizer_contrast : str
        Localizer contrast for voxel selection
        
    Returns:
    --------
    MVPADataset : Dataset with selected voxels
    """
    # Load full dataset
    loader = BetaLoader(data_dir)
    dataset = loader.load_glm_results(subject_id, task, model_name, roi)
    
    # Load localizer
    localizer_loader = LocalizerLoader(data_dir)
    roi_mask_full = loader._load_roi_mask(
        Path(data_dir) / subject_id / 'roi_masks' / f'{roi}.nii.gz'
    )
    localizer_values = localizer_loader.load_localizer_map(
        subject_id, localizer_contrast, roi_mask_full
    )
    
    # Select voxels
    voxel_selector = VoxelSelector(localizer_values)
    voxel_mask = voxel_selector.select_top_voxels(n_voxels)
    
    # Return dataset with selected voxels
    return dataset.select_voxels(voxel_mask)


def create_binary_dataset(dataset: MVPADataset, positive_condition: str,
                          negative_condition: str) -> MVPADataset:
    """
    Create binary dataset from multi-condition dataset
    
    Parameters:
    -----------
    dataset : MVPADataset
        Input dataset
    positive_condition : str
        Condition to label as positive (1)
    negative_condition : str
        Condition to label as negative (0)
        
    Returns:
    --------
    MVPADataset : Binary dataset
    """
    
    # Filter to only include the two conditions
    mask = np.isin(dataset.labels, [positive_condition, negative_condition])
    filtered_dataset = dataset.select_samples(mask)
    
    # Create binary labels
    binary_labels = (filtered_dataset.labels == positive_condition).astype(int)
    
    # Create new dataset with binary labels
    return MVPADataset(
        data=filtered_dataset.data,
        labels=binary_labels,
        subject_id=filtered_dataset.subject_id,
        task=filtered_dataset.task,
        roi=filtered_dataset.roi,
        runs=filtered_dataset.runs,
        splits=filtered_dataset.splits,
        trial_indices=filtered_dataset.trial_indices,
        behavior=filtered_dataset.behavior,
        voxel_coords=filtered_dataset.voxel_coords,
        affine=filtered_dataset.affine
    )
    

def split_by_view(dataset: MVPADataset) -> Tuple[MVPADataset, MVPADataset]:
    """
    Split dataset by view (A vs B) based on behavior data
    
    Parameters:
    -----------
    dataset : MVPADataset
        Input dataset with behavior data
        
    Returns:
    --------
    view_a_dataset, view_b_dataset : Tuple[MVPADataset, MVPADataset]
        Datasets for view A and view B
    """
    if dataset.behavior is None:
        raise ValueError("Behavior data required for view splitting")
    
    # Assuming 'initpos' column indicates view (1=A, 2=B)
    if 'initpos' not in dataset.behavior.columns:
        raise ValueError("'initpos' column not found in behavior data")
    
    view_a_mask = dataset.behavior['initpos'] == 1
    view_b_mask = dataset.behavior['initpos'] == 2
    
    view_a_dataset = dataset.select_samples(view_a_mask.values)
    view_b_dataset = dataset.select_samples(view_b_mask.values)
    
    return view_a_dataset, view_b_dataset