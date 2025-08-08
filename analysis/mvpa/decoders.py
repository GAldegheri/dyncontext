import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

from analysis.mvpa.base import MVPAAnalysis
from analysis.mvpa.classifiers import (
    TrainTestClassifier, CrossValidationClassifier, ViewSpecificClassifier,
    SplitTrialsClassifier, MetricType
)
from analysis.mvpa.loaders import (
    MVPADataset, BetaLoader, LocalizerLoader, VoxelSelector,
    load_experiment_data, load_with_voxel_selection, create_binary_dataset,
    split_by_view
)
from models.trial_filters import (
    WideNarrowFilter, CongruencyFilter, WideNarrowCongruencyFilter,
    ViewSpecificFilter, TrialSplitter
)

logger = logging.getLogger(__name__)

class Experiment1Decoder(MVPAAnalysis):
    """
    Decoder for Experiment 1: View-specific cross-decoding with congruency analysis
    """
    def __init__(self, roi: str, metric: MetricType = 'classifier_information'):
        super().__init__(
            name="experiment1_decoder",
            description="Experiment 1: View-specific wide vs narrow with congruency",
            roi=roi
        )
        self.metric = metric
        self.voxel_counts = list(range(100, 6100, 100))
        
    def load_data(self, subject_id: str, data_dir: Union[str, Path]):
        """Load training and test data for Experiment 1"""
        self.subject_id = subject_id
        self.data_dir = Path(data_dir)
        
        # Load datasets
        self.train_dataset, self.test_dataset = load_experiment_data(
            subject_id, data_dir, experiment=1, roi=self.roi
        )
        
        self.localizer_loader = LocalizerLoader(data_dir)
        
        logger.info(f"Loaded Experiment 1 data for {subject_id}")
        logger.info(f"Train: {self.train_dataset.n_samples} samples, {self.train_dataset.n_features} features")
        logger.info(f"Test: {self.test_dataset.n_samples} samples, {self.test_dataset.n_features} features")
        
    def run_complete_analysis(self) -> pd.DataFrame:
        """
        Run the complete Experiment 1 analysis across all voxel counts
        """
        results = []
        
        for n_voxels in self.voxel_counts:
            logger.info(f"Running analysis with {n_voxels} voxels")
            
            # Select voxels
            train_selected = self._select_voxels(self.train_dataset, n_voxels)
            test_selected = self._select_voxels(self.test_dataset, n_voxels)
            
            # Run view-specific cross-decoding
            result = self._run_view_specific_analysis(train_selected, test_selected)
            result['n_voxels'] = n_voxels
            result['roi'] = self.roi
            result['subject_id'] = self.subject_id
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _select_voxels(self, dataset: MVPADataset, n_voxels: int) -> MVPADataset:
        """Select top N voxels based on localizer"""
        # Load localizer data for this ROI
        roi_mask_path = self.data_dir / self.subject_id / 'roi_masks' / f'{self.roi}.nii.gz'
        loader = BetaLoader(self.data_dir)
        roi_mask = loader._load_roi_mask(roi_mask_path)
        
        localizer_values = self.localizer_loader.load_localizer_map(
            self.subject_id, 'stimulus>baseline', roi_mask
        )
        
        # Select voxels
        voxel_selector = VoxelSelector(localizer_values)
        voxel_mask = voxel_selector.select_top_voxels(n_voxels)
        
        return dataset.select_voxels(voxel_mask)
    
    def _run_view_specific_analysis(self, train_dataset: MVPADataset,
                                    test_dataset: MVPADataset) -> Dict:
        """
        Run view-specific cross-decoding analysis
        """
        # Split training data by view (A and B)
        train_view_a, train_view_b = split_by_view(train_dataset)
        
        # Prepare training data for each background
        train_backgrounds = self._prepare_training_backgrounds(train_view_a, train_view_b)
        
        # Prepare test data by condition
        test_data = self._prepare_test_data(test_dataset)
        
        # Run background-matched decoding for each test condition
        results = {}
        
        # Analyze congruent trials (3 splits)
        congruent_results = []
        for split in [1, 2, 3]:
            split_result = self._decode_condition_with_backgrounds(
                train_backgrounds, test_data, 'congruent', split
            )
            congruent_results.append(split_result['mean_score'])
        
        results['congruent_accuracy'] = np.mean(congruent_results)
        results['congruent_std'] = np.std(congruent_results)
        
        # Analyze incongruent trials
        incongruent_result = self._decode_condition_with_backgrounds(
            train_backgrounds, test_data, 'incongruent', None
        )
        results['incongruent_accuracy'] = incongruent_result['mean_score']
        
        # Compute difference
        results['congruent_minus_incongruent'] = (
            results['congruent_accuracy'] - results['incongruent_accuracy']
        )
        
        return results
    
    def _prepare_training_backgrounds(self, train_view_a: MVPADataset,
                                      train_view_b: MVPADataset) -> Dict:
        """
        Prepare training data for background-matched decoding
        
        Returns dict with '30deg' and '90deg' backgrounds, each containing (X, y) tuples
        """
        backgrounds = {}
        
        # 30° background: A30 (wide) vs B30 (narrow)
        a30_data = self._extract_condition_data(train_view_a, 'A_wide') # A30 = wide
        b30_data = self._extract_condition_data(train_view_b, 'B_narrow') # B30 = narrow
        
        if a30_data is not None and b30_data is not None:
            X_30 = np.vstack([a30_data, b30_data])
            y_30 = np.concatenate([
                np.ones(len(a30_data)), # wide = 1
                np.zeros(len(b30_data)) # narrow = 0
            ])
            backgrounds['30deg'] = (X_30, y_30)
            
        # 90° background: A90 (narrow) vs B90 (wide)
        a90_data = self._extract_condition_data(train_view_a, 'A_narrow') # A90 = narrow
        b90_data = self._extract_condition_data(train_view_b, 'B_wide') # B90 = wide
        
        if a90_data is not None and b90_data is not None:
            X_90 = np.vstack([a90_data, b90_data])
            y_90 = np.concatenate([
                np.zeros(len(a90_data)), # narrow = 0
                np.ones(len(b90_data)) # wide = 1
            ])
            backgrounds['90deg'] = (X_90, y_90)
            
        return backgrounds
    
    def _extract_condition_data(self, dataset: MVPADataset, condition: str) -> Optional[np.ndarray]:
        """Extract data for a specific condition"""
        condition_mask = dataset.labels == condition
        if not np.any(condition_mask):
            logger.warning(f"No data found for condition: {condition}")
            return None
        
        return dataset.data[condition_mask]
    
    def _prepare_test_data(self, test_dataset: MVPADataset) -> Dict:
        """
        Prepare test data organized by condition and split
        """
        test_data = {}
        
        for condition in test_dataset.unique_labels:
            # Parse condition (e.g., 'wide_congruent_1', 'narrow_incongruent')
            parts = condition.split('_')
            shape = parts[0]
            congruency = parts[1]
            split = int(parts[2]) if len(parts) > 2 else None
            
            key = (congruency, split)
            if key not in test_data:
                test_data[key] = {'data': [], 'labels': []}
                
            # Get data for this condition
            condition_mask = test_dataset.labels == condition
            condition_data = test_dataset.data[condition_mask]
            condition_labels = np.ones(len(condition_data)) if shape == 'wide' else np.zeros(len(condition_data))
            
            test_data[key]['data'].append(condition_data)
            test_data[key]['labels'].append(condition_labels)
        
        # Combine data for each key
        for key in test_data:
            test_data[key]['data'] = np.vstack(test_data[key]['data'])
            test_data[key]['labels'] = np.concatenate(test_data[key]['labels'])
        
        return test_data
    
    def _decode_condition_with_backgrounds(self, train_backgrounds: Dict,
                                           test_data: Dict, congruency: str,
                                           split: Optional[int]) -> Dict:
        """Decode specific condition across backgrounds"""
        # Get test data for this condition
        test_key = (congruency, split)
        if test_key not in test_data:
            logger.warning(f"No test data for {congruency}, split {split}")
            return {'mean_score': np.nan}
        
        test_X, test_y = test_data[test_key]['data'], test_data[test_key]['labels']
        
        # Prepare test backgrounds (same data for both backgrounds)
        test_backgrounds = {bg: (test_X, test_y) for bg in train_backgrounds.keys()}
        
        # Run background-matched decoding
        view_classifier = ViewSpecificClassifier()
        
        return view_classifier.background_matched_decode(
            train_backgrounds, test_backgrounds, metric=self.metric
        )
        
class Experiment2Decoder(MVPAAnalysis):
    """
    Decoder for Experiment 2: Cross-decoding from visible to occluded objects
    """
    def __init__(self, roi: str, metric: MetricType = 'classifier_information'):
        super().__init__(
            name="experiment2_decoder", 
            description="Experiment 2: Cross-decoding to occluded objects",
            roi=roi
        )
        self.metric = metric
        self.voxel_counts = list(range(100, 6100, 100))
        
    def load_data(self, subject_id: str, data_dir: Union[str, Path]):
        """Load training and test data for Experiment 2"""
        self.subject_id = subject_id
        self.data_dir = Path(data_dir)
        
        # Load datasets
        self.train_dataset, self.test_dataset = load_experiment_data(
            subject_id, data_dir, experiment=2, roi=self.roi
        )
        
        # Load localizer
        self.localizer_loader = LocalizerLoader(data_dir)
        
        logger.info(f"Loaded Experiment 2 data for {subject_id}")
        
    def run_complete_analysis(self) -> pd.DataFrame:
        """Run the complete Experiment 2 analysis"""
        results = []
        
        for n_voxels in self.voxel_counts:
            logger.info(f"Running analysis with {n_voxels} voxels")
            
            # Select voxels
            train_selected = self._select_voxels(self.train_dataset, n_voxels)
            test_selected = self._select_voxels(self.test_dataset, n_voxels)
            
            # Run cross-decoding
            result = self._run_cross_decoding(train_selected, test_selected)
            result['n_voxels'] = n_voxels
            result['roi'] = self.roi
            result['subject_id'] = self.subject_id
            
            results.append(result)
            
        return pd.DataFrame(results)
    
    def _select_voxels(self, dataset: MVPADataset, n_voxels: int) -> MVPADataset:
        """Select top N voxels based on localizer"""
        roi_mask_path = self.data_dir / self.subject_id / 'roi_masks' / f'{self.roi}.nii'
        loader = BetaLoader(self.data_dir)
        roi_mask = loader._load_roi_mask(roi_mask_path)
        
        localizer_values = self.localizer_loader.load_localizer_map(
            self.subject_id, 'stimulus>baseline', roi_mask
        )
        
        voxel_selector = VoxelSelector(localizer_values)
        voxel_mask = voxel_selector.select_top_voxels(n_voxels)
        
        return dataset.select_voxels(voxel_mask)
    
    def _run_cross_decoding(self, train_dataset: MVPADataset, 
                            test_dataset: MVPADataset) -> Dict:
        """
        Run cross-decoding from visible objects to occluded objects
        """
        # Prepare training data (visible objects: wide vs narrow)
        train_X, train_y = self._prepare_training_data(train_dataset)
        
        # Prepare test data (occluded objects, omission trials only)
        test_X, test_y = self._prepare_test_data(test_dataset)
        
        if train_X is None or test_X is None:
            return {'accuracy': np.nan}
        
        cross_decoder = TrainTestClassifier()
        result = cross_decoder.train_test(train_X, train_y,
                                          test_X, test_y,
                                          metric=self.metric)
        
        return result