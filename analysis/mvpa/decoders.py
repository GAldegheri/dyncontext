import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import re

from analysis.mvpa.classifiers import (
    TrainTestClassifier, CrossValidationClassifier, ViewSpecificClassifier,
    MetricType
)
from analysis.mvpa.loaders import (
    MVPADataset, LocalizerLoader, ExperimentDataLoader
)
from analysis.mvpa.utils import remove_nan_voxels

logger = logging.getLogger(__name__)

class Experiment1Decoder:
    """
    Decoder for Experiment 1: View-specific cross-decoding with congruency analysis
    """
    def __init__(self, data_dir: Union[str, Path], roi: str, metric: MetricType = 'classifier_information'):
        self.data_dir = Path(data_dir)
        self.roi = roi
        self.metric = metric
        self.voxel_counts = list(range(100, 6100, 100))
        
        # Create data loader and localizer
        self.experiment_loader = ExperimentDataLoader(self.data_dir)
        self.localizer = LocalizerLoader(self.data_dir, task='funcloc',
                                         model_name='exp1_objscr_baseline',
                                         contrast_name='objscr-vs-baseline')
        
    def run_complete_analysis(self, subject_id) -> pd.DataFrame:
        """
        Run the complete Experiment 1 analysis across all voxel counts
        """
        results = []
        
        # Parse ROI name to check if bilateral
        base_roi_name, hemisphere = self.experiment_loader.parse_roi_name(self.roi)
        
        if hemisphere is None and self.experiment_loader.has_bilateral_roi(base_roi_name):
            # Bilateral analysis
        
            for n_voxels in self.voxel_counts:
                logger.info(f"Running analysis with {n_voxels} voxels")
                
                # Load data with voxel selection
                train_dataset, test_dataset = self.experiment_loader.load_experiment_1_data(
                    subject_id, self.roi, localizer=self.localizer, n_voxels=n_voxels
                )
                
                # Remove NaN voxels
                train_clean, test_clean = remove_nan_voxels(train_dataset, test_dataset)
                
                # Run view-specific cross-decoding
                result = self._run_view_specific_analysis(train_clean, test_clean)
                result['n_voxels'] = n_voxels
                result['roi'] = self.roi
                result['subject_id'] = self.subject_id
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def _run_view_specific_analysis(self, train_dataset: MVPADataset,
                                    test_dataset: MVPADataset) -> Dict:
        """
        Run view-specific cross-decoding analysis
        """
        # Split training data by view (30 and 90)
        train_view_30, train_view_90 = self.experiment_loader.split_by_view(train_dataset)
        
        # Prepare training data for each background
        train_backgrounds = self._prepare_training_backgrounds(train_view_30, train_view_90)
        
        # Prepare test data by condition
        test_data = self._prepare_test_data(test_dataset, self.experiment_loader)
        
        # Run background-matched decoding for each test condition
        results = {}
        
        # Analyze congruent trials (3 splits)
        congruent_results = []
        for split in [1, 2, 3]:
            split_result = self._decode_condition_with_backgrounds(
                train_backgrounds, test_data, 'congruent', split
            )
            if not np.isnan(split_result['mean_score']):
                congruent_results.append(split_result['mean_score'])
        
        if congruent_results:
            results['congruent_accuracy'] = np.mean(congruent_results)
            results['congruent_std'] = np.std(congruent_results)
        else:
            results['congruent_accuracy'] = np.nan
            results['congruent_std'] = np.nan
        
        # Analyze incongruent trials
        incongruent_result = self._decode_condition_with_backgrounds(
            train_backgrounds, test_data, 'incongruent', None
        )
        results['incongruent_accuracy'] = incongruent_result['mean_score']
        
        # Compute difference
        if not (np.isnan(results['congruent_accuracy']) or np.isnan(results['incongruent_accuracy'])):
            results['congruent_minus_incongruent'] = (
                results['congruent_accuracy'] - results['incongruent_accuracy']
            )
        else:
            results['congruent_minus_incongruent'] = np.nan
        
        return results
    
    def _prepare_training_backgrounds(self, train_view_30: MVPADataset,
                                      train_view_90: MVPADataset) -> Dict:
        """
        Prepare training data for background-matched decoding
        """
        backgrounds = {}
        
        try:
            # 30° background: wide 30 vs narrow 30
            if train_view_30.n_samples > 0:
                shape_labels_30 = self.experiment_loader.create_shape_labels(train_view_30)
                backgrounds['30deg'] = (train_view_30.data, shape_labels_30)
                
            # 90° background: wide 90 vs narrow 90
            if train_view_90.n_samples > 0:
                shape_labels_90 = self.experiment_loader.create_shape_labels(train_view_90)
                backgrounds['90deg'] = (train_view_90.data, shape_labels_90)
        
        except Exception as e:
            logger.warning(f"Error preparing training backgrounds: {e}")
        
        return backgrounds
    
    def _prepare_test_data(self, test_dataset: MVPADataset) -> Dict:
        """
        Prepare test data organized by condition and split
        """
        test_data = {}
        
        # Get congruency labels
        try:
            congruency_labels = self.experiment_loader.create_congruency_labels(test_dataset)
            shape_labels = self.experiment_loader.create_shape_labels(test_dataset)
            
            # Group by congruency and split
            for i, (condition, congruency, shape) in enumerate(zip(
                test_dataset.labels, congruency_labels, shape_labels
            )):
                # Extract split information from condition name if available
                split = None
                if 'split' in condition.lower():
                    split_match = re.search(r'split(\d+)', condition)
                    if split_match:
                        split = int(split_match.group(1))
                        
                key = (congruency, split)
                if key not in test_data:
                    test_data[key] = {'data': [], 'labels': []}
                    
                test_data[key]['data'].append(test_dataset.data[i:i+1])
                test_data[key]['labels'].append(shape)
                
            # Combine data for each key
            for key in test_data:
                if test_data[key]['data']:
                    test_data[key]['data'] = np.vstack(test_data[key]['data'])
                    test_data[key]['labels'] = np.array(test_data[key]['labels'])
                    
        except Exception as e:
            logger.warning(f"Error preparing test data: {e}")
        
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
        
        
class Experiment2Decoder:
    """
    Decoder for Experiment 2: Cross-decoding from visible to occluded objects
    """
    def __init__(self, data_dir: Union[str, Path], roi: str, metric: MetricType = 'classifier_information'):
        self.data_dir = Path(data_dir)
        self.roi = roi
        self.metric = metric
        self.voxel_counts = list(range(100, 6100, 100))
        
        # Create data loader and localizer
        self.experiment_loader = ExperimentDataLoader(self.data_dir)
        self.localizer = LocalizerLoader(self.data_dir, task='funcloc',
                                         model_name='exp2_objscr_baseline',
                                         contrast_name='objscr-vs-baseline')
        
    def run_complete_analysis(self, subject_id) -> pd.DataFrame:
        """Run the complete Experiment 2 analysis"""
        results = []
        
        for n_voxels in self.voxel_counts:
            logger.info(f"Running analysis with {n_voxels} voxels")
            
            # Load data with voxel selection
            train_dataset, test_dataset = self.experiment_loader.load_experiment_2_data(
                subject_id, self.roi, localizer=self.localizer, n_voxels=n_voxels
            )
            
            # Remove NaN voxels
            train_clean, test_clean = remove_nan_voxels(train_dataset, test_dataset)
            
            # Run cross-decoding
            result = self._run_cross_decoding(train_clean, test_clean)
            result['n_voxels'] = n_voxels
            result['roi'] = self.roi
            result['subject_id'] = subject_id
            
            results.append(result)
            
        return pd.DataFrame(results)        
    
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