import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import re

from analysis.mvpa.classifiers import (
    TrainTestClassifier, CrossValidationClassifier, ViewSpecificClassifier
)
from analysis.mvpa.loaders import (
    MVPADataset, LocalizerLoader, ExperimentDataLoader
)
from analysis.mvpa.utils import remove_nan_voxels
import ipdb

logger = logging.getLogger(__name__)
class Experiment1Decoder:
    """
    Decoder for Experiment 1: View-specific cross-decoding with congruency analysis
    """
    def __init__(self, data_dir: Union[str, Path], roi: str,
                 voxel_counts: List[int] = list(range(100, 6100, 100))):
        self.data_dir = Path(data_dir)
        self.roi = roi
        self.voxel_counts = voxel_counts
        
        # Create data loader
        self.experiment_loader = ExperimentDataLoader(self.data_dir)
        self.localizer = LocalizerLoader(self.data_dir, task='funcloc',
                                         model_name='exp1_objscr_baseline',
                                         contrast_name='objscr-vs-baseline')
        
    def run_complete_analysis(self, subject_id: str) -> pd.DataFrame:
        """
        Run the complete Experiment 1 analysis
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
            
        Returns:
        --------
        pd.DataFrame with all analysis results
        """
        all_results = []
        
        # Parse ROI name to check if bilateral
        base_roi_name, hemisphere = self.experiment_loader.parse_roi_name(self.roi)
        
        for n_voxels in self.voxel_counts:
            
            if hemisphere is None and self.experiment_loader.has_bilateral_roi(base_roi_name):
                # Bilateral analysis - process each hemisphere separately
                (train_L, test_L), (train_R, test_R) = self.experiment_loader.load_bilateral_data(
                    subject_id, base_roi_name, experiment=1,
                    localizer=self.localizer, n_voxels=n_voxels
                )
                
                # Process left hemisphere
                try:
                    train_clean_L, test_clean_L = remove_nan_voxels(train_L, test_L)
                    result_df_L = self._run_view_specific_analysis(train_clean_L, test_clean_L)
                    result_df_L['hemisphere'] = 'L'
                    result_df_L['n_voxels'] = n_voxels
                    result_df_L['roi'] = base_roi_name
                    result_df_L['subject_id'] = subject_id
                    all_results.append(result_df_L)
                except Exception as e:
                    logger.warning(f"Left hemisphere analysis failed for {n_voxels} voxels: {e}")
                
                # Process right hemisphere
                try:
                    train_clean_R, test_clean_R = remove_nan_voxels(train_R, test_R)
                    result_df_R = self._run_view_specific_analysis(train_clean_R, test_clean_R)
                    result_df_R['hemisphere'] = 'R'
                    result_df_R['n_voxels'] = n_voxels
                    result_df_R['roi'] = base_roi_name
                    result_df_R['subject_id'] = subject_id
                    all_results.append(result_df_R)
                except Exception as e:
                    logger.warning(f"Right hemisphere analysis failed for {n_voxels} voxels: {e}")
                    
            else:
                # Unilateral analysis (or specific hemisphere requested)
                train_dataset, test_dataset = self.experiment_loader.load_experiment_1_data(
                    subject_id, self.roi, localizer=self.localizer, n_voxels=n_voxels
                )
                
                try:
                    train_clean, test_clean = remove_nan_voxels(train_dataset, test_dataset)
                    result_df = self._run_view_specific_analysis(train_clean, test_clean)
                    result_df['n_voxels'] = n_voxels
                    result_df['roi'] = self.roi
                    result_df['subject_id'] = subject_id
                    
                    # Add hemisphere info if specific hemisphere was requested
                    if hemisphere:
                        result_df['hemisphere'] = hemisphere
                        
                    all_results.append(result_df)
                except Exception as e:
                    logger.warning(f"Analysis failed for {n_voxels} voxels: {e}")
                    
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def _run_view_specific_analysis(self, train_dataset: MVPADataset,
                                    test_dataset: MVPADataset) -> pd.DataFrame:
        """
        Run view-specific cross-decoding analysis
        """
        
        # Run forward direction (train on training runs, test on main task by congruency)
        forward_results = self._analyze_by_congruency(
            base_dataset=train_dataset,
            congruency_dataset=test_dataset,
            direction='forward'
        )
        
        # Run backward direction (train on main task by congruency, test on training runs)
        backward_results = self._analyze_by_congruency(
            base_dataset=train_dataset,
            congruency_dataset=test_dataset,
            direction='backward'
        )
        
        # Combine results
        forward_results['direction'] = 'forward'
        backward_results['direction'] = 'backward'
        
        return pd.concat([forward_results, backward_results], ignore_index=True)
    
    def _analyze_by_congruency(self, base_dataset: MVPADataset,
                               congruency_dataset: MVPADataset,
                               direction: str = 'forward') -> pd.DataFrame:
        """
        Analyze decoding performance by congruency conditions.
        """
        
        # Get congruency conditions from the congruency dataset
        congruency_conditions = self._get_congruency_conditions(congruency_dataset)
        
        all_results = []
        
        for condition_label, condition_names in congruency_conditions:
            
            condition_dataset = congruency_dataset.filter_by_condition(condition_names)
            
            if condition_dataset.n_samples == 0:
                continue
            
            # Prepare data for background-matched decoding
            if direction == 'forward':
                # Forward: train on base (training runs), test on condition
                train_backgrounds = self._create_view_backgrounds(base_dataset)
                test_backgrounds = self._create_view_backgrounds(condition_dataset)
            else:
                # Backward: train on condition, test on base (training runs)
                train_backgrounds = self._create_view_backgrounds(condition_dataset)
                test_backgrounds = self._create_view_backgrounds(base_dataset)
            
            # Run background-matched decoding
            view_classifier = ViewSpecificClassifier()
            result_df = view_classifier.background_matched_decode(
                train_backgrounds, test_backgrounds
            )
            
            # Parse condition label to extract congruency and split info
            if 'incongruent' in condition_label:
                congruency = 'incongruent'
                split = None
            elif 'congruent_' in condition_label:
                congruency = 'congruent'
                split_part = condition_label.split('_')[1]
                split = None if split_part == 'no_split' else int(split_part)
            
            # Add metadata
            result_df['congruency'] = congruency
            result_df['split'] = split
            
            all_results.append(result_df)
            
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def _create_view_backgrounds(self, dataset: MVPADataset) -> Dict:
        """
        Create background-specific training/test data from any dataset.
        This method leverages existing dataloader methods.
        
        Parameters:
        -----------
        dataset : MVPADataset
            Input dataset to split by view
            
        Returns:
        --------
        Dict with keys '30deg' and '90deg', values are (X, y) tuples
        """
        backgrounds = {}
        
        # Use existing split_by_view method from experiment_loader
        view_30, view_90 = self.experiment_loader.split_by_view(dataset)
        
        # Create background data for 30° view using existing create_shape_labels
        if view_30.n_samples > 0:
            shape_labels_30 = self.experiment_loader.create_shape_labels(view_30)
            backgrounds['30deg'] = (view_30.data, shape_labels_30)
        
        # Create background data for 90° view using existing create_shape_labels
        if view_90.n_samples > 0:
            shape_labels_90 = self.experiment_loader.create_shape_labels(view_90)
            backgrounds['90deg'] = (view_90.data, shape_labels_90)
            
        return backgrounds
    
    def _get_congruency_conditions(self, dataset: MVPADataset) -> List[Tuple[str, List[str]]]:
        """
        Get unique congruency condition groups from dataset.
        
        Returns:
        --------
        List of tuples: (congruency_type, list_of_condition_labels)
        """
        # Group conditions by their congruency type
        congruent_conditions = []
        incongruent_conditions = []
        
        for label in dataset.unique_labels:
            if 'incongruent' in label.lower():
                incongruent_conditions.append(label)
            elif 'congruent' in label.lower():
                congruent_conditions.append(label)
        
        conditions = []
        
        # Add congruent conditions - may be split into multiple groups (split1, split2, split3)
        if congruent_conditions:
            # Check if there are splits
            split_groups = {}
            for cond in congruent_conditions:
                if 'split' in cond.lower():
                    # Extract split number
                    import re
                    split_match = re.search(r'split(\d+)', cond.lower())
                    if split_match:
                        split_num = split_match.group(1)
                        if split_num not in split_groups:
                            split_groups[split_num] = []
                        split_groups[split_num].append(cond)
                else:
                    # No split, treat as single group
                    if 'no_split' not in split_groups:
                        split_groups['no_split'] = []
                    split_groups['no_split'].append(cond)
            
            # Add each split group
            for split_id, cond_list in split_groups.items():
                conditions.append((f'congruent_{split_id}', cond_list))
        
        # Add incongruent conditions (typically no splits)
        if incongruent_conditions:
            conditions.append(('incongruent', incongruent_conditions))
        
        return conditions
    
class InfoCouplingDecoder(Experiment1Decoder):
    """
    Decoder for information coupling analysis of experiment 1.
    """
    def __init__(self, data_dir: Union[str, Path], roi: str,
                 voxel_counts: List[int] = list(range(500, 1100, 100))):
        super().__init__(data_dir, roi, voxel_counts=voxel_counts)
        
    def run_complete_analysis(self, subject_id: str) -> pd.DataFrame:
        """
        Run time-resolved multivariate decoding
        """
        all_results = []
        
        for n_voxels in self.voxel_counts:
            
            try:
                train_dataset, test_dataset = self.experiment_loader.load_experiment_1_data(
                    subject_id, self.roi, localizer=self.localizer, n_voxels=n_voxels, fir=True
                )
                
                train_clean, test_clean = remove_nan_voxels(train_dataset, test_dataset)
                result_df = self._run_time_resolved_decoding(train_clean, test_clean)
                result_df['roi'] = self.roi
                result_df['subject_id'] = subject_id
                
                all_results.append(result_df)
            except Exception as e:
                logger.warning(f"Analysis failed for {n_voxels} voxels: {e}")
            
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def _run_time_resolved_decoding(self, train_dataset: MVPADataset, 
                                    test_dataset: MVPADataset) -> pd.DataFrame:
        all_delay_results = []
        for delay in np.unique(test_dataset.delays):
            this_delay_data = test_dataset.filter_by_delay(delay)
            this_delay_res = self._analyze_by_congruency(base_dataset=train_dataset,
                                                         congruency_dataset=this_delay_data)
            this_delay_res['delay'] = delay
            all_delay_results.append(this_delay_res)
        
        return pd.concat(all_delay_results, ignore_index=True)
        
        
        
class Experiment2Decoder:
    """
    Decoder for Experiment 2: Cross-decoding from visible to occluded objects
    Now supports bilateral (L+R hemisphere) and bidirectional (train↔test) analysis
    """
    def __init__(self, data_dir: Union[str, Path], roi: str,
                 voxel_counts: List[int] = list(range(100, 6100, 100))):
        self.data_dir = Path(data_dir)
        self.roi = roi
        self.voxel_counts = voxel_counts
        
        # Create data loader
        self.experiment_loader = ExperimentDataLoader(self.data_dir)
        self.localizer = LocalizerLoader(self.data_dir, task='funcloc',
                                         model_name='exp2_objscr_baseline',
                                         contrast_name='objscr-vs-baseline')
        
    def run_complete_analysis(self, subject_id: str) -> pd.DataFrame:
        """
        Run the complete Experiment 2 analysis with bilateral and bidirectional support
        """
        all_results = []
        
        # Parse ROI name to check if bilateral
        base_roi_name, hemisphere = self.experiment_loader.parse_roi_name(self.roi)
        
        for n_voxels in self.voxel_counts:
            
            if hemisphere is None and self.experiment_loader.has_bilateral_roi(base_roi_name):
                # Bilateral analysis
                (train_L, test_L), (train_R, test_R) = self.experiment_loader.load_bilateral_data(
                    subject_id, base_roi_name, experiment=2,
                    localizer=self.localizer, n_voxels=n_voxels
                )
                
                # Process left hemisphere
                try:
                    train_clean_L, test_clean_L = remove_nan_voxels(train_L, test_L)
                    result_df_L = self._run_bidirectional_analysis(train_clean_L, test_clean_L,
                                                                   self._run_cross_decoding)
                    result_df_L['hemisphere'] = 'L'
                    result_df_L['n_voxels'] = n_voxels
                    result_df_L['roi'] = base_roi_name
                    result_df_L['subject_id'] = subject_id
                    all_results.append(result_df_L)
                except Exception as e:
                    logger.warning(f"Left hemisphere analysis failed for {n_voxels} voxels: {e}")
                    
                # Process right hemisphere
                try:
                    train_clean_R, test_clean_R = remove_nan_voxels(train_R, test_R)
                    result_df_R = self._run_bidirectional_analysis(train_clean_R, test_clean_R,
                                                                   self._run_cross_decoding)
                    result_df_R['hemisphere'] = 'R'
                    result_df_R['n_voxels'] = n_voxels
                    result_df_R['roi'] = base_roi_name
                    result_df_R['subject_id'] = subject_id
                    all_results.append(result_df_R)
                except Exception as e:
                    logger.warning(f"Right hemisphere analysis failed for {n_voxels} voxels: {e}")
            
            else:
                # Unilateral analysis
                train_dataset, test_dataset = self.experiment_loader.load_experiment_2_data(
                    subject_id, self.roi, localizer=self.localizer, n_voxels=n_voxels
                )
                
                try:
                    train_clean, test_clean = remove_nan_voxels(train_dataset, test_dataset)
                    result_df = self._run_bidirectional_analysis(train_clean, test_clean)
                    result_df['n_voxels'] = n_voxels
                    result_df['roi'] = self.roi
                    result_df['subject_id'] = subject_id
                    
                    # Add hemisphere info if specific hemisphere was requested
                    if hemisphere:
                        result_df['hemisphere'] = hemisphere
                        
                    all_results.append(result_df)
                
                except Exception as e:
                    logger.warning(f"Analysis failed for {n_voxels} voxels: {e}")
                
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def _run_bidirectional_analysis(self, dataset1: MVPADataset, dataset2: MVPADataset, 
                                    analysis_function) -> pd.DataFrame:
        """
        Run any analysis function in both directions and concatenate the results.
        
        Parameters:
        -----------
        dataset1, dataset2 : MVPADataset
            The two datasets
        analysis_function : callable
            Function that takes (train_dataset, test_dataset) and returns Dict
            
        Returns:
        --------
        pd.DataFrame
        """
        # Direction 1: dataset1 -> dataset2
        result1_df = analysis_function(dataset1, dataset2)
        result1_df['direction'] = 'forward'
        
        # Direction 2: dataset2 -> dataset1
        result2_df = analysis_function(dataset2, dataset1)
        result2_df['direction'] = 'backward'
        
        # Average the two results
        return pd.concat([result1_df, result2_df], ignore_index=True)
    
    def _run_cross_decoding(self, train_dataset: MVPADataset, 
                            test_dataset: MVPADataset) -> pd.DataFrame:
        """
        Run cross-decoding from visible objects to occluded objects
        """
        
        train_X, train_y = self._prepare_data(train_dataset)
        test_X, test_y = self._prepare_data(test_dataset)
        
        if train_X is None or test_X is None:
            return pd.DataFrame()
        
        cross_decoder = TrainTestClassifier()
        result_df = cross_decoder.train_test(train_X, train_y,
                                          test_X, test_y)
        
        return result_df
    
    def _prepare_data(self, dataset: MVPADataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data (with wide/narrow labels)
        """
        try:
            shape_labels = self.experiment_loader.create_shape_labels(dataset)
            return dataset.data, shape_labels
        except Exception as e:
            logger.warning(f"Error preparing data: {e}")
            return None, None