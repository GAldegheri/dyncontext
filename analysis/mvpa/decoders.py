import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

from analysis.mvpa.base import MVPAAnalysis
from analysis.mvpa.classifiers import (
    cross_decode, cross_validate, background_matched_decode,
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