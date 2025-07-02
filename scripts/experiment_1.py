import logging
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.mvpa.decoders import Experiment1Decoder
from analysis.mvpa.loaders import (
    MVPADataset, BetaLoader, LocalizerLoader, VoxelSelector,
    load_experiment_data, split_by_view
)
from analysis.mvpa.classifiers import background_matched_decode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exp1_analysis(subject_id: str, roi: str, data_dir: str, n_voxels: int = 1000):
    
    logger.info(f"Subject: {subject_id}, ROI: {roi}, N. voxels: {n_voxels}")
    
    # Load datasets
    train_dataset, test_dataset = load_experiment_data(
        subject_id=subject_id,
        data_dir=data_dir,
        experiment=1,
        roi=roi
    )
    
    logger.info(f"Training dataset:")
    logger.info(f"  Shape: {train_dataset.data.shape}")
    logger.info(f"  Labels: {train_dataset.unique_labels}")
    logger.info(f"  N samples per condition:")
    for label in train_dataset.unique_labels:
        count = np.sum(train_dataset.labels == label)
        logger.info(f"    {label}: {count}")
        
    logger.info(f"\nTest dataset:")
    logger.info(f"  Shape: {test_dataset.data.shape}")
    logger.info(f"  Labels: {test_dataset.unique_labels}")
    logger.info(f"  N samples per condition:")
    for label in test_dataset.unique_labels:
        count = np.sum(test_dataset.labels == label)
        logger.info(f"    {label}: {count}")