import logging
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.mvpa.decoders import Experiment1Decoder
from parallel.torque_funcs import JobManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_exp1_analysis(subject_id: str, roi: str, data_dir: str, n_voxels: int = 1000):
    
    logger.info(f"Subject: {subject_id}, ROI: {roi}")
    
    # Create decoder
    decoder = Experiment1Decoder(roi=roi, metric='classifier_information')
    
    decoder.load_data(subject_id=subject_id, data_dir=data_dir)
    
    logger.info(f"Training dataset:")
    logger.info(f"  Shape: {decoder.train_dataset.data.shape}")
    logger.info(f"  Labels: {decoder.train_dataset.unique_labels}")
    logger.info(f"  N samples per condition:")
    for label in decoder.train_dataset.unique_labels:
        count = np.sum(decoder.train_dataset.labels == label)
        logger.info(f"    {label}: {count}")
        
    logger.info(f"\nTest dataset:")
    logger.info(f"  Shape: {decoder.test_dataset.data.shape}")
    logger.info(f"  Labels: {decoder.test_dataset.unique_labels}")
    logger.info(f"  N samples per condition:")
    for label in decoder.test_dataset.unique_labels:
        count = np.sum(decoder.test_dataset.labels == label)
        logger.info(f"    {label}: {count}")
        

def main():

    run_exp1_analysis('sub-001', 'ba-17-18', data_dir='project/3018040.05/bids')
    
if __name__=="__main__":
    main()