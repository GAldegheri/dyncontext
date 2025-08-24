import numpy as np
import logging
from typing import Union, Tuple
from analysis.mvpa.loaders import MVPADataset

logger = logging.getLogger(__name__)

def remove_nan_voxels(*datasets: MVPADataset) -> Union[MVPADataset, Tuple[MVPADataset, ...]]:
        """
        Remove voxels with NaN values consistently across datasets.
        
        Parameters:
        -----------
        *datasets: MVPADataset
            Variable number of datasets to process
        
        Returns:
        -----------
        MVPADataset or Tuple[MVPADataset, ...]: Cleaned dataset(s)
        """
        
        if len(datasets) == 1:
            # Single dataset case (e.g. cross-validation)
            dataset = datasets[0]
            valid_voxels = np.all(np.isfinite(dataset.data), axis=0)
            n_removed = np.sum(~valid_voxels)
            if n_removed > 0:
                logger.info(f"Removing {n_removed} NaN voxels from single dataset")
            
            return dataset.select_voxels(valid_voxels)
        
        else:
            # Multiple datasets - find voxels valid in ALL datasets
            valid_masks = [np.all(np.isfinite(ds.data), axis=0) for ds in datasets]
            valid_voxels = np.logical_and.reduce(valid_masks)
            
            n_removed = np.sum(~valid_voxels)
            if n_removed > 0:
                logger.info(f"Removing {n_removed} NaN voxels from {len(datasets)} datasets")
                
            # Apply mask to all datasets
            cleaned_datasets = tuple(ds.select_voxels(valid_voxels) for ds in datasets)
            
            return cleaned_datasets