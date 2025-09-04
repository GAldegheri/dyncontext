"""
Information Coupling Analysis for Experiment 1
Computes correlations between EVC multivariate decoding timeseries 
and whole-brain univariate activation timeseries.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import new_img_like
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from analysis.mvpa.loaders import MVPADataset, BetaLoader, LocalizerLoader, ExperimentDataLoader
from analysis.mvpa.decoders import InfoCouplingDecoder
from analysis.mvpa.utils import remove_nan_voxels

logger = logging.getLogger(__name__)


class InfoCouplingAnalyzer:
    
    def __init__(self, data_dir: Path, source_roi: str = 'ba-17-18',
                 voxel_counts: List[int] = list(range(500, 1100, 100))):
        self.data_dir = Path(data_dir)
        self.source_roi = source_roi
        self.voxel_counts = voxel_counts

        self.decoder = InfoCouplingDecoder(data_dir=data_dir, roi=source_roi,
                                           voxel_counts=voxel_counts)
        self.loader = ExperimentDataLoader(data_dir=data_dir)
        
    def run_subject_analysis(self, subject_id: str) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
        """
        Run complete information coupling analysis for one subject.
        """
        logger.info(f"Running information coupling analysis for {subject_id}")
        
        # Get decoding timeseries for all voxel counts
        multivar_df = self.decoder.run_complete_analysis(subject_id=subject_id)
        
        # Load univariate whole-brain data
        wholebrain_dataset = self.loader.load_experiment_1_infocoupling(subject_id=subject_id,
                                                                    roi='wholebrain',
                                                                    fir=True)
        # Collect correlation maps across different voxel counts
        congruent_maps = []
        incongruent_maps = []
        
        for n_voxels in self.voxel_counts:
            
            nvox_multivar = multivar_df[multivar_df['n_voxels'] == n_voxels]
            
            if nvox_multivar.empty:
                logger.warning(f"No multivariate data found for {n_voxels} voxels")
                continue
            
            # Compute correlation maps for this voxel count
            try:
                congruent_corr_map, incongruent_corr_map = self._compute_correlation_maps(
                    nvox_multivar, wholebrain_dataset
                )
                congruent_maps.append(congruent_corr_map)
                incongruent_maps.append(incongruent_corr_map)
                
            except Exception as e:
                logger.warning(f"Failed to compute correlations for {n_voxels} voxels: {e}")
                continue
            
            # Average across voxel counts
        logger.info("Averaging across voxel counts...")
        avg_congruent_data = np.nanmean(np.stack([img.get_fdata() for img in congruent_maps]), axis=0)
        avg_incongruent_data = np.nanmean(np.stack([img.get_fdata() for img in incongruent_maps]), axis=0)
        
        # Create final Nifti images
        congruent_img = nib.Nifti1Image(avg_congruent_data, affine=congruent_maps[0].affine)
        incongruent_img = nib.Nifti1Image(avg_incongruent_data, affine=incongruent_maps[0].affine)
        
        logger.info("Information coupling analysis completed")
        return congruent_img, incongruent_img 
            
    def _compute_correlation_maps(self, multivar_df: pd.DataFrame,
                                  wholebrain_dataset: MVPADataset) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
        """
        Compute correlation maps between multivariate and univariate timeseries.
        
        Parameters:
        -----------
        multivar_df : pd.DataFrame
            Multivariate decoding results with delay and congruency information
        wholebrain_dataset : MVPADataset  
            Whole-brain univariate data with delay information
            
        Returns:
        --------
        congruent_map, incongruent_map : Tuple[nib.Nifti1Image, nib.Nifti1Image]
        """
        
        # Extract multivariate timeseries grouped by condition and delay
        multivar_grouped = multivar_df.groupby(['congruency', 'delay'])['classifier_info'].mean().reset_index()
        
        # Get unique delays from multivariate data
        delays = sorted(multivar_grouped['delay'].unique())
        n_delays = len(delays)
        
        if n_delays < 2:
            raise ValueError("Need at least 2 time points for correlation analysis")
        
        # Extract multivariate timeseries for each condition
        congruent_multivar = multivar_grouped[multivar_grouped['congruency'] == 'congruent']
        incongruent_multivar = multivar_grouped[multivar_grouped['congruency'] == 'incongruent']
        
        if len(congruent_multivar) != n_delays or len(incongruent_multivar) != n_delays:
            raise ValueError("Incomplete multivariate timeseries data")
        
        congruent_mv_ts = congruent_multivar['classifier_info'].values
        incongruent_mv_ts = incongruent_multivar['classifier_info'].values
        
        # Extract univariate timeseries for each voxel and condition
        congruent_uv_data = []
        incongruent_uv_data = []
        
        # Group univariate data by congruency condition
        for delay in delays:
            delay_data = wholebrain_dataset.filter_by_delay(delay)
            
            # Separate by congruency
            congruent_trials = []
            incongruent_trials = []
            
            for i, label in enumerate(delay_data.labels):
                if 'congruent' in label.lower() and 'incongruent' not in label.lower():
                    congruent_trials.append(delay_data.data[i])
                elif 'incongruent' in label.lower():
                    incongruent_trials.append(delay_data.data[i])
            
            if congruent_trials:
                congruent_uv_data.append(np.mean(congruent_trials, axis=0))
            if incongruent_trials:
                incongruent_uv_data.append(np.mean(incongruent_trials, axis=0))
                
        if len(congruent_uv_data) != n_delays or len(incongruent_uv_data) != n_delays:
            raise ValueError("Incomplete univariate timeseries data")
        
        # Convert to arrays: (n_voxels, n_delays)
        congruent_uv_array = np.array(congruent_uv_data).T # transpose to get voxels x time
        incongruent_uv_array = np.array(incongruent_uv_data).T
        
        # Normalize timeseries (z-score across time)
        congruent_mv_ts_norm = self._normalize_timeseries(congruent_mv_ts)
        incongruent_mv_ts_norm = self._normalize_timeseries(incongruent_mv_ts)
        
        congruent_uv_norm = self._normalize_timeseries(congruent_uv_array, axis=1)
        incongruent_uv_norm = self._normalize_timeseries(incongruent_uv_array, axis=1)
        
        # Compute correlations for each voxel
        congruent_correlations = self._compute_pearson_correlations(
            congruent_uv_norm, congruent_mv_ts_norm, n_delays
        )
        incongruent_correlations = self._compute_pearson_correlations(
            incongruent_uv_norm, incongruent_mv_ts_norm, n_delays
        )
        
        # Convert to brain maps
        congruent_map = self._create_brain_map(congruent_correlations, wholebrain_dataset)
        incongruent_map = self._create_brain_map(incongruent_correlations, wholebrain_dataset)
        
        return congruent_map, incongruent_map
    
    def _normalize_timeseries(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Normalize timeseries by z-scoring across specified axis"""
        return (data - np.mean(data, axis=axis, keepdims=True)) / np.std(data, axis=axis, keepdims=True)
    
    def _compute_pearson_correlations(self, univariate_data: np.ndarray, 
                                      multivariate_ts: np.ndarray, n_delays: int) -> np.ndarray:
        """
        Compute Pearson correlations between univariate voxel timeseries and multivariate timeseries
        
        Parameters:
        -----------
        univariate_data : np.ndarray
            Shape (n_voxels, n_delays) - normalized univariate data
        multivariate_ts : np.ndarray  
            Shape (n_delays,) - normalized multivariate timeseries
        n_delays : int
            Number of time points
            
        Returns:
        --------
        np.ndarray : Correlation values for each voxel
        """
        # Compute correlations using dot product (since data is normalized)
        correlations = np.dot(univariate_data, multivariate_ts) / (n_delays - 1)
        return correlations
    
    def _create_brain_map(self, correlation_values: np.ndarray, 
                          wholebrain_dataset: MVPADataset) -> nib.Nifti1Image:
        """
        Convert correlation values back to 3D brain map
        
        Parameters:
        -----------
        correlation_values : np.ndarray
            Correlation values for each voxel
        wholebrain_dataset : MVPADataset
            Dataset containing coordinate information
            
        Returns:
        --------
        nib.Nifti1Image : 3D brain map
        """
        
        wholebrain_mask_path = self.data_dir / 'roi_masks' / 'wholebrain.nii'
        ref_img = nib.load(wholebrain_mask_path)
        ref_hdr = ref_img.header.copy()
        ref_shape = ref_img.shape
        
        # Initialize brain map with NaNs
        brain_map = np.full(ref_shape, np.nan, dtype=np.float32)
        
        # Map correlations back to brain coordinates
        if wholebrain_dataset.voxel_coords is not None:
            coords = wholebrain_dataset.voxel_coords
            i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
            
            # Ensure coordinates are within bounds
            valid_mask = ((i >= 0) & (i < ref_shape[0]) & 
                         (j >= 0) & (j < ref_shape[1]) & 
                         (k >= 0) & (k < ref_shape[2]))
            
            if np.any(~valid_mask):
                logger.warning(f"Some coordinates are out of bounds, excluding {np.sum(~valid_mask)} voxels")
            
            brain_map[i[valid_mask], j[valid_mask], k[valid_mask]] = correlation_values[valid_mask]
            
        else:
            logger.warning("No voxel coordinates available, cannot create brain map")
            
        # Create Nifti image
        return nib.Nifti1Image(brain_map, affine=ref_img.affine, header=ref_hdr)
    
    def save_results(self, congruent_map: nib.Nifti1Image, incongruent_map: nib.Nifti1Image,
                     subject_id: str, output_dir: Optional[Path] = None):
        """Save correlation maps to disk"""
        
        if output_dir is None:
            output_dir = (self.data_dir / 'experiment_1' / 'derivatives' / 'spm-preproc' / 
                          'derivatives' / 'spm-stats' / 'betas' / 'derivatives' / 'mvpa_results' / 
                          'info_coupling_results')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save correlation maps
        congruent_path = output_dir / f'{subject_id}_congruent_correlation_map.nii.gz'
        incongruent_path = output_dir / f'{subject_id}_incongruent_correlation_map.nii.gz'
        
        nib.save(congruent_map, congruent_path)
        nib.save(incongruent_map, incongruent_path)
        
        logger.info(f"Saved correlation maps to {output_dir}")
        
        return congruent_path, incongruent_path
    

def main(subject_id: str, data_dir: str, source_roi: str = 'ba-17-18', 
         voxel_counts: List[int] = None, output_dir: str = None):
    """
    Main function to run information coupling analysis
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    data_dir : str  
        Path to data directory
    source_roi : str
        Source ROI for multivariate decoding (default: ba-17-18)
    voxel_counts : List[int]
        List of voxel counts to use (default: 500-1000 in steps of 100)
    output_dir : str
        Output directory for results
    """
    
    if voxel_counts is None:
        voxel_counts = list(range(500, 1100, 100))
        
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create analyzer
        analyzer = InfoCouplingAnalyzer(
            data_dir=Path(data_dir),
            source_roi=source_roi,
            voxel_counts=voxel_counts
        )
        
        # Run analysis
        congruent_map, incongruent_map = analyzer.run_subject_analysis(subject_id)
        
        # Save results
        output_paths = analyzer.save_results(
            congruent_map, incongruent_map, subject_id, output_dir
        )
        
        logger.info(f"Analysis completed successfully for {subject_id}")
        logger.info(f"Results saved to: {Path(output_paths[0]).parent}")
        
        return congruent_map, incongruent_map
        
    except Exception as e:
        logger.error(f"Analysis failed for {subject_id}: {str(e)}")
        raise

    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run information coupling analysis')
    parser.add_argument('--subject_id', required=True, help='Subject identifier')
    parser.add_argument('--data_dir', required=True, help='Data directory path')
    parser.add_argument('--source_roi', default='ba-17-18', help='Source ROI for multivariate decoding')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--voxel_start', type=int, default=500)
    parser.add_argument('--voxel_end', type=int, default=1100)
    parser.add_argument('--voxel_step', type=int, default=100)
    
    args = parser.parse_args()
    
    voxel_counts = list(range(args.voxel_start, args.voxel_end, args.voxel_step))
    
    main(
        subject_id=args.subject_id,
        data_dir=args.data_dir, 
        source_roi=args.source_roi,
        voxel_counts=voxel_counts,
        output_dir=args.output_dir
    )
        
        
        
            
            
        
        
    