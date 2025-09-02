#!/usr/bin/env python
"""
Run MVPA analysis for Experiment 1
"""
import argparse
from pathlib import Path
import pandas as pd
from analysis.mvpa.decoders import Experiment1Decoder
from parallel.slurm_funcs import JobManager

def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1 MVPA analysis')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID (e.g., sub-01)')
    parser.add_argument('--roi', type=str, required=True,
                       help='ROI name (e.g., ba-17-18, ba-19-37)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path')
    parser.add_argument('--voxel_start', type=int, default=100,
                       help='Starting number of voxels')
    parser.add_argument('--voxel_end', type=int, default=6100,
                       help='Ending number of voxels')
    parser.add_argument('--voxel_step', type=int, default=100,
                       help='Step size for voxel counts')
    
    args = parser.parse_args()
    
    # Create voxel counts list
    voxel_counts = list(range(args.voxel_start, args.voxel_end, args.voxel_step))
    
    # Initialize decoder
    decoder = Experiment1Decoder(
        data_dir=args.data_dir,
        roi=args.roi,
        voxel_counts=voxel_counts
    )
    
    # Run analysis
    results = decoder.run_complete_analysis(args.subject)
    
    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        # Default output name
        output_path = f"exp1_results_{args.subject}_{args.roi}.csv"
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    # Print summary
    print(f"\nAnalysis complete for {args.subject}, ROI: {args.roi}")
    print(f"Total samples analyzed: {len(results)}")
    if 'correct' in results.columns:
        print(f"Overall accuracy: {results['correct'].mean():.3f}")

if __name__ == '__main__':
    main()