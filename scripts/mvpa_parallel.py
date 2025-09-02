#!/usr/bin/env python
"""
Parallel submission of MVPA analysis for Experiment 1
Submits one job per subject using SLURM
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
from parallel.slurm_funcs import JobManager

def parse_subjects(subjects_str: str) -> List[str]:
    """Parse subjects string into list of subject IDs."""
    if subjects_str == 'all':
        return 'all'
    elif ',' in subjects_str:
        return [s.strip() for s in subjects_str.split(',')]
    elif '-' in subjects_str and subjects_str.startswith('sub-'):
        # Handle range like "sub-001-sub-005"
        parts = subjects_str.split('-')
        if len(parts) >= 3:  # sub, 001, sub, 005
            start_num = int(parts[1])
            end_num = int(parts[3])
            return [f"sub-{i:03d}" for i in range(start_num, end_num + 1)]
    else:
        return [subjects_str]

def auto_detect_subjects(data_dir: str) -> List[str]:
    """Auto-detect subjects in data directory."""
    data_path = Path(data_dir)
    subjects = []
    
    for path in data_path.iterdir():
        if path.is_dir() and path.name.startswith('sub-'):
            subjects.append(path.name)
    
    return sorted(subjects)

def create_job_parameters(subjects: List[str], rois: List[str], 
                         data_dir: str, output_dir: str,
                         voxel_start: int = 100, voxel_end: int = 6100, 
                         voxel_step: int = 100) -> List[Dict[str, Any]]:
    """Create parameter combinations for all subject/ROI pairs."""
    param_list = []
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for subject in subjects:
        for roi in rois:
            output_file = Path(output_dir) / f"exp1_results_{subject}_{roi}.csv"
            
            params = {
                'data_dir': data_dir,
                'subject': subject,
                'roi': roi,
                'output': str(output_file),
                'voxel_start': voxel_start,
                'voxel_end': voxel_end,
                'voxel_step': voxel_step
            }
            param_list.append(params)
    
    return param_list

def main():
    parser = argparse.ArgumentParser(
        description='Run Experiment 1 MVPA analysis in parallel')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--subjects', type=str, required=True,
                       help='Subjects: single (sub-001), comma-separated (sub-001,sub-002), '
                            'range (sub-001-sub-005), or "all"')
    parser.add_argument('--rois', type=str, required=True,
                       help='ROIs (comma-separated): ba-17-18,ba-19-37')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for result files')
    
    # SLURM parameters
    parser.add_argument('--walltime', type=str, default='04:00:00',
                       help='Wall time (HH:MM:SS)')
    parser.add_argument('--memory', type=str, default='16G',
                       help='Memory per job')
    parser.add_argument('--cpus', type=int, default=1,
                       help='CPUs per job')
    parser.add_argument('--partition', type=str, default='batch',
                       help='SLURM partition')
    parser.add_argument('--max_concurrent', type=int, default=100,
                       help='Max concurrent jobs')
    
    # Optional parameters
    parser.add_argument('--voxel_start', type=int, default=100)
    parser.add_argument('--voxel_end', type=int, default=6100)
    parser.add_argument('--voxel_step', type=int, default=100)
    parser.add_argument('--account', type=str, default=None,
                       help='SLURM account (if required)')
    
    args = parser.parse_args()
    
    # Parse subjects
    if args.subjects == 'all':
        subjects = auto_detect_subjects(args.data_dir)
        if not subjects:
            print("No subjects found in data directory")
            sys.exit(1)
        print(f"Auto-detected subjects: {subjects}")
    else:
        subjects = parse_subjects(args.subjects)
    
    # Parse ROIs
    rois = [roi.strip() for roi in args.rois.split(',')]
    
    # Validate inputs
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    # Show summary
    print(f"Subjects: {len(subjects)}")
    print(f"ROIs: {len(rois)}")
    print(f"Total jobs: {len(subjects) * len(rois)}")
    print(f"Output directory: {args.output_dir}")
    
    # Create parameters
    param_list = create_job_parameters(
        subjects=subjects,
        rois=rois,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        voxel_start=args.voxel_start,
        voxel_end=args.voxel_end,
        voxel_step=args.voxel_step
    )
    
    # Submit jobs
    job_manager = JobManager(max_concurrent=args.max_concurrent)
    
    submitted = job_manager.submit_batch(
        script_path="scripts/run_mvpa_exp1.py",
        param_list=param_list,
        job_prefix="exp1_mvpa",
        pattern="exp1_mvpa_batch",
        walltime=args.walltime,
        memory=args.memory,
        cpus=args.cpus,
        partition=args.partition,
        account=args.account
    )
    
    print(f"\nSubmitted {submitted} jobs successfully")
    print(f"Monitor with: squeue -u $USER | grep exp1_mvpa")

if __name__ == '__main__':
    main()