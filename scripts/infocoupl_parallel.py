"""
Parallel submission of information/activation coupling analysis for Experiment 1
Submits one job per subject (congruent, incongruent) using SLURM
"""
import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Any
from parallel.slurm_funcs import JobManager

def parse_subjects(subjects_str: str) -> List[str]:
    """Parse subjects string into list of subject IDs."""
    if ',' in subjects_str:
        return [s.strip() for s in subjects_str.split(',')]
    elif ':' in subjects_str and subjects_str.startswith('sub-'):
        # Handle range like "sub-001:sub-005"
        start_str, end_str = subjects_str.split(':')
        match_start = re.match(r"sub-(\d+)", start_str)
        match_end = re.match(r"sub-(\d+)", end_str)
        if match_start and match_end:
            start_num = int(match_start.group(1))
            end_num = int(match_end.group(1))
            return [f"sub-{i:03d}" for i in range(start_num, end_num + 1)]
    else:
        return [subjects_str]
    
def create_job_parameters(subjects: List[str], source_roi: str, 
                         data_dir: str, output_dir: str,
                         voxel_start: int = 500, voxel_end: int = 1100, 
                         voxel_step: int = 100) -> List[Dict[str, Any]]:
    """Create parameter combinations for all subject/ROI pairs."""
    param_list = []
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for subject in subjects:
        
        params = {
            'subject_id': subject,
            'data_dir': data_dir,
            'source_roi': source_roi,
            'output_dir': output_dir,
            'voxel_start': voxel_start,
            'voxel_end': voxel_end,
            'voxel_step': voxel_step
        }
        param_list.append(params)
    
    return param_list


def main():
    parser = argparse.ArgumentParser(
        description='Run information-activation coupling in parallel'
    )
    parser.add_argument('--subjects', type=str, required=True, help='Subject identifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory path')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    
    # SLURM parameters
    parser.add_argument('--walltime', type=str, default='02:00:00',
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
    parser.add_argument('--source_roi', type=str, default='ba-17-18', help='Source ROI for multivariate decoding')
    parser.add_argument('--voxel_start', type=int, default=500)
    parser.add_argument('--voxel_end', type=int, default=1100)
    parser.add_argument('--voxel_step', type=int, default=100)
    parser.add_argument('--account', type=str, default=None,
                       help='SLURM account (if required)')
    
    args = parser.parse_args()
    
    subjects = parse_subjects(args.subjects)
    
    # Validate inputs
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
        
    # Show summary
    print(f"Subjects: {len(subjects)}")
    print(f"Output directory: {args.output_dir}")
    
    # Create parameters
    param_list = create_job_parameters(
        subjects=subjects,
        data_dir=args.data_dir,
        source_roi=args.source_roi,
        output_dir=args.output_dir,
        voxel_start=args.voxel_start,
        voxel_end=args.voxel_end,
        voxel_step=args.voxel_step
    )
    
    # Submit jobs
    job_manager = JobManager(max_concurrent=args.max_concurrent)
    
    submitted = job_manager.submit_batch(
        script_path='scripts/run_infocoupling.py',
        param_list=param_list,
        job_prefix='exp1_infocoupl',
        pattern='exp1_infocoupl_batch',
        walltime=args.walltime,
        memory=args.memory,
        cpus=args.cpus,
        partition=args.partition,
        account=args.account
    )
    
    print(f"\nSubmitted {submitted} jobs successfully")
    
if __name__ == '__main__':
    main()