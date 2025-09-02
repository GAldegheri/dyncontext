#!/usr/bin/env python
"""
Test Script for SLURM Job Manager
==================================

This script tests the SLURM job submission and monitoring system with simple operations
before running the full analysis.

Usage:
    python test_slurm_manager.py submit    # Submit test jobs
    python test_slurm_manager.py monitor   # Monitor test jobs
    python test_slurm_manager.py status    # Check job status
    python test_slurm_manager.py collect   # Collect test results
    python test_slurm_manager.py clean     # Clean up test files
    python test_slurm_manager.py cancel    # Cancel all test jobs
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path
import getpass

# Add analysis path - update this to your project path
sys.path.append('/project/3018040.05/dyncontext/')

from parallel.slurm_funcs import (
    JobManager, submit_job, monitor_jobs_live, 
    get_job_status_summary, cleanup_tracking,
    cancel_jobs, get_job_info, check_queue_status,
    estimate_wait_time
)

# =============================================================================
# Test Configuration
# =============================================================================

# Simple test parameters
TEST_SUBJECTS = ['sub-001', 'sub-002', 'sub-003']
TEST_CONDITIONS = ['condition_A', 'condition_B']
TEST_NUMBERS = [10, 20, 30]

# SLURM settings for testing (conservative)
MAX_TEST_JOBS = 5  # Keep it small for testing
JOB_CHECK_INTERVAL = 10  # Check every 10 seconds
TEST_OUTPUT_DIR = 'test_results'

# SLURM-specific settings
DEFAULT_PARTITION = 'batch'  # Update based on your cluster
DEFAULT_ACCOUNT = None  # Set if your cluster requires account specification

# =============================================================================
# Create Test Worker Script
# =============================================================================

def create_test_worker():
    """Create a simple worker script for testing"""
    
    worker_script = 'test_worker.py'
    
    worker_content = '''#!/usr/bin/env python
"""
Simple test worker that performs basic operations
"""
import sys
import argparse
import time
import random
import math
from pathlib import Path

def run_test_job(subject, condition, number, sleep_time, output_dir):
    """
    Run a simple test job that:
    1. Does some basic calculations
    2. Sleeps for a random time (simulates real work)
    3. Writes results to file
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Simulate some work
    print(f"Processing {subject} - {condition} - {number}")
    print(f"Sleeping for {sleep_time} seconds...")
    
    start_time = time.time()
    
    # Do some calculations
    results = []
    for i in range(number):
        value = math.sqrt(i * random.random())
        results.append(value)
        time.sleep(sleep_time / number)  # Distribute sleep time
    
    elapsed = time.time() - start_time
    
    # Write results
    output_file = output_path / f"test_result_{subject}_{condition}_{number}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Test Results\\n")
        f.write(f"============\\n")
        f.write(f"Subject: {subject}\\n")
        f.write(f"Condition: {condition}\\n")
        f.write(f"Number: {number}\\n")
        f.write(f"Processing time: {elapsed:.2f} seconds\\n")
        f.write(f"Mean result: {sum(results)/len(results):.4f}\\n")
        f.write(f"Results: {results[:5]}...\\n")  # First 5 values
    
    print(f"Results written to {output_file}")
    print(f"Job completed in {elapsed:.2f} seconds")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Test worker script')
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--condition', type=str, required=True)
    parser.add_argument('--number', type=int, required=True)
    parser.add_argument('--sleep_time', type=float, default=10.0)
    parser.add_argument('--output_dir', type=str, default='test_results')
    
    args = parser.parse_args()
    
    return run_test_job(
        args.subject,
        args.condition,
        args.number,
        args.sleep_time,
        args.output_dir
    )

if __name__ == '__main__':
    sys.exit(main())
'''
    
    # Write worker script
    with open(worker_script, 'w') as f:
        f.write(worker_content)
    
    # Make executable
    os.chmod(worker_script, 0o755)
    
    print(f"Created worker script: {worker_script}")
    return worker_script

# =============================================================================
# Test Functions
# =============================================================================

def submit_test_jobs():
    """Submit test jobs using SLURM"""
    
    print("SLURM Test Job Submission")
    print("=" * 30)
    
    # Check queue status first
    print("\nChecking queue status...")
    queue_info = check_queue_status(DEFAULT_PARTITION)
    if queue_info:
        for q in queue_info.get('queues', []):
            if q['partition'] == DEFAULT_PARTITION or DEFAULT_PARTITION is None:
                print(f"Partition: {q['partition']}, State: {q['state']}, Available: {q['available']}")
    
    # Estimate wait time
    wait = estimate_wait_time(DEFAULT_PARTITION, "4G", 1)
    if wait:
        print(f"Estimated wait time: {wait}")
    
    # Generate all parameter combinations
    all_params = []
    for subject in TEST_SUBJECTS:
        for condition in TEST_CONDITIONS:
            for number in TEST_NUMBERS:
                params = {
                    'subject': subject,
                    'condition': condition,
                    'number': number,
                    'sleep_time': random.uniform(5, 15),  # Random sleep 5-15 seconds
                    'output_dir': TEST_OUTPUT_DIR
                }
                all_params.append(params)
    
    total_jobs = len(all_params)
    
    print(f"\nPreparing to submit {total_jobs} test jobs")
    print(f"Max concurrent jobs: {MAX_TEST_JOBS}")
    print(f"Partition: {DEFAULT_PARTITION}")
    
    # Confirm submission
    response = input("\nProceed with submission? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Setup
    output_dir = Path(TEST_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    worker_script = create_test_worker()
    
    # Use JobManager for batch submission
    job_manager = JobManager(
        max_concurrent=MAX_TEST_JOBS, 
        check_interval=JOB_CHECK_INTERVAL,
        user=getpass.getuser()
    )
    
    print(f"\nSubmitting {total_jobs} test jobs...")
    
    # Submit batch with automatic rate limiting
    job_kwargs = {
        'walltime': "00:10:00",  # 10 minutes for testing
        'memory': "4G",           # Less memory for testing
        'cpus': 1,                # Single CPU for simple tests
        'partition': DEFAULT_PARTITION
    }
    
    if DEFAULT_ACCOUNT:
        job_kwargs['account'] = DEFAULT_ACCOUNT
    
    submitted_count = job_manager.submit_batch(
        script_path=worker_script,
        param_list=all_params,
        job_prefix="slurm_test",
        pattern="slurm_test",
        stagger_delay=0.5,  # Half second between submissions
        **job_kwargs
    )
    
    print(f"\nSubmission complete!")
    print(f"Successfully submitted: {submitted_count}")
    print(f"Failed submissions: {total_jobs - submitted_count}")
    
    if submitted_count > 0:
        print("\nYou can monitor jobs with: python test_slurm_manager.py monitor")

def monitor_test_jobs():
    """Monitor test job progress"""
    
    print("SLURM Test Job Monitoring")
    print("=" * 30)
    
    # Get current user
    user = getpass.getuser()
    print(f"User: {user}")
    print(f"Pattern: slurm_test")
    
    # Use the built-in monitor function with pattern
    monitor_jobs_live(pattern="slurm_test", check_interval=JOB_CHECK_INTERVAL, user=user)

def check_job_status():
    """Check current status of test jobs"""
    
    print("SLURM Job Status Check")
    print("=" * 30)
    
    # Get summary for test jobs
    summary = get_job_status_summary(pattern="slurm_test")
    
    print("\nTest job summary:")
    print(f"  Total: {summary.get('total', 0)}")
    print(f"  Running: {summary.get('RUNNING', 0) + summary.get('running', 0)}")
    print(f"  Pending: {summary.get('PENDING', 0) + summary.get('pending', 0)}")
    print(f"  Completed: {summary.get('COMPLETED', 0) + summary.get('completed', 0)}")
    print(f"  Failed: {summary.get('FAILED', 0) + summary.get('failed', 0)}")
    
    # Show queue status
    print("\nQueue status:")
    queue_info = check_queue_status(DEFAULT_PARTITION)
    if queue_info:
        for q in queue_info.get('queues', []):
            print(f"  {q['partition']}: {q['state']} ({q['nodes']} nodes)")

def collect_test_results():
    """Collect test results and show summary"""
    
    print("Test Results Collection")
    print("=" * 30)
    
    output_path = Path(TEST_OUTPUT_DIR)
    
    if not output_path.exists():
        print(f"Output directory {output_path} does not exist!")
        return
    
    # Find result files
    result_files = list(output_path.glob("test_result_*.txt"))
    
    # Find SLURM output files
    slurm_output_dir = Path("slurm/jobs/output")
    slurm_outputs = []
    slurm_errors = []
    
    if slurm_output_dir.exists():
        slurm_outputs = list(slurm_output_dir.glob("slurm_test_*_output.txt"))
        slurm_errors = list(slurm_output_dir.glob("slurm_test_*_error.txt"))
    
    print(f"\nFound {len(result_files)} result files")
    print(f"Found {len(slurm_outputs)} SLURM output files")
    print(f"Found {len(slurm_errors)} SLURM error files")
    
    if result_files:
        print("\nSample results (first 3):")
        
        for i, file in enumerate(result_files[:3]):
            print(f"\n{i+1}. {file.name}")
            with open(file, 'r') as f:
                lines = f.readlines()[:6]  # First 6 lines
                for line in lines:
                    print(f"   {line.strip()}")
    
    # Check for errors
    if slurm_errors:
        error_count = 0
        for error_file in slurm_errors:
            if error_file.stat().st_size > 0:  # Non-empty error file
                error_count += 1
        
        if error_count > 0:
            print(f"\n⚠ Warning: {error_count} jobs had errors")
            print("Check slurm/jobs/output/ for details")

def clean_test_files():
    """Clean up test files and job scripts"""
    
    print("Cleaning Test Files")
    print("=" * 20)
    
    # Confirm cleanup
    response = input("This will delete all test files. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Clean result files
    output_path = Path(TEST_OUTPUT_DIR)
    if output_path.exists():
        result_files = list(output_path.glob("test_result_*.txt"))
        for f in result_files:
            f.unlink()
        print(f"Deleted {len(result_files)} result files")
    
    # Clean SLURM job scripts
    slurm_scripts = Path("slurm/jobs")
    if slurm_scripts.exists():
        scripts = list(slurm_scripts.glob("slurm_test_*.sh"))
        for s in scripts:
            s.unlink()
        print(f"Deleted {len(scripts)} job scripts")
    
    # Clean SLURM outputs
    slurm_output_dir = Path("slurm/jobs/output")
    if slurm_output_dir.exists():
        outputs = list(slurm_output_dir.glob("slurm_test_*"))
        for o in outputs:
            o.unlink()
        print(f"Deleted {len(outputs)} SLURM output files")
    
    # Clean worker script
    worker_script = Path("test_worker.py")
    if worker_script.exists():
        worker_script.unlink()
        print("Deleted test worker script")
    
    # Clear tracking
    cleanup_tracking()
    
    print("\nCleanup complete!")

def cancel_test_jobs():
    """Cancel all test jobs"""
    
    print("Cancelling Test Jobs")
    print("=" * 20)
    
    # Confirm cancellation
    response = input("Cancel all test jobs? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    cancelled = cancel_jobs(pattern="slurm_test")
    print(f"Cancelled {cancelled} jobs")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test SLURM job management system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'action',
        choices=['submit', 'monitor', 'status', 'collect', 'clean', 'cancel'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--partition',
        default=DEFAULT_PARTITION,
        help='SLURM partition to use'
    )
    
    parser.add_argument(
        '--account',
        default=DEFAULT_ACCOUNT,
        help='Account to charge (if required)'
    )
    
    args = parser.parse_args()
    
    # Execute action
    if args.action == 'submit':
        submit_test_jobs()
    elif args.action == 'monitor':
        monitor_test_jobs()
    elif args.action == 'status':
        check_job_status()
    elif args.action == 'collect':
        collect_test_results()
    elif args.action == 'clean':
        clean_test_files()
    elif args.action == 'cancel':
        cancel_test_jobs()

if __name__ == '__main__':
    main()