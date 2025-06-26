"""
Simple Test Script for Job Manager
==================================

This script tests the job submission and monitoring system with simple operations
before running the full MVPA analysis.

Usage:
    python test_job_manager.py submit    # Submit test jobs
    python test_job_manager.py monitor   # Monitor test jobs
    python test_job_manager.py collect   # Collect test results
    python test_job_manager.py clean     # Clean up test files
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path

# Add analysis path
sys.path.append('/project/3018040.05/dyncontext/')

from parallel.torque_funcs import (
    JobManager, submit_job, monitor_jobs_live, 
    get_job_status_summary, cleanup_tracking
)

# =============================================================================
# Test Configuration
# =============================================================================

# Simple test parameters
TEST_SUBJECTS = ['sub-001', 'sub-002', 'sub-003']
TEST_CONDITIONS = ['condition_A', 'condition_B']
TEST_NUMBERS = [10, 20, 30]

# Cluster settings for testing (conservative)
MAX_TEST_JOBS = 5  # Keep it small for testing
JOB_CHECK_INTERVAL = 10  # Check every 10 seconds
TEST_OUTPUT_DIR = 'test_results'

# =============================================================================
# Create Test Worker Script
# =============================================================================

def create_test_worker():
    """Create a simple worker script for testing"""
    
    worker_content = '''
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
    3. Saves a simple result
    """
    print(f"Starting test job:")
    print(f"  Subject: {subject}")
    print(f"  Condition: {condition}")
    print(f"  Number: {number}")
    print(f"  Sleep time: {sleep_time:.1f}s")
    
    try:
        # Simulate some computational work
        print("Performing calculations...")
        
        # Basic math operations
        result_sum = sum(range(number * 100))
        result_sqrt = math.sqrt(number * 1000)
        result_power = number ** 3
        
        # Simulate variable processing time
        actual_sleep = sleep_time + random.uniform(0, 2)
        print(f"Processing for {actual_sleep:.1f} seconds...")
        time.sleep(actual_sleep)
        
        # Create results
        results = {
            'subject': subject,
            'condition': condition,
            'input_number': number,
            'sum_result': result_sum,
            'sqrt_result': result_sqrt,
            'power_result': result_power,
            'processing_time': actual_sleep
        }
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"test_result_{subject}_{condition}_{number}.txt"
        
        with open(output_file, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\\n")
        
        print(f"✓ Test completed successfully!")
        print(f"  Results saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ Error in test job: {e}")
        
        # Save error info
        error_file = Path(output_dir) / f"error_{subject}_{condition}_{number}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\\n")
            f.write(f"Subject: {subject}\\n")
            f.write(f"Condition: {condition}\\n")
            f.write(f"Number: {number}\\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Worker")
    parser.add_argument('--subject', required=True, help='Subject ID')
    parser.add_argument('--condition', required=True, help='Test condition')
    parser.add_argument('--number', type=int, required=True, help='Test number')
    parser.add_argument('--sleep_time', type=float, required=True, help='Sleep time')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    run_test_job(args.subject, args.condition, args.number, args.sleep_time, args.output_dir)
'''

    # Write worker script
    worker_path = Path("torque/test_worker.py")
    worker_path.parent.mkdir(exist_ok=True)
    
    with open(worker_path, 'w') as f:
        f.write(worker_content)
    
    os.chmod(worker_path, 0o755)
    return str(worker_path)

# =============================================================================
# Test Job Generation
# =============================================================================

def generate_test_parameters():
    """Generate test job parameters"""
    
    parameters = []
    
    for subject in TEST_SUBJECTS:
        for condition in TEST_CONDITIONS:
            for number in TEST_NUMBERS:
                
                # Random sleep time between 5-30 seconds to simulate real work
                sleep_time = random.uniform(5, 30)
                
                params = {
                    'subject': subject,
                    'condition': condition,
                    'number': number,
                    'sleep_time': sleep_time,
                    'output_dir': TEST_OUTPUT_DIR
                }
                
                parameters.append(params)
    
    return parameters

# =============================================================================
# Main Functions
# =============================================================================

def submit_test_jobs():
    """Submit test jobs using JobManager"""
    
    print("Job Manager Test - Submitting Test Jobs")
    print("=" * 45)
    
    # Generate parameters
    all_params = generate_test_parameters()
    total_jobs = len(all_params)
    
    print(f"Subjects: {TEST_SUBJECTS}")
    print(f"Conditions: {TEST_CONDITIONS}")
    print(f"Numbers: {TEST_NUMBERS}")
    print(f"Total test jobs: {total_jobs}")
    print(f"Max concurrent: {MAX_TEST_JOBS}")
    
    # Confirm
    response = input(f"\nSubmit {total_jobs} test jobs? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Setup
    output_dir = Path(TEST_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    worker_script = create_test_worker()
    
    # Use JobManager for batch submission
    job_manager = JobManager(max_concurrent=MAX_TEST_JOBS, check_interval=JOB_CHECK_INTERVAL)
    
    print(f"\nSubmitting {total_jobs} test jobs...")
    
    # Submit batch with automatic rate limiting
    submitted_count = job_manager.submit_batch(
        script_path=worker_script,
        param_list=all_params,
        job_prefix="test_job",
        pattern="test_job",
        walltime="00:10:00",  # 10 minutes for testing
        memory="4g"           # Less memory for testing
    )
    
    print(f"\nSubmission complete!")
    print(f"Successfully submitted: {submitted_count}")
    print(f"Failed submissions: {total_jobs - submitted_count}")

def monitor_test_jobs():
    """Monitor test job progress"""
    
    print("Test Job Monitoring")
    print("=" * 25)
    
    # Use the built-in monitor function with pattern
    monitor_jobs_live(pattern="test_job", check_interval=JOB_CHECK_INTERVAL)

def collect_test_results():
    """Collect test results and show summary"""
    
    print("Test Results Collection")
    print("=" * 30)
    
    output_path = Path(TEST_OUTPUT_DIR)
    
    if not output_path.exists():
        print(f"Output directory {output_path} does not exist!")
        return
    
    # Find result and error files
    result_files = list(output_path.glob("test_result_*.txt"))
    error_files = list(output_path.glob("error_*.txt"))
    
    print(f"Found {len(result_files)} result files")
    print(f"Found {len(error_files)} error files")
    
    if result_files:
        print("\nSample results:")
        
        # Show first 3 results
        for i, file in enumerate(result_files[:3]):
            print(f"\n{i+1}. {file.name}:")
            try:
                with open(file, 'r') as f:
                    content = f.read().strip()
                    for line in content.split('\n')[:4]:  # First 4 lines
                        print(f"   {line}")
            except Exception as e:
                print(f"   Error reading file: {e}")
        
        if len(result_files) > 3:
            print(f"   ... and {len(result_files) - 3} more result files")
    
    if error_files:
        print(f"\n⚠ Errors in {len(error_files)} jobs:")
        for error_file in error_files[:5]:  # Show max 5 errors
            print(f"  {error_file.name}")
        if len(error_files) > 5:
            print(f"  ... and {len(error_files) - 5} more errors")
    
    # Calculate success rate
    expected_jobs = len(TEST_SUBJECTS) * len(TEST_CONDITIONS) * len(TEST_NUMBERS)
    success_rate = len(result_files) / expected_jobs * 100 if expected_jobs > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Expected jobs: {expected_jobs}")
    print(f"  Successful: {len(result_files)} ({success_rate:.1f}%)")
    print(f"  Failed: {len(error_files)}")

def cleanup_test_files():
    """Clean up test files"""
    
    print("Cleaning up test files...")
    
    # Remove output directory
    output_path = Path(TEST_OUTPUT_DIR)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
        print(f"✓ Removed {output_path}")
    
    # Remove worker script
    worker_path = Path("torque/test_worker.py")
    if worker_path.exists():
        worker_path.unlink()
        print(f"✓ Removed {worker_path}")
    
    # Remove job scripts
    job_dir = Path("torque/jobs")
    if job_dir.exists():
        test_scripts = list(job_dir.glob("test_job_*.sh"))
        for script in test_scripts:
            script.unlink()
        print(f"✓ Removed {len(test_scripts)} job scripts")
    
    # Clear tracking data
    cleanup_tracking()
    
    print("Cleanup complete!")

def show_test_status():
    """Show current test job status"""
    
    print("Test Job Status")
    print("=" * 20)
    
    # Get status summary for test jobs
    status = get_job_status_summary("test_job")
    
    if not status or status.get('total', 0) == 0:
        print("No test jobs found")
        return
    
    # Display status
    print(f"Total test jobs: {status.get('total', 0)}")
    print(f"  Running: {status.get('R', 0)}")
    print(f"  Queued: {status.get('Q', 0)}")
    print(f"  Completed: {status.get('C', 0)}")
    
    # For tracked jobs, we can also show the summary
    tracked = get_job_status_summary()  # No pattern = all tracked jobs
    if tracked.get('total', 0) > 0:
        print(f"\nAll tracked jobs: {tracked.get('total', 0)}")

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simple Job Manager Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'command',
        choices=['submit', 'monitor', 'collect', 'status', 'clean'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'submit':
        submit_test_jobs()
    elif args.command == 'monitor':
        monitor_test_jobs()
    elif args.command == 'collect':
        collect_test_results()
    elif args.command == 'status':
        show_test_status()
    elif args.command == 'clean':
        cleanup_test_files()

if __name__ == "__main__":
    main()