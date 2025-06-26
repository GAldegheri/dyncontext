"""
Enhanced Torque Functions with Job Monitoring
==============================================

Self-contained torque job management with automatic job tracking,
pattern matching with full job names, and intelligent monitoring.

The functions in this module handle all the complexity internally,
providing simple interfaces for job submission and monitoring.
"""

import os
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# Global job tracking
_submitted_jobs: Set[str] = set()
_job_patterns: Dict[str, Set[str]] = {}  # pattern -> set of job_ids


def serialize_args(kwargs: Dict) -> str:
    """
    Serialize the kwargs into a command line argument string.
    Handles tuples and complex types properly.
    """
    args_str = []
    for key, value in kwargs.items():
        if isinstance(value, (tuple, list)):
            args_str.append(f"--{key} {repr(value)}")
        elif isinstance(value, str):
            args_str.append(f"--{key} '{value}'")
        else:
            args_str.append(f"--{key} {value}")
    
    return " ".join(args_str)


def create_job_script(script_path: str, args_str: str, script_name: str, 
                     walltime: str = "04:00:00", memory: str = "16g",
                     nodes: str = "1:ppn=1", queue: str = "batch") -> None:
    """Create a PBS job script file."""
    
    # Ensure output directory exists
    output_dir = Path("torque/jobs/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create job script directory if it doesn't exist
    script_dir = Path(script_name).parent
    script_dir.mkdir(parents=True, exist_ok=True)
    
    job_basename = os.path.basename(script_name).replace('.sh', '')
    
    with open(script_name, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write(f"#PBS -N {job_basename}\n")
        file.write(f"#PBS -l walltime={walltime},mem={memory}\n")
        file.write(f"#PBS -l nodes={nodes}\n")
        file.write(f"#PBS -q {queue}\n")
        file.write(f"#PBS -o torque/jobs/output/{job_basename}_output.txt\n")
        file.write(f"#PBS -e torque/jobs/output/{job_basename}_error.txt\n")
        file.write("# Change to analysis directory\n")
        file.write("cd /project/3018040.05/dyncontext/\n")
        file.write("# Activate conda environment\n")
        file.write("source activate dyncontext\n")
        file.write("# Log start time\n")
        file.write('echo "Job started at: $(date)"\n')
        file.write("# Run the Python script\n")
        file.write(f"python {script_path} {args_str}\n")
        file.write("# Log completion time\n")
        file.write('echo "Job completed at: $(date)"\n')
    
    # Make script executable
    os.chmod(script_name, 0o755)


def get_job_status(job_id: str) -> Optional[str]:
    """Get the status of a specific job by ID."""
    try:
        result = subprocess.run(['qstat', job_id], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 3:  # Header + job line
                parts = lines[2].split()
                if len(parts) >= 5:
                    return parts[4]  # Status column
        return None
        
    except (subprocess.TimeoutExpired, Exception):
        return None


def _parse_qstat_full_format(qstat_output: str) -> List[Tuple[str, str, str]]:
    """Parse qstat -f output to extract complete job information."""
    jobs = []
    current_job = {}
    
    for line in qstat_output.split('\n'):
        line = line.strip()
        
        if line.startswith('Job Id:'):
            # Save previous job if complete
            if current_job and all(key in current_job for key in ['job_id', 'job_name', 'status']):
                jobs.append((current_job['job_id'], current_job['job_name'], current_job['status']))
            
            # Start new job
            job_id = line.split(':')[1].strip().split('.')[0]
            current_job = {'job_id': job_id}
            
        elif '=' in line and current_job:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'Job_Name':
                current_job['job_name'] = value
            elif key == 'job_state':
                current_job['status'] = value
    
    # Don't forget the last job
    if current_job and all(key in current_job for key in ['job_id', 'job_name', 'status']):
        jobs.append((current_job['job_id'], current_job['job_name'], current_job['status']))
    
    return jobs


def get_all_jobs_with_full_names(user: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    Get all jobs with complete job names using qstat -f.
    This is the definitive method for accurate pattern matching.
    """
    try:
        cmd = ['qstat', '-f']
        if user:
            cmd.extend(['-u', user])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return _parse_qstat_full_format(result.stdout)
        else:
            return []
            
    except (subprocess.TimeoutExpired, Exception):
        return []


def submit_job(script_path: str, kwargs: Dict, job_name: str,
               walltime: str = "04:00:00", memory: str = "16g",
               pattern: Optional[str] = None) -> Optional[str]:
    """
    Submit a job and automatically track it.
    
    Parameters:
    -----------
    script_path : str
        Path to the Python script
    kwargs : dict
        Keyword arguments for the script
    job_name : str
        Base name for the job
    walltime : str
        Maximum wall time
    memory : str
        Memory requirement
    pattern : str, optional
        Pattern to associate this job with for group monitoring
        
    Returns:
    --------
    str or None
        Job ID if successful, None if failed
    """
    global _submitted_jobs, _job_patterns
    
    args_str = serialize_args(kwargs)
    
    # Create safe job name (torque has limitations)
    safe_job_name = job_name.replace('/', '_').replace(' ', '_')[:50]
    
    # Create unique script name
    param_hash = abs(hash(str(sorted(kwargs.items())))) % 10000
    script_name = f"torque/jobs/{safe_job_name}_{param_hash}.sh"
    
    try:
        create_job_script(script_path, args_str, script_name, walltime, memory)
        
        # Submit the job
        result = subprocess.run(['qsub', script_name], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split('.')[0]  # Clean job ID
            
            # Track this job automatically
            _submitted_jobs.add(job_id)
            
            # Associate with pattern if provided
            if pattern:
                if pattern not in _job_patterns:
                    _job_patterns[pattern] = set()
                _job_patterns[pattern].add(job_id)
            
            print(f"Submitted job {job_id}: {safe_job_name}")
            return job_id
        else:
            print(f"Failed to submit job {safe_job_name}: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout submitting job {safe_job_name}")
        return None
    except Exception as e:
        print(f"Error submitting job {safe_job_name}: {e}")
        return None


def count_jobs_by_pattern(pattern: str, user: Optional[str] = None) -> Dict[str, int]:
    """
    Count jobs matching a pattern by status.
    
    Uses the most appropriate method:
    1. If we have tracked jobs for this pattern, check those directly (fastest)
    2. Otherwise, use qstat -f to get full job names for pattern matching
    """
    global _job_patterns
    
    # Method 1: Use tracked jobs if available (fastest and most accurate)
    if pattern in _job_patterns:
        tracked_jobs = _job_patterns[pattern].copy()  # Copy to avoid modification during iteration
        counts = {}
        
        for job_id in tracked_jobs:
            status = get_job_status(job_id)
            if status:
                counts[status] = counts.get(status, 0) + 1
            else:
                # Job no longer exists, remove from tracking
                _job_patterns[pattern].discard(job_id)
                _submitted_jobs.discard(job_id)
        
        return counts
    
    # Method 2: Pattern matching with full job names (slower but comprehensive)
    jobs = get_all_jobs_with_full_names(user)
    counts = {}
    
    for job_id, job_name, status in jobs:
        if pattern in job_name:
            counts[status] = counts.get(status, 0) + 1
    
    return counts


def get_tracked_job_summary() -> Dict[str, int]:
    """
    Get summary of all jobs we've submitted and are tracking.
    This is the fastest method for monitoring your own jobs.
    """
    global _submitted_jobs
    
    if not _submitted_jobs:
        return {'total': 0, 'running': 0, 'queued': 0, 'completed': 0, 'unknown': 0}
    
    # Check status of all tracked jobs
    tracked_jobs = _submitted_jobs.copy()  # Copy to avoid modification during iteration
    summary = {'running': 0, 'queued': 0, 'completed': 0, 'unknown': 0}
    
    for job_id in tracked_jobs:
        status = get_job_status(job_id)
        if status == 'R':
            summary['running'] += 1
        elif status == 'Q':
            summary['queued'] += 1
        elif status is None:
            summary['completed'] += 1
            _submitted_jobs.discard(job_id)  # Remove completed jobs
        else:
            summary['unknown'] += 1
    
    summary['total'] = sum(summary.values())
    return summary


def wait_for_job_completion(pattern: Optional[str] = None, max_wait_time: int = 7200, 
                           check_interval: int = 30) -> bool:
    """
    Wait for jobs to complete.
    
    Parameters:
    -----------
    pattern : str, optional
        If provided, wait for jobs matching this pattern
        If None, wait for all tracked jobs
    max_wait_time : int
        Maximum time to wait in seconds
    check_interval : int
        Time between checks in seconds
        
    Returns:
    --------
    bool
        True if all jobs completed, False if timeout
    """
    start_time = time.time()
    
    print(f"Waiting for job completion...")
    if pattern:
        print(f"  Pattern: '{pattern}'")
    else:
        print(f"  All tracked jobs")
    
    while time.time() - start_time < max_wait_time:
        if pattern:
            counts = count_jobs_by_pattern(pattern)
            active_jobs = counts.get('R', 0) + counts.get('Q', 0)
            total_jobs = sum(counts.values())
        else:
            summary = get_tracked_job_summary()
            active_jobs = summary['running'] + summary['queued']
            total_jobs = summary['total']
        
        if active_jobs == 0:
            if total_jobs > 0:
                print(f"✓ All jobs completed!")
            else:
                print("No jobs found or all completed")
            return True
        
        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] (+{elapsed:5.0f}s) Active: {active_jobs}, Total: {total_jobs}")
        time.sleep(check_interval)
    
    print(f"⚠ Timeout after {max_wait_time} seconds")
    return False


def monitor_jobs_live(pattern: Optional[str] = None, check_interval: int = 30) -> None:
    """
    Monitor jobs with live updates until completion or user interruption.
    
    Parameters:
    -----------
    pattern : str, optional
        Pattern to monitor. If None, monitors all tracked jobs.
    check_interval : int
        Seconds between status updates
    """
    print("Job Monitoring")
    print("=" * 30)
    
    if pattern:
        print(f"Monitoring jobs with pattern: '{pattern}'")
    else:
        print("Monitoring all tracked jobs")
    
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        start_time = time.time()
        
        while True:
            if pattern:
                counts = count_jobs_by_pattern(pattern)
                active = counts.get('R', 0) + counts.get('Q', 0)
                total = sum(counts.values())
                running = counts.get('R', 0)
                queued = counts.get('Q', 0)
                completed = counts.get('C', 0)
            else:
                summary = get_tracked_job_summary()
                active = summary['running'] + summary['queued']
                total = summary['total']
                running = summary['running']
                queued = summary['queued']
                completed = summary['completed']
            
            elapsed = time.time() - start_time
            
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"(+{elapsed:5.0f}s) "
                  f"Total: {total}, "
                  f"Running: {running}, "
                  f"Queued: {queued}, "
                  f"Completed: {completed}")
            
            if active == 0:
                if total > 0:
                    print(f"\n✓ All jobs completed in {elapsed:.0f} seconds!")
                else:
                    print("\nNo jobs found or all completed")
                break
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


# Alias for backwards compatibility
monitor_jobs = monitor_jobs_live


def get_job_status_summary(pattern: Optional[str] = None) -> Dict[str, int]:
    """
    Get a quick status summary.
    
    Parameters:
    -----------
    pattern : str, optional
        Pattern to check. If None, checks all tracked jobs.
        
    Returns:
    --------
    Dict[str, int]
        Dictionary with status counts
    """
    if pattern:
        return count_jobs_by_pattern(pattern)
    else:
        return get_tracked_job_summary()


def cleanup_tracking() -> None:
    """Clear all job tracking data."""
    global _submitted_jobs, _job_patterns
    _submitted_jobs.clear()
    _job_patterns.clear()
    print("Cleared job tracking data")


class JobManager:
    """
    Simple job manager that provides batch submission with automatic rate limiting.
    All the complexity is handled by the underlying torque functions.
    """
    
    def __init__(self, max_concurrent: int = 100, check_interval: int = 30):
        self.max_concurrent = max_concurrent
        self.check_interval = check_interval
        self.pattern = None
    
    def submit_batch(self, script_path: str, param_list: List[Dict], 
                    job_prefix: str, pattern: Optional[str] = None, **job_kwargs) -> int:
        """
        Submit a batch of jobs with automatic rate limiting.
        
        Returns:
        --------
        int
            Number of successfully submitted jobs
        """
        self.pattern = pattern or job_prefix
        submitted_count = 0
        
        for i, params in enumerate(param_list):
            # Wait for available slots
            while True:
                summary = get_job_status_summary(self.pattern)
                active = summary.get('running', 0) + summary.get('queued', 0)
                
                if active < self.max_concurrent:
                    break
                
                print(f"Waiting for slots: {active}/{self.max_concurrent} active")
                time.sleep(self.check_interval)
            
            # Submit job
            job_name = f"{job_prefix}_{i}"
            job_id = submit_job(script_path, params, job_name, pattern=self.pattern, **job_kwargs)
            
            if job_id:
                submitted_count += 1
            
            # Small delay to avoid overwhelming scheduler
            time.sleep(1)
        
        return submitted_count
    
    def monitor(self) -> None:
        """Monitor the jobs submitted by this manager."""
        if self.pattern:
            monitor_jobs_live(self.pattern, self.check_interval)
        else:
            monitor_jobs_live(check_interval=self.check_interval)
    
    def wait_for_completion(self, timeout: int = 7200) -> bool:
        """Wait for all jobs to complete."""
        return wait_for_job_completion(self.pattern, timeout, self.check_interval)