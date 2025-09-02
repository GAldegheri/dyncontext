"""
Enhanced SLURM Functions with Job Monitoring
============================================

Self-contained SLURM job management with automatic job tracking,
pattern matching with full job names, and intelligent monitoring.

The functions in this module handle all the complexity internally,
providing simple interfaces for job submission and monitoring.

Converted from Torque/PBS to SLURM scheduler.
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
                     walltime: str = "04:00:00", memory: str = "16G",
                     cpus: int = 1, partition: str = "normal",
                     account: Optional[str] = None) -> None:
    """
    Create a SLURM job script file.
    
    Parameters:
    -----------
    script_path : str
        Path to the Python script to run
    args_str : str
        Serialized arguments for the script
    script_name : str
        Name/path for the job script file
    walltime : str
        Maximum wall time (format: HH:MM:SS)
    memory : str
        Memory requirement (e.g., "16G", "32G")
    cpus : int
        Number of CPUs per task
    partition : str
        SLURM partition/queue to use
    account : str, optional
        Account to charge resources to (if required by cluster)
    """
    
    # Ensure output directory exists
    output_dir = Path("slurm/jobs/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create job script directory if it doesn't exist
    script_dir = Path(script_name).parent
    script_dir.mkdir(parents=True, exist_ok=True)
    
    job_basename = os.path.basename(script_name).replace('.sh', '')
    
    with open(script_name, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH --job-name={job_basename}\n")
        file.write(f"#SBATCH --time={walltime}\n")
        file.write(f"#SBATCH --mem={memory}\n")
        file.write(f"#SBATCH --cpus-per-task={cpus}\n")
        file.write(f"#SBATCH --partition={partition}\n")
        if account:
            file.write(f"#SBATCH --account={account}\n")
        file.write(f"#SBATCH --output=slurm/jobs/output/{job_basename}_%j_output.txt\n")
        file.write(f"#SBATCH --error=slurm/jobs/output/{job_basename}_%j_error.txt\n")
        file.write("\n# Job information\n")
        file.write('echo "Job started on $(hostname) at $(date)"\n')
        file.write('echo "SLURM_JOB_ID: $SLURM_JOB_ID"\n')
        file.write('echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"\n')
        file.write("\n# Change to analysis directory\n")
        file.write("cd /project/3018040.05/dyncontext/\n")
        file.write("\n# Activate conda environment\n")
        file.write("source activate dyncontext\n")
        file.write("\n# Run the Python script\n")
        file.write(f"python {script_path} {args_str}\n")
        file.write("\n# Log completion time\n")
        file.write('echo "Job completed at: $(date)"\n')
    
    # Make script executable
    os.chmod(script_name, 0o755)


def get_job_status(job_id: str) -> Optional[str]:
    """
    Get the status of a specific job by ID.
    
    Returns SLURM status codes:
    - PD: Pending
    - R: Running
    - CG: Completing
    - CD: Completed
    - F: Failed
    - None: Job not found (likely completed or cancelled)
    """
    try:
        # Use squeue to check job status
        result = subprocess.run(
            ['squeue', '--job', job_id, '--noheader', '--format=%T'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        
        # If not in queue, check accounting info for recently completed jobs
        result = subprocess.run(
            ['sacct', '-j', job_id, '--noheader', '--format=State', '-n'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # sacct might return multiple lines for job steps, get the first
            states = result.stdout.strip().split('\n')
            if states:
                return states[0].strip()
        
        return None
        
    except (subprocess.TimeoutExpired, Exception):
        return None


def _parse_squeue_output(squeue_output: str) -> List[Tuple[str, str, str]]:
    """Parse squeue output to extract job information."""
    jobs = []
    
    for line in squeue_output.strip().split('\n'):
        if not line:
            continue
        parts = line.split('|')
        if len(parts) >= 3:
            job_id = parts[0].strip()
            job_name = parts[1].strip()
            status = parts[2].strip()
            jobs.append((job_id, job_name, status))
    
    return jobs


def get_all_jobs_with_full_names(user: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    Get all jobs with complete job names using squeue.
    
    Returns:
    --------
    List of tuples (job_id, job_name, status)
    """
    try:
        cmd = ['squeue', '--noheader', '--format=%i|%j|%T']
        if user:
            cmd.extend(['--user', user])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return _parse_squeue_output(result.stdout)
        else:
            return []
            
    except (subprocess.TimeoutExpired, Exception):
        return []


def submit_job(script_path: str, kwargs: Dict, job_name: str,
               walltime: str = "04:00:00", memory: str = "16G",
               cpus: int = 1, partition: str = "normal",
               account: Optional[str] = None,
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
        Maximum wall time (HH:MM:SS)
    memory : str
        Memory requirement (e.g., "16G")
    cpus : int
        Number of CPUs per task
    partition : str
        SLURM partition to use
    account : str, optional
        Account to charge (if required)
    pattern : str, optional
        Pattern to associate this job with for group monitoring
        
    Returns:
    --------
    str or None
        Job ID if successful, None if failed
    """
    global _submitted_jobs, _job_patterns
    
    args_str = serialize_args(kwargs)
    
    # Create safe job name (SLURM has limitations on job names)
    safe_job_name = job_name.replace('/', '_').replace(' ', '_')[:50]
    
    # Create unique script name
    param_hash = abs(hash(str(sorted(kwargs.items())))) % 10000
    script_name = f"slurm/jobs/{safe_job_name}_{param_hash}.sh"
    
    try:
        create_job_script(script_path, args_str, script_name, 
                         walltime, memory, cpus, partition, account)
        
        # Submit the job
        result = subprocess.run(['sbatch', script_name], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Extract job ID from output (format: "Submitted batch job 12345")
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
            else:
                # Fallback for different output formats
                job_id = output.strip()
            
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
    2. Otherwise, use squeue to get full job names for pattern matching
    
    Returns dict with SLURM status codes as keys:
    - PENDING: Job is waiting in queue
    - RUNNING: Job is executing
    - COMPLETING: Job is finishing
    - COMPLETED: Job finished successfully
    - FAILED: Job failed
    - CANCELLED: Job was cancelled
    """
    global _job_patterns
    
    # Method 1: Use tracked jobs if available (fastest and most accurate)
    if pattern in _job_patterns:
        tracked_jobs = _job_patterns[pattern].copy()  # Copy to avoid modification during iteration
        counts = {}
        
        for job_id in tracked_jobs:
            status = get_job_status(job_id)
            if status:
                # Normalize status codes
                if status in ['PENDING', 'PD']:
                    counts['PENDING'] = counts.get('PENDING', 0) + 1
                elif status in ['RUNNING', 'R']:
                    counts['RUNNING'] = counts.get('RUNNING', 0) + 1
                elif status in ['COMPLETING', 'CG']:
                    counts['COMPLETING'] = counts.get('COMPLETING', 0) + 1
                elif status in ['COMPLETED', 'CD']:
                    counts['COMPLETED'] = counts.get('COMPLETED', 0) + 1
                    _job_patterns[pattern].discard(job_id)
                    _submitted_jobs.discard(job_id)
                elif status in ['FAILED', 'F']:
                    counts['FAILED'] = counts.get('FAILED', 0) + 1
                    _job_patterns[pattern].discard(job_id)
                    _submitted_jobs.discard(job_id)
                else:
                    counts[status] = counts.get(status, 0) + 1
            else:
                # Job no longer exists, assume completed
                counts['COMPLETED'] = counts.get('COMPLETED', 0) + 1
                _job_patterns[pattern].discard(job_id)
                _submitted_jobs.discard(job_id)
        
        return counts
    
    # Method 2: Pattern matching with full job names (slower but comprehensive)
    jobs = get_all_jobs_with_full_names(user)
    counts = {}
    
    for job_id, job_name, status in jobs:
        if pattern in job_name:
            # Normalize status codes
            if status in ['PENDING', 'PD']:
                counts['PENDING'] = counts.get('PENDING', 0) + 1
            elif status in ['RUNNING', 'R']:
                counts['RUNNING'] = counts.get('RUNNING', 0) + 1
            elif status in ['COMPLETING', 'CG']:
                counts['COMPLETING'] = counts.get('COMPLETING', 0) + 1
            else:
                counts[status] = counts.get(status, 0) + 1
    
    return counts


def get_tracked_job_summary() -> Dict[str, int]:
    """
    Get summary of all jobs we've submitted and are tracking.
    This is the fastest method for monitoring your own jobs.
    """
    global _submitted_jobs
    
    if not _submitted_jobs:
        return {'total': 0, 'running': 0, 'pending': 0, 'completed': 0, 'failed': 0, 'unknown': 0}
    
    # Check status of all tracked jobs
    tracked_jobs = _submitted_jobs.copy()  # Copy to avoid modification during iteration
    summary = {'running': 0, 'pending': 0, 'completed': 0, 'failed': 0, 'unknown': 0}
    
    for job_id in tracked_jobs:
        status = get_job_status(job_id)
        if status in ['RUNNING', 'R']:
            summary['running'] += 1
        elif status in ['PENDING', 'PD']:
            summary['pending'] += 1
        elif status in ['COMPLETED', 'CD']:
            summary['completed'] += 1
            _submitted_jobs.discard(job_id)  # Remove completed jobs
        elif status in ['FAILED', 'F', 'CANCELLED', 'CA']:
            summary['failed'] += 1
            _submitted_jobs.discard(job_id)  # Remove failed jobs
        elif status is None:
            summary['completed'] += 1  # Assume completed if not found
            _submitted_jobs.discard(job_id)
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
            active_jobs = counts.get('RUNNING', 0) + counts.get('PENDING', 0) + counts.get('COMPLETING', 0)
            total_jobs = sum(counts.values())
            failed_jobs = counts.get('FAILED', 0)
        else:
            summary = get_tracked_job_summary()
            active_jobs = summary['running'] + summary['pending']
            total_jobs = summary['total']
            failed_jobs = summary['failed']
        
        if active_jobs == 0:
            if failed_jobs > 0:
                print(f"⚠ Warning: {failed_jobs} jobs failed")
            if total_jobs > 0:
                print(f"✓ All jobs completed!")
            else:
                print("No jobs found or all completed")
            return True
        
        elapsed = time.time() - start_time
        status_str = f"[{time.strftime('%H:%M:%S')}] (+{elapsed:5.0f}s) Active: {active_jobs}, Total: {total_jobs}"
        if failed_jobs > 0:
            status_str += f", Failed: {failed_jobs}"
        print(status_str)
        time.sleep(check_interval)
    
    print(f"⚠ Timeout after {max_wait_time} seconds")
    return False


def monitor_jobs_live(pattern: Optional[str] = None, check_interval: int = 30,
                     user: Optional[str] = None) -> None:
    """
    Monitor jobs with live updates until completion or user interruption.
    
    Parameters:
    -----------
    pattern : str, optional
        Pattern to monitor. If None, monitors all tracked jobs.
    check_interval : int
        Seconds between status updates
    user : str, optional
        Username to filter jobs (useful for shared clusters)
    """
    print("SLURM Job Monitoring")
    print("=" * 30)
    
    if pattern:
        print(f"Monitoring jobs with pattern: '{pattern}'")
    else:
        print("Monitoring all tracked jobs")
    
    if user:
        print(f"User: {user}")
    
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        start_time = time.time()
        
        while True:
            if pattern:
                counts = count_jobs_by_pattern(pattern, user)
                active = counts.get('RUNNING', 0) + counts.get('PENDING', 0) + counts.get('COMPLETING', 0)
                total = sum(counts.values())
                running = counts.get('RUNNING', 0)
                pending = counts.get('PENDING', 0)
                completed = counts.get('COMPLETED', 0)
                failed = counts.get('FAILED', 0)
            else:
                summary = get_tracked_job_summary()
                active = summary['running'] + summary['pending']
                total = summary['total']
                running = summary['running']
                pending = summary['pending']
                completed = summary['completed']
                failed = summary['failed']
            
            elapsed = time.time() - start_time
            
            status_msg = (f"[{time.strftime('%H:%M:%S')}] "
                         f"(+{elapsed:5.0f}s) "
                         f"Total: {total}, "
                         f"Running: {running}, "
                         f"Pending: {pending}, "
                         f"Completed: {completed}")
            
            if failed > 0:
                status_msg += f", Failed: {failed}"
            
            print(status_msg)
            
            if active == 0:
                if failed > 0:
                    print(f"\n⚠ Warning: {failed} jobs failed")
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


def get_job_status_summary(pattern: Optional[str] = None, user: Optional[str] = None) -> Dict[str, int]:
    """
    Get a quick status summary.
    
    Parameters:
    -----------
    pattern : str, optional
        Pattern to check. If None, checks all tracked jobs.
    user : str, optional
        Username to filter jobs
        
    Returns:
    --------
    Dict[str, int]
        Dictionary with status counts
    """
    if pattern:
        return count_jobs_by_pattern(pattern, user)
    else:
        return get_tracked_job_summary()


def cancel_jobs(pattern: Optional[str] = None, job_ids: Optional[List[str]] = None) -> int:
    """
    Cancel jobs by pattern or specific job IDs.
    
    Parameters:
    -----------
    pattern : str, optional
        Cancel all jobs matching this pattern
    job_ids : list, optional
        List of specific job IDs to cancel
        
    Returns:
    --------
    int
        Number of jobs cancelled
    """
    global _submitted_jobs, _job_patterns
    
    jobs_to_cancel = []
    
    if pattern and pattern in _job_patterns:
        jobs_to_cancel = list(_job_patterns[pattern])
    elif job_ids:
        jobs_to_cancel = job_ids
    elif pattern:
        # Search for jobs matching pattern
        all_jobs = get_all_jobs_with_full_names()
        jobs_to_cancel = [job_id for job_id, job_name, _ in all_jobs if pattern in job_name]
    else:
        # Cancel all tracked jobs
        jobs_to_cancel = list(_submitted_jobs)
    
    cancelled_count = 0
    for job_id in jobs_to_cancel:
        try:
            result = subprocess.run(['scancel', job_id], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                cancelled_count += 1
                _submitted_jobs.discard(job_id)
                # Remove from pattern tracking
                for pat in _job_patterns:
                    _job_patterns[pat].discard(job_id)
        except (subprocess.TimeoutExpired, Exception):
            pass
    
    print(f"Cancelled {cancelled_count} jobs")
    return cancelled_count


def cleanup_tracking() -> None:
    """Clear all job tracking data."""
    global _submitted_jobs, _job_patterns
    _submitted_jobs.clear()
    _job_patterns.clear()
    print("Cleared job tracking data")


def get_job_info(job_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific job.
    
    Returns:
    --------
    Dict with job information or None if job not found
    """
    try:
        # Try scontrol first for active jobs
        result = subprocess.run(['scontrol', 'show', 'job', job_id],
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout:
            info = {}
            for line in result.stdout.split('\n'):
                for item in line.split():
                    if '=' in item:
                        key, value = item.split('=', 1)
                        info[key] = value
            return info
        
        # Try sacct for completed jobs
        result = subprocess.run(
            ['sacct', '-j', job_id, '--format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS', '--parsable2'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Header + data
                headers = lines[0].split('|')
                values = lines[1].split('|')
                return dict(zip(headers, values))
        
        return None
        
    except (subprocess.TimeoutExpired, Exception):
        return None


class JobManager:
    """
    Simple job manager that provides batch submission with automatic rate limiting.
    All the complexity is handled by the underlying SLURM functions.
    """
    
    def __init__(self, max_concurrent: int = 100, check_interval: int = 30, 
                 user: Optional[str] = None):
        self.max_concurrent = max_concurrent
        self.check_interval = check_interval
        self.pattern = None
        self.user = user
    
    def submit_batch(self, script_path: str, param_list: List[Dict], 
                    job_prefix: str, pattern: Optional[str] = None, 
                    stagger_delay: float = 0.5, **job_kwargs) -> int:
        """
        Submit a batch of jobs with automatic rate limiting.
        
        Parameters:
        -----------
        script_path : str
            Path to the Python script
        param_list : list
            List of parameter dictionaries
        job_prefix : str
            Prefix for job names
        pattern : str, optional
            Pattern for tracking these jobs
        stagger_delay : float
            Delay between submissions to avoid overwhelming scheduler
        **job_kwargs : additional keyword arguments for submit_job
            (walltime, memory, cpus, partition, account)
        
        Returns:
        --------
        int
            Number of successfully submitted jobs
        """
        self.pattern = pattern or job_prefix
        submitted_count = 0
        failed_submissions = []
        
        print(f"Submitting {len(param_list)} jobs with prefix '{job_prefix}'")
        
        for i, params in enumerate(param_list):
            # Wait for available slots
            while True:
                summary = get_job_status_summary(self.pattern, self.user)
                active = summary.get('RUNNING', 0) + summary.get('running', 0) + \
                        summary.get('PENDING', 0) + summary.get('pending', 0)
                
                if active < self.max_concurrent:
                    break
                
                print(f"Waiting for slots: {active}/{self.max_concurrent} active")
                time.sleep(self.check_interval)
            
            # Submit job
            job_name = f"{job_prefix}_{i:04d}"
            job_id = submit_job(script_path, params, job_name, pattern=self.pattern, **job_kwargs)
            
            if job_id:
                submitted_count += 1
                print(f"  [{submitted_count}/{len(param_list)}] Submitted {job_name} (ID: {job_id})")
            else:
                failed_submissions.append((i, params))
                print(f"  [FAILED] Could not submit job {i}")
            
            # Small delay to avoid overwhelming scheduler
            time.sleep(stagger_delay)
        
        print(f"\nBatch submission complete:")
        print(f"  Successfully submitted: {submitted_count}")
        print(f"  Failed submissions: {len(failed_submissions)}")
        
        if failed_submissions:
            print("\nFailed job indices:", [i for i, _ in failed_submissions])
        
        return submitted_count
    
    def monitor(self) -> None:
        """Monitor the jobs submitted by this manager."""
        if self.pattern:
            monitor_jobs_live(self.pattern, self.check_interval, self.user)
        else:
            monitor_jobs_live(check_interval=self.check_interval, user=self.user)
    
    def wait_for_completion(self, timeout: int = 7200) -> bool:
        """Wait for all jobs to complete."""
        return wait_for_job_completion(self.pattern, timeout, self.check_interval)
    
    def cancel_all(self) -> int:
        """Cancel all jobs managed by this instance."""
        return cancel_jobs(pattern=self.pattern)
    
    def get_summary(self) -> Dict[str, int]:
        """Get current status summary."""
        return get_job_status_summary(self.pattern, self.user)


# Utility functions for common SLURM operations

def check_queue_status(partition: Optional[str] = None) -> Dict:
    """
    Check the overall status of the SLURM queue.
    
    Parameters:
    -----------
    partition : str, optional
        Specific partition to check
        
    Returns:
    --------
    Dict with queue information
    """
    try:
        cmd = ['sinfo']
        if partition:
            cmd.extend(['-p', partition])
        cmd.extend(['--format=%P|%a|%l|%D|%T|%N', '--noheader'])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            queue_info = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split('|')
                if len(parts) >= 5:
                    queue_info.append({
                        'partition': parts[0].rstrip('*'),
                        'available': parts[1],
                        'timelimit': parts[2],
                        'nodes': parts[3],
                        'state': parts[4],
                        'nodelist': parts[5] if len(parts) > 5 else ''
                    })
            return {'queues': queue_info}
        return {}
        
    except (subprocess.TimeoutExpired, Exception):
        return {}


def estimate_wait_time(partition: str = "normal", memory: str = "16G", cpus: int = 1) -> Optional[str]:
    """
    Estimate wait time for a job with given resources.
    This is a simple heuristic based on queue depth.
    
    Returns:
    --------
    Estimated wait time as string or None
    """
    try:
        # Count pending jobs in partition
        result = subprocess.run(
            ['squeue', '-p', partition, '-t', 'PENDING', '--noheader', '-r'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            pending_jobs = len(result.stdout.strip().split('\n'))
            
            # Very rough estimate: assume 30 minutes per job in queue ahead
            estimated_minutes = pending_jobs * 30
            
            if estimated_minutes < 60:
                return f"~{estimated_minutes} minutes"
            else:
                return f"~{estimated_minutes // 60} hours"
        
        return None
        
    except (subprocess.TimeoutExpired, Exception):
        return None


# Compatibility layer for easy migration from Torque

def torque_to_slurm_params(torque_params: Dict) -> Dict:
    """
    Convert Torque/PBS parameters to SLURM equivalents.
    
    Mappings:
    - nodes=1:ppn=X → cpus=X
    - mem=XGb → memory=XG
    - walltime remains the same
    - queue → partition
    """
    slurm_params = {}
    
    # Handle nodes specification
    if 'nodes' in torque_params:
        nodes_spec = torque_params['nodes']
        if ':ppn=' in nodes_spec:
            cpus = int(nodes_spec.split(':ppn=')[1])
            slurm_params['cpus'] = cpus
    
    # Handle memory
    if 'mem' in torque_params or 'memory' in torque_params:
        mem = torque_params.get('mem', torque_params.get('memory', '16g'))
        # Convert format if needed
        if mem.endswith('gb'):
            mem = mem[:-2] + 'G'
        elif mem.endswith('g'):
            mem = mem[:-1] + 'G'
        slurm_params['memory'] = mem
    
    # Handle walltime
    if 'walltime' in torque_params:
        slurm_params['walltime'] = torque_params['walltime']
    
    # Handle queue/partition
    if 'queue' in torque_params:
        slurm_params['partition'] = torque_params['queue']
    
    return slurm_params