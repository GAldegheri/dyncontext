#!/usr/bin/env python3
"""
Script to copy .tsv files from func subdirectories while maintaining directory structure.
Usage: python copy_tsv_files.py <source_dir> <target_dir> <subject_numbers>

Example:
python copy_tsv_files.py /path/to/source /path/to/target "001,002,003,016"
python copy_tsv_files.py /path/to/source /path/to/target "001-010,016"
"""

import os
import shutil
import sys
import glob
import argparse
from pathlib import Path


def parse_subject_numbers(subject_spec):
    """
    Parse subject number specification.
    Supports formats like:
    - "001,002,003" (comma-separated)
    - "001-005" (range)
    - "001-005,010,015-020" (mixed)
    """
    subjects = set()
    
    for part in subject_spec.split(','):
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # Handle range (e.g., "001-005")
            start, end = part.split('-', 1)
            start_num = int(start)
            end_num = int(end)
            for i in range(start_num, end_num + 1):
                subjects.add(f"{i:03d}")
        else:
            # Handle single number
            subjects.add(part)
    
    return sorted(subjects)


def copy_tsv_files(source_dir, target_dir, subject_numbers, dry_run=False):
    """
    Copy .tsv files from source_dir/sub-XXX/func/ to target_dir/sub-XXX/func/
    
    Args:
        source_dir (str): Source directory path
        target_dir (str): Target directory path  
        subject_numbers (list): List of subject numbers (e.g., ['001', '002'])
        dry_run (bool): If True, only print what would be copied without actually copying
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return False
    
    copied_files = 0
    skipped_subjects = []
    
    for sub_num in subject_numbers:
        sub_folder = f"sub-{sub_num}"
        source_func_dir = source_path / sub_folder / "func"
        target_func_dir = target_path / sub_folder / "func"
        
        if not source_func_dir.exists():
            print(f"Warning: Source func directory not found: {source_func_dir}")
            skipped_subjects.append(sub_num)
            continue
            
        # Find all .tsv files in the func directory
        tsv_files = list(source_func_dir.glob("*.tsv"))
        
        if not tsv_files:
            print(f"No .tsv files found in {source_func_dir}")
            continue
            
        # Create target directory if it doesn't exist
        if not dry_run:
            target_func_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy each .tsv file
        for tsv_file in tsv_files:
            target_file = target_func_dir / tsv_file.name
            
            if dry_run:
                print(f"Would copy: {tsv_file} -> {target_file}")
            else:
                try:
                    shutil.copy2(tsv_file, target_file)
                    print(f"Copied: {tsv_file.name} -> {target_file}")
                    copied_files += 1
                except Exception as e:
                    print(f"Error copying {tsv_file}: {e}")
    
    if not dry_run:
        print(f"\nSummary: Copied {copied_files} .tsv files")
    else:
        print(f"\nDry run completed. Would copy files for {len(subject_numbers) - len(skipped_subjects)} subjects")
    
    if skipped_subjects:
        print(f"Skipped subjects (func dir not found): {', '.join(skipped_subjects)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Copy .tsv files from func subdirectories while maintaining structure",
        epilog="""
Examples:
  %(prog)s /source /target "001,002,003"
  %(prog)s /source /target "001-010"
  %(prog)s /source /target "001-005,010,015-020"
  %(prog)s /source /target "001-010" --dry-run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('source_dir', help='Source directory path')
    parser.add_argument('target_dir', help='Target directory path')
    parser.add_argument('subjects', help='Subject numbers (e.g., "001,002,003" or "001-010")')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be copied without actually copying')
    
    args = parser.parse_args()
    
    # Parse subject numbers
    try:
        subject_numbers = parse_subject_numbers(args.subjects)
        print(f"Processing subjects: {', '.join(subject_numbers)}")
    except ValueError as e:
        print(f"Error parsing subject numbers: {e}")
        return 1
    
    # Perform the copy operation
    success = copy_tsv_files(args.source_dir, args.target_dir, subject_numbers, args.dry_run)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
