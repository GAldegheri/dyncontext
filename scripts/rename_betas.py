import os
import shutil
import re
from pathlib import Path
from glob import glob
import scipy.io
import argparse
import sys

MODEL_ALIASES = {
    'test':
        {
            5: 'exp1_cong_incong',
            28: 'exp1_full_model',
            30: 'exp1_cong_incong_thirds'
        },
    'train':
        {
            4: 'exp1_viewspec_training'
        },
    'funcloc':
        {
            3: 'exp1_objscr_baseline'
        }
}

def load_spm_mat(spm_path):
    """Load SPM.mat file and extract regressor names and beta file names."""
    try:
        spm_data = scipy.io.loadmat(spm_path, struct_as_record=False, squeeze_me=True)
        spm = spm_data['SPM']
        
        # Extract regressor names (remove 'Sn(X) ' prefix and '*bf(1)' suffix)
        regr_names = []
        for name in spm.xX.name:
            # Remove 'Sn(X) ' prefix
            clean_name = re.sub(r'^Sn\(\d+\) ', '', name)
            # Remove '*bf(1)' suffix if present
            if '*bf(1)' in clean_name:
                clean_name = clean_name.replace('*bf(1)', '')
            regr_names.append(clean_name)
        
        # Extract beta file names
        beta_files = [beta.fname for beta in spm.Vbeta]
        
        return regr_names, beta_files
    except Exception as e:
        print(f"Error loading SPM.mat: {e}")
        return None, None

def create_condition_mapping(task, model_no):
    """Create mapping from original condition names to new naming scheme."""
    if task == 'test' and model_no == 28:
        mapping = {
            'A_30_exp_1': 'A_wide_congruent_split1',
            'A_30_exp_2': 'A_wide_congruent_split2', 
            'A_30_exp_3': 'A_wide_congruent_split3',
            'A_30_unexp': 'A_wide_incongruent',
            'A_90_exp_1': 'A_narrow_congruent_split1',
            'A_90_exp_2': 'A_narrow_congruent_split2',
            'A_90_exp_3': 'A_narrow_congruent_split3', 
            'A_90_unexp': 'A_narrow_incongruent',
            'B_30_exp_1': 'B_wide_congruent_split1',
            'B_30_exp_2': 'B_wide_congruent_split2',
            'B_30_exp_3': 'B_wide_congruent_split3',
            'B_30_unexp': 'B_wide_incongruent',
            'B_90_exp_1': 'B_narrow_congruent_split1',
            'B_90_exp_2': 'B_narrow_congruent_split2',
            'B_90_exp_3': 'B_narrow_congruent_split3',
            'B_90_unexp': 'B_narrow_incongruent'
        }
    elif task == 'test' and model_no == 5:
        mapping = {
            'expected': 'congruent',
            'unexpected': 'incongruent'
        }
    elif task == 'test' and model_no == 30:
        mapping = {
            'expected_1': 'congruent_split1',
            'expected_2': 'congruent_split2',
            'expected_3': 'congruent_split3',
            'unexpected': 'incongruent'
        }
    elif task == 'train' and model_no == 4:
        mapping = {
            'A30_1': 'A_wide_mb01',
            'A30_2': 'A_wide_mb02',
            'A30_3': 'A_wide_mb03',
            'A30_4': 'A_wide_mb04',
            'A30_5': 'A_wide_mb05',
            # -----------------------
            'A90_1': 'A_narrow_mb01',
            'A90_2': 'A_narrow_mb02',
            'A90_3': 'A_narrow_mb03',
            'A90_4': 'A_narrow_mb04',
            'A90_5': 'A_narrow_mb05',
            # -----------------------
            'B30_1': 'B_narrow_mb01',
            'B30_2': 'B_narrow_mb02',
            'B30_3': 'B_narrow_mb03',
            'B30_4': 'B_narrow_mb04',
            'B30_5': 'B_narrow_mb05',
            # -----------------------
            'B90_1': 'B_wide_mb01',
            'B90_2': 'B_wide_mb02',
            'B90_3': 'B_wide_mb03',
            'B90_4': 'B_wide_mb04',
            'B90_5': 'B_wide_mb05',
        }
    else:
        raise ValueError(f'Task {task}, Model {model_no} not valid.')
    return mapping

def is_nuisance_regressor(regressor_name):
    """Check if a regressor is a nuisance variable that should be excluded."""
    nuisance_patterns = [
        'buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz',
        'run', 'motion', 'realign', 'drift'
    ]
    
    regressor_lower = regressor_name.lower()
    return any(pattern in regressor_lower for pattern in nuisance_patterns)

def reorganize_betas(subject_id, source_base_dir, target_base_dir, task, model_no, dry_run=False):
    """
    Reorganize and rename beta files for a single subject.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    source_base_dir : str or Path
        Base directory containing the original beta files
    target_base_dir : str or Path  
        Base directory for the reorganized structure
    dry_run : bool
        If True, only print what would be done without actually moving files
    """
    
    # Construct paths
    source_dir = f'model_{model_no:g}'
    source_dir = Path(source_base_dir) / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas' / subject_id / task / source_dir
    target_dir = MODEL_ALIASES[task][model_no]
    target_dir = Path(target_base_dir) / 'experiment_1' / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas' / subject_id / task / target_dir
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}")
        return False
    
    # Load SPM.mat file
    spm_path = source_dir / 'SPM.mat'
    if not spm_path.exists():
        print(f"SPM.mat file not found: {spm_path}")
        return False
    
    regr_names, beta_files = load_spm_mat(spm_path)
    if regr_names is None or beta_files is None:
        print(f"Failed to load SPM.mat for {subject_id}")
        return False
    
    # Create condition mapping
    condition_mapping = create_condition_mapping(model_no)
    
    # Create target directory if it doesn't exist
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created target directory: {target_dir}")
    else:
        print(f"Would create target directory: {target_dir}")
    
    # Process each beta file
    moved_count = 0
    skipped_count = 0
    
    run_counts = {}
    
    for i, (regressor_name, beta_filename) in enumerate(zip(regr_names, beta_files)):
        
        source_beta_path = source_dir / beta_filename
        
        # Skip nuisance regressors
        if is_nuisance_regressor(regressor_name):
            print(f"Skipping nuisance regressor: {regressor_name}")
            skipped_count += 1
            continue
        
        # Check if this regressor has a new name mapping
        if regressor_name in condition_mapping:
            if regressor_name not in run_counts.keys():
                run_counts[regressor_name] = 1
            else:
                run_counts[regressor_name] += 1
            
            new_name = condition_mapping[regressor_name]
            # Extract beta number from original filename (e.g., beta_0001.nii -> 0001)
            beta_match = re.search(r'beta_(\d+)\.nii', beta_filename)
            if beta_match:
                beta_num = beta_match.group(1)
                new_filename = f"beta_{new_name}_run{run_counts[regressor_name]:02d}.nii"
            else:
                new_filename = f"{new_name}.nii"
            
            target_beta_path = target_dir / new_filename
            
            if dry_run:
                print(f"Would move: {source_beta_path} -> {target_beta_path}")
            else:
                try:
                    shutil.copy2(source_beta_path, target_beta_path)
                    print(f"Moved: {regressor_name} -> {new_name}")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {source_beta_path}: {e}")
        else:
            print(f"Warning: No mapping found for regressor '{regressor_name}' - skipping")
            skipped_count += 1
    
    print(f"\nSummary for {subject_id}:")
    print(f"  Beta files moved: {moved_count}")
    print(f"  Files skipped (nuisance or unmapped): {skipped_count}")
    
    return True

def get_subject_list(base_dir, model_no):
    """Get list of all subjects in the base directory."""
    betas_dir = Path(base_dir) / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas'
    
    if not betas_dir.exists():
        print(f"Betas directory does not exist: {betas_dir}")
        return []
    
    subjects = []
    for item in betas_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            model_dir = item / 'test' / f'model_{model_no:g}'
            if model_dir.exists():
                subjects.append(item.name)
    
    return sorted(subjects)

def main():
    parser = argparse.ArgumentParser(description='Reorganize and rename beta files for Experiment 1 MVPA analysis')
    parser.add_argument('--subject', '-s', type=str, help='Subject ID (e.g., sub-001). If not specified, processes all subjects.')
    parser.add_argument('--source-dir', type=str, default='/project/3018040.05/bids', 
                        help='Source base directory (default: /project/3018040.05/bids)')
    parser.add_argument('--target-dir', type=str, default='/project/3018040.05/dyncontext_bids',
                        help='Target base directory (default: /project/3018040.05/dyncontext_bids)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually moving files')
    parser.add_argument('--task', '-t', type=str, default='test',
                        help='Task to be transfered (default: test)')
    parser.add_argument('--model_no', '-m', type=int, default=28,
                        help='Model to be transfered (default: 28/full model)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be moved")
        print("-" * 50)
    
    if args.subject:
        # Process single subject
        subjects = [args.subject]
    else:
        # Process all subjects
        subjects = get_subject_list(source_dir, args.model_no)
        if not subjects:
            print("No subjects found in source directory")
            sys.exit(1)
        print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
    
    # Process each subject
    success_count = 0
    for subject in subjects:
        print(f"\nProcessing {subject}...")
        success = reorganize_betas(subject, source_dir, target_dir, args.task, args.model_no, dry_run=args.dry_run)
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed processing {success_count}/{len(subjects)} subjects successfully")
    
    if args.dry_run:
        print("\nThis was a dry run. Use --dry-run=False to actually move the files.")

if __name__ == "__main__":
    main()