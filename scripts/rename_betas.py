import os
import shutil
import re
from pathlib import Path
from glob import glob
import scipy.io
import argparse
import sys

MODEL_ALIASES = {
    'experiment_1': {
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
    },
    'experiment_2': {
        'test':
            {
                4: 'exp2_wide_narrow'
            },
        'train':
            {
                6: 'exp2_widenarr_training'
            },
        'funcloc':
            {
                3: 'exp2_objscr_baseline'
            }
    }
}

CONTRAST_ALIASES = {
    'experiment_1': {
        'funcloc': {
            3: 
                {1: 'objscr-vs-baseline'}
        },
        'test': {
            5: 
                {1: 'incong-vs-cong',
                2: 'cong-vs-incong'}
        }
    },
    'experiment_2': {
        'funcloc': {
            3:
                {1: 'objscr-vs-baseline'}
        }
    }
}

def load_spm_mat(spm_path, fir=False):
    """Load SPM.mat file and extract regressor names and beta file names."""
    try:
        spm_data = scipy.io.loadmat(spm_path, struct_as_record=False, squeeze_me=True)
        spm = spm_data['SPM']
        
        # Extract regressor names (remove 'Sn(X) ' prefix and '*bf(1)' suffix)
        regr_names = []
        delays = [] if fir else None
        for name in spm.xX.name:
            # Remove 'Sn(X) ' prefix
            clean_name = re.sub(r'^Sn\(\d+\) ', '', name)
            # Remove '*bf(1)' suffix if present
            if fir:
                match = re.search(r"bf\((\d+)\)", clean_name)
                if match:
                    delay = int(match.group(1))
                    delays.append(delay)
                    clean_name = re.sub(r"\*bf\(\d+\)", "", clean_name)
                else:
                    delays.append(None)
            else:
                if '*bf(1)' in clean_name:
                    clean_name = clean_name.replace('*bf(1)', '')
            regr_names.append(clean_name)
        
        # Extract beta file names
        beta_files = [beta.fname for beta in spm.Vbeta]
        
        return regr_names, beta_files, delays
    except Exception as e:
        print(f"Error loading SPM.mat: {e}")
        return None, None, None

def create_condition_mapping(experiment, task, model_no):
    """Create mapping from original condition names to new naming scheme."""
    if experiment == 1:
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
        elif task == 'funcloc' and model_no == 3:
            mapping = {
                'objscr': 'objscr',
                'baseline': 'baseline'
            }
        else:
            raise ValueError(f'Task {task}, Model {model_no} not valid for experiment 1.')
    elif experiment == 2:
        if task == 'test' and model_no == 4:
            mapping = {
                'wide': 'wide',
                'narrow': 'narrow'
            }
        elif task == 'train' and model_no == 6:
            mapping = {
                'wide_1': 'wide_mb01',
                'wide_2': 'wide_mb02',
                'wide_3': 'wide_mb03',
                'wide_4': 'wide_mb04',
                'wide_5': 'wide_mb05',
                'wide_6': 'wide_mb06',
                'wide_7': 'wide_mb07',
                'wide_8': 'wide_mb08',
                'wide_9': 'wide_mb09',
                'wide_10': 'wide_mb10',
                # --------------------
                'narrow_1': 'narrow_mb01',
                'narrow_2': 'narrow_mb02',
                'narrow_3': 'narrow_mb03',
                'narrow_4': 'narrow_mb04',
                'narrow_5': 'narrow_mb05',
                'narrow_6': 'narrow_mb06',
                'narrow_7': 'narrow_mb07',
                'narrow_8': 'narrow_mb08',
                'narrow_9': 'narrow_mb09',
                'narrow_10': 'narrow_mb10',
            }
        elif task == 'funcloc' and model_no == 3:
            mapping = {
                'objscr': 'objscr',
                'baseline': 'baseline'
            }
        else:
            raise ValueError(f'Task {task}, Model {model_no} not valid for experiment 2.')
    return mapping

def is_nuisance_regressor(regressor_name):
    """Check if a regressor is a nuisance variable that should be excluded."""
    nuisance_patterns = [
        'buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz',
        'run', 'motion', 'realign', 'drift'
    ]
    
    regressor_lower = regressor_name.lower()
    return any(pattern in regressor_lower for pattern in nuisance_patterns)

def reorganize_betas(subject_id, source_base_dir, target_base_dir, experiment, task, model_no, 
                     contrast_no=None, fir=False, dry_run=False, move=False):
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
    
    movefunc = shutil.move if move else shutil.copy2    
    
    # Construct paths
    source_dir = f'model_{model_no:g}'
    target_dir = MODEL_ALIASES[f'experiment_{experiment:g}'][task][model_no]
    if contrast_no:
        src_dir_contrast = Path(source_base_dir) / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'contrasts' / subject_id / task / source_dir
        tgt_dir_contrast = Path(target_base_dir) / f'experiment_{experiment:g}' / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'contrasts' / subject_id / task / target_dir
    source_dir = Path(source_base_dir) / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas' / subject_id / task / source_dir
    target_dir = Path(target_base_dir) / f'experiment_{experiment:g}' / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas' / subject_id / task / target_dir
    if fir:
        source_dir = source_dir / 'FIR'
        target_dir = target_dir / 'FIR'
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}")
        return False
    
    # Load SPM.mat file
    spm_path = source_dir / 'SPM.mat'
    if not spm_path.exists():
        print(f"SPM.mat file not found: {spm_path}")
        return False
    
    regr_names, beta_files, delays = load_spm_mat(spm_path, fir=fir)
    if regr_names is None or beta_files is None:
        print(f"Failed to load SPM.mat for {subject_id}")
        return False
    
    # Create condition mapping
    condition_mapping = create_condition_mapping(experiment, task, model_no)
    
    # Create target directory if it doesn't exist
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created target directory: {target_dir}")
        if contrast_no:
            tgt_dir_contrast.mkdir(parents=True, exist_ok=True)
            print(f"Created target contrast directory: {tgt_dir_contrast}")
    else:
        print(f"Would create target directory: {target_dir}")
        if contrast_no:
            print(f"Would create target contrast directory: {tgt_dir_contrast}")
    
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
                if not fir:
                    run_counts[regressor_name] += 1
                else:
                    if delays[i] == 1:
                        # only increase run count for first delay
                        run_counts[regressor_name] += 1
            
            new_name = condition_mapping[regressor_name]
            # Extract beta number from original filename (e.g., beta_0001.nii -> 0001)
            beta_match = re.search(r'beta_(\d+)\.nii', beta_filename)
            if beta_match:
                beta_num = beta_match.group(1)
                if fir:
                    new_filename = f"beta_{new_name}_delay{delays[i]:02d}_run{run_counts[regressor_name]:02d}.nii"
                else:
                    new_filename = f"beta_{new_name}_run{run_counts[regressor_name]:02d}.nii"
            else:
                new_filename = f"{new_name}.nii"
            
            target_beta_path = target_dir / new_filename
            
            if dry_run:
                print(f"Would move: {source_beta_path} -> {target_beta_path}")
            else:
                try:
                    movefunc(source_beta_path, target_beta_path)
                    print(f"Moved: {regressor_name} -> {new_name}")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {source_beta_path}: {e}")
        else:
            if fir:
                assert(delays[i]==None)
            print(f"Warning: No mapping found for regressor '{regressor_name}' - skipping")
            skipped_count += 1
    
    print(f"\nSummary for {subject_id}:")
    print(f"  Beta files moved: {moved_count}")
    print(f"  Files skipped (nuisance or unmapped): {skipped_count}")
    
    if contrast_no:
        
        source_contrast_path = src_dir_contrast / f'con_{contrast_no:04d}.nii'
        source_spmT_path = src_dir_contrast / f'spmT_{contrast_no:04d}.nii'
        
        new_contr_name = CONTRAST_ALIASES[f'experiment_{experiment:g}'][task][model_no][contrast_no]
        
        target_contrast_path = tgt_dir_contrast / f'con_{new_contr_name}.nii'
        target_spmT_path = tgt_dir_contrast / f'spmT_{new_contr_name}.nii'
        
        if dry_run:
            print("Would move:")
            print(f"{source_contrast_path} -> {target_contrast_path}")
            print(f"{source_spmT_path} -> {target_spmT_path}")
        else:
            try:
                movefunc(source_contrast_path, target_contrast_path)
                print(f"Moved: {source_contrast_path} -> {target_contrast_path}")
            except Exception as e:
                print(f"Error moving {source_contrast_path}: {e}")
            try:
                movefunc(source_spmT_path, target_spmT_path)
                print(f"Moved: {source_spmT_path} -> {target_spmT_path}")
            except Exception as e:
                print(f"Error moving {source_spmT_path}: {e}")
    
    return True

def get_subject_list(base_dir, task, model_no):
    """Get list of all subjects in the base directory."""
    betas_dir = Path(base_dir) / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas'
    
    if not betas_dir.exists():
        print(f"Betas directory does not exist: {betas_dir}")
        return []
    
    subjects = []
    for item in betas_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            model_dir = item / task / f'model_{model_no:g}'
            if model_dir.exists():
                subjects.append(item.name)
    
    return sorted(subjects)

def main():
    parser = argparse.ArgumentParser(description='Reorganize and rename beta files for Experiment 1 MVPA analysis')
    parser.add_argument('--subject', '-s', type=str, help='Subject ID (e.g., sub-001). If not specified, processes all subjects.')
    parser.add_argument('--source-dir', type=str, default='/project/3018040.07/bids', 
                        help='Source base directory (default: /project/3018040.07/bids)')
    parser.add_argument('--target-dir', type=str, default='/project/3018040.05/dyncontext_bids',
                        help='Target base directory (default: /project/3018040.05/dyncontext_bids)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually moving files')
    parser.add_argument('--experiment', '-e', type=int, default=1,
                        help='Experiment to be transfered (1 or 2, default 1)')
    parser.add_argument('--task', '-t', type=str, default='test',
                        help='Task to be transfered (default: test)')
    parser.add_argument('--model_no', '-m', type=int, default=28,
                        help='Model to be transfered (default: 28/full model)')
    parser.add_argument('--contrast_no', '-c', type=int, default=None,
                        help='Contrast to be transfered (default: None)')
    parser.add_argument('--fir', action='store_true',
                        help='Whether to use the FIR response function')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying them')
    
    
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
        subjects = get_subject_list(source_dir, args.task, args.model_no)
        if not subjects:
            print("No subjects found in source directory")
            sys.exit(1)
        print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
    
    # Process each subject
    success_count = 0
    for subject in subjects:
        print(f"\nProcessing {subject}...")
        success = reorganize_betas(subject, source_dir, target_dir, args.experiment,
                                   args.task, args.model_no, args.contrast_no,
                                   fir=args.fir, dry_run=args.dry_run, move=args.move)
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed processing {success_count}/{len(subjects)} subjects successfully")
    
    if args.dry_run:
        print("\nThis was a dry run. Remove --dry-run to actually move the files.")

if __name__ == "__main__":
    main()