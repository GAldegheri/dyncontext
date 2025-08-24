import shutil
import os
from glob import glob
import argparse
from pathlib import Path
import re

def get_mapping_dict(task):
    if task == 'test':

        mapping_dict = {
            'A_wide_congruent_split1': 'wide_30_congruent_split1',
            'A_wide_congruent_split2': 'wide_30_congruent_split2', 
            'A_wide_congruent_split3': 'wide_30_congruent_split3',
            'A_wide_incongruent': 'wide_30_incongruent',
            'A_narrow_congruent_split1': 'narrow_90_congruent_split1',
            'A_narrow_congruent_split2': 'narrow_90_congruent_split2',
            'A_narrow_congruent_split3': 'narrow_90_congruent_split3', 
            'A_narrow_incongruent': 'narrow_90_incongruent',
            
            'B_wide_congruent_split1': 'narrow_30_congruent_split1',
            'B_wide_congruent_split2': 'narrow_30_congruent_split2',
            'B_wide_congruent_split3': 'narrow_30_congruent_split3',
            'B_wide_incongruent': 'narrow_30_incongruent',
            'B_narrow_congruent_split1': 'wide_90_congruent_split1',
            'B_narrow_congruent_split2': 'wide_90_congruent_split2',
            'B_narrow_congruent_split3': 'wide_90_congruent_split3',
            'B_narrow_incongruent': 'wide_90_incongruent'
        }
        
    elif task == 'train':

        mapping_dict = {
            'A_wide_mb01': 'wide_30_mb01',
            'A_wide_mb02': 'wide_30_mb02',
            'A_wide_mb03': 'wide_30_mb03',
            'A_wide_mb04': 'wide_30_mb04',
            'A_wide_mb05': 'wide_30_mb05',
            # -----------------------
            'A_narrow_mb01': 'narrow_90_mb01',
            'A_narrow_mb02': 'narrow_90_mb02',
            'A_narrow_mb03': 'narrow_90_mb03',
            'A_narrow_mb04': 'narrow_90_mb04',
            'A_narrow_mb05': 'narrow_90_mb05',
            # -----------------------
            'B_narrow_mb01': 'narrow_30_mb01',
            'B_narrow_mb02': 'narrow_30_mb02',
            'B_narrow_mb03': 'narrow_30_mb03',
            'B_narrow_mb04': 'narrow_30_mb04',
            'B_narrow_mb05': 'narrow_30_mb05',
            # -----------------------
            'B_wide_mb01': 'wide_90_mb01',
            'B_wide_mb02': 'wide_90_mb02',
            'B_wide_mb03': 'wide_90_mb03',
            'B_wide_mb04': 'wide_90_mb04',
            'B_wide_mb05': 'wide_90_mb05'
        }
        
    return mapping_dict

def rename_betas(subject_id, betas_dir, task, fir=False, dry_run=False):
    
    
    if fir:
        pattern = r"beta_(.+?)_delay(\d+)_run(\d+)\.nii"
    else:
        pattern = r"beta_(.+?)_run(\d+)\.nii"
    
    if task == 'test':
        model_dir = Path(betas_dir) / subject_id / 'test' / 'exp1_full_model'
        if fir:
            model_dir = model_dir / 'FIR'
    elif task == 'train':
        model_dir = Path(betas_dir) / subject_id / 'train' / 'exp1_viewspec_training'
    
    mapping_dict = get_mapping_dict(task)
    
    moved_count = 0
     
    for beta_file in model_dir.glob('*.nii'):
        match = re.search(pattern, beta_file.name)
        
        if match:
            label = match.group(1)
            if fir:
                delay_no = int(match.group(2))
                run_no = int(match.group(3))
            else:
                run_no = int(match.group(2))
        else:
            raise Exception(f'Invalid filename {beta_file.name}!')
        
        new_label = mapping_dict[label]
        
        if fir:
            new_filename = f'beta_{new_label}_delay{delay_no:02d}_run{run_no:02d}.nii'
        else:
            new_filename = f'beta_{new_label}_run{run_no:02d}.nii'
            
        new_path = beta_file.parent / new_filename    
        
        if dry_run:
            print(f"Would move: {beta_file} -> {new_path}")
        else:
            try:
                shutil.move(beta_file, new_path)
                print(f"Moved: {beta_file} -> {new_path}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {beta_file}: {e}")
                
    print(f"\nSummary for {subject_id}:")
    print(f"  Beta files moved: {moved_count}")
    
    return True
        
def get_subject_list(betas_dir, task):
    """Get list of all subjects in the base directory."""
    
    betas_dir = Path(betas_dir)
    
    if not betas_dir.exists():
        print(f"Betas directory does not exist: {betas_dir}")
        return []
    
    if task == 'test':
        model_name = 'exp1_full_model'
    elif task == 'train':
        model_name = 'exp1_viewspec_training'
    
    subjects = []
    for item in betas_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            model_dir = item / task / model_name
            if model_dir.exists():
                subjects.append(item.name)
    
    return sorted(subjects)   

if __name__=="__main__":
    
    betas_dir = '/project/3018040.05/dyncontext_bids/experiment_1/derivatives/spm-preproc/derivatives/spm-stats/betas'
    
    subjlist = get_subject_list(betas_dir, 'test')
    
    for s in subjlist:
        print(f"\n{'-'*30}")
        print(f'Processing subject {s}')
        print(f"{'-'*30}\n")
        rename_betas(subject_id=s, betas_dir=betas_dir, task='train',
                     fir=False, dry_run=False)