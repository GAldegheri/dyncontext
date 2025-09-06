import shutil
import os
from glob import glob

if __name__=="__main__":
    
    dry_run = False
    
    base_directory = '/project/3018040.05/bids/derivatives/spm-preproc/'
    
    subj_list = [f'sub-{s:03d}' for s in range(1, 36)]
    
    delete_dirs = ['coregister', 'fieldmap']
    
    delete_nii = ['realign_unwarp']
    
    for sub in subj_list:
        all_subfolders = glob(os.path.join(base_directory, sub, '*'))
        for f in all_subfolders:
            if os.path.isdir(f):
                if os.path.basename(f) in delete_dirs:
                    if dry_run:
                        print(f"Would delete: {f}")
                    else:
                        shutil.rmtree(f)
                        print(f"Deleted {f}")
                elif os.path.basename(f) in delete_nii:
                    all_nii_files = glob(os.path.join(f, '*.nii'))
                    for i in all_nii_files:
                        if dry_run:
                            print(f"Would delete: {i}")
                        else:
                            os.remove(i)
                            print(f"Deleted {i}")
        