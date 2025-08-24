import shutil
import os
from glob import glob

if __name__=="__main__":
    
    dry_run = False
    
    base_directory = '/project/3018040.05/dyncontext_bids/experiment_1/derivatives/spm-preproc/derivatives/spm-stats/betas'
    
    subj_list = [f'sub-{s:03d}' for s in range(1, 35)]
    
    for sub in subj_list:
        this_sub_dir = os.path.join(base_directory, sub, 'test', 'exp2_wide_narrow')
        if os.path.exists(this_sub_dir):
            if dry_run:
                print(f'Would delete: {this_sub_dir}')
            else:
                shutil.rmtree(this_sub_dir)
                print(f'Deleted {this_sub_dir}')
        
        