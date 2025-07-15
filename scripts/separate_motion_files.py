import numpy as np
import os
from glob import glob

START_DICT = {
        'test': ([12, 417, 1475, 1880, 2619, 3024, 4082], 404),
        'train': ([822, 2285, 3429], 333),
        'funcloc': ([1156, 3763], 318)
    }

def separate_motpar(subj, base_dir):
    
    thissubj_dir = os.path.join(base_dir, subj, 'realign_unwarp')
    motion_files = glob(os.path.join(thissubj_dir, '*.txt'))
    motionarray = np.empty((0, 6))
    for f in motion_files:
        motionarray = np.append(motionarray, np.loadtxt(f), axis=0)
    
    for (t, n_runs) in [('test', 7), ('train', 3), ('funcloc', 2)]:
        run_length = START_DICT[t][1]
        for r in range(n_runs):
            run_start = START_DICT[t][0][r]
            if motionarray.shape == (4476, 6): # subj. 14 missed the 10 inverted bold scans
                run_start -= 10
                
            thismotion = motionarray[run_start:run_start+run_length, :]
                
            thisrun_outfile = f'rp_{subj}_task-{t}_run-{r+1}_bold.txt'
            np.savetxt(os.path.join(thissubj_dir, thisrun_outfile), thismotion)
    
    return

if __name__=="__main__":
    
    base_dir = '/project/3018040.05/bids/derivatives/spm-preproc/'
    
    subjlist = [f'sub-{i:03d}' for i in range(1, 36)]
    for s in subjlist:
        separate_motpar(s, base_dir)