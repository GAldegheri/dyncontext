import os
import shutil

def main(task, model, contrasts=False):
    betadir = '../dyncontext_bids/experiment_1/derivatives/spm-preproc/derivatives/spm-stats/betas/'
    contrdir = '../dyncontext_bids/experiment_1/derivatives/spm-preproc/derivatives/spm-stats/contrasts/'
    subjlist = [f'sub-{i:03d}' for i in range(1, 36)]
    for s in subjlist:
        thisbetadir = os.path.join(betadir, s, task, model)
        if os.path.isdir(thisbetadir):
            shutil.rmtree(thisbetadir)
            print(thisbetadir, 'removed.')
        else:
            print(thisbetadir, 'not found.')
        if contrasts:
            thiscontrdir = os.path.join(contrdir, s, task, model)
            if os.path.isdir(thiscontrdir):
                shutil.rmtree(thiscontrdir)
                print(thiscontrdir, 'removed.')
            else:
                print(thiscontrdir, 'not found.')
                
if __name__=="__main__":
    main('test', 'full_model', contrasts=False)