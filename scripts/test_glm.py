from pathlib import Path

from analysis.glm.spm_glm_runner import SPMGLMRunner
from models.models import create_individual_miniblock_training_model

def main():
    """Main pipeline demonstration"""
    
    subject_id = 'sub-001'
    
    data_dir = Path("/project/3018040.05/bids")
    derivatives_dir = data_dir / "derivatives" / "spm-preproc"
    
    # Functional files (preprocessed)
    func_files = [
        derivatives_dir / subject_id / "smooth" / f"s3wu{subject_id}_task-train_run-{i}_bold.nii"
        for i in range(1, 4)
    ]
    
    events_files = [
        data_dir / subject_id / "func" / f"{subject_id}_task-train_run-{i}_events.tsv"
        for i in range(1, 4)
    ]
    
    confounds_files = [
        derivatives_dir / subject_id / "realign_unwarp" / f"rp_{subject_id}_task-inverted_run-1_sbref.txt"
    ]
    
    # Output directory
    output_dir = Path("./glm_results")
    
    print("GLM PIPELINE DEMONSTRATION")
    
    # STEP 1: Create the model
    print("STEP 1: MODEL CREATION")
    
    model = create_individual_miniblock_training_model(viewspecific=True)
    
    print(f"Model name: {model.name}")
    print(f"Training filter: {type(model.training_filter).__name__}")
    print(f"Test filter: {type(model.test_filter).__name__ if model.test_filter else 'None'}")
    print(f"Config: {model.config}")

if __name__=="__main__":
    
    main()