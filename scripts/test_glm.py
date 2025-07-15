import pandas as pd
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
        derivatives_dir / subject_id / "realign_unwarp" / f"rp_{subject_id}_task-train_run-{i}_bold.txt"
        for i in range(1, 4)
    ]
    
    # Output directory
    output_dir = Path("/project/3018040.05/dyncontext/glm_results")
    
    print("GLM PIPELINE DEMONSTRATION")
    
    # STEP 1: Create the model
    print("================================")
    print("STEP 1: MODEL CREATION")
    print("================================")
    
    model = create_individual_miniblock_training_model(viewspecific=True)
    
    print(f"Model name: {model.name}")
    print(f"Training filter: {type(model.training_filter).__name__}")
    print(f"Test filter: {type(model.test_filter).__name__ if model.test_filter else 'None'}")
    print(f"Config: {model.config}")
    
    print("================================")
    print("STEP 2: TRAINING EVENTS INSPECTION")
    print("================================")
    
    # Load and examine first training run events
    events_sample = pd.read_csv(events_files[0], sep='\t')
    print(f"Loaded events from: {events_files[0]}")
    print(f"Events shape: {events_sample.shape}")
    print(f"Columns: {list(events_sample.columns)}")
    print(f"\nFirst 10 events:")
    print(events_sample.head(10))
    
    print(f"\nTrial types: {events_sample['trial_type'].value_counts()}")
    
    # Show what the model would extract from this run
    try:
        session_info = model.specify_model(events_files[0])
        print(f"\nModel specification for this run:")
        print(f"Conditions: {session_info.conditions}")
        print(f"Number of events per condition:")
        for i, (cond, onsets) in enumerate(zip(session_info.conditions, session_info.onsets)):
            print(f"  {cond}: {len(onsets)} events")
            if len(onsets) > 0:
                print(f"    First onset: {onsets[0]:.2f}s")
    except Exception as e:
        print(f"Error in model specification: {e}")
        import traceback
        traceback.print_exc()
    
    print("================================")    
    print("STEP 3: GLM RUNNER SETUP")
    print("================================")
    
    glm_runner = SPMGLMRunner(
        tr=1.0,
        high_pass=128.0,
        working_dir=output_dir / 'glm_wf'
    )
    
    print(f"GLM parameters:")
    print(f"  TR: {glm_runner.tr}s")
    print(f"  High-pass: {glm_runner.high_pass}")
    
    print("================================")
    print("STEP 4: TRAINING GLM EXECUTION")
    print("================================")
    
    # Regular GLM returns: beta_maps, contrast_maps, glm_model
    beta_maps, contrast_maps, glm_model = glm_runner.run_subject_glm(
        subject_id=subject_id,
        model=model,
        func_files=func_files,
        events_files=events_files,
        confounds_files=confounds_files,
        output_dir=output_dir / "training"
    )
    
    print("\nTraining GLM completed successfully!")
    print(f"Beta maps for {len(beta_maps)} conditions:")
    for condition, maps in beta_maps.items():
        print(f"  {condition}: {len(maps)} beta maps (one per run)")
        if len(maps) > 0:
            print(f"    Shape: {maps[0].shape}")
    

if __name__=="__main__":
    
    main()