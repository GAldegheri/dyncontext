# ───────────────────────────────────────────────────────────
# MONKEY-PATCH OUT SPM VERSION CHECK
# import nipype.interfaces.spm.base as _spmbase

# @classmethod
# def _stub_getinfo(cls, matlab_cmd=None, paths=None, use_mcr=None):
#     return {
#         "name":    "SPM12",
#         "path":    (paths[0] if paths else ""),
#         "release": "12"
#     }
# _spmbase.Info.getinfo = _stub_getinfo
# ───────────────────────────────────────────────────────────

import os
os.environ.pop("FORCE_SPMMCR", None)
os.environ.pop("SPMMCRCMD", None)
# import nipype
# nipype.config.set('execution', 'stop_on_first_crash', 'false')
# nipype.config.set('execution', 'check_version', 'false')
# from nipype.interfaces import spm

import nipype.utils.spm_docs as spm_docs

def test_spm_setup():
    """Test SPM setup step by step"""
    
    print("=" * 60)
    print("SPM/Nipype Diagnostic Test")
    print("=" * 60)
    
    # Step 1: Test MATLAB availability
    print("\n1. Testing MATLAB availability...")
    try:
        from nipype.interfaces.matlab import MatlabCommand
        matlab_cmd = '/opt/matlab/R2022b/bin/matlab -nojvm -nodisplay'
        MatlabCommand.set_default_matlab_cmd(matlab_cmd)
        print(f"✓ MATLAB command set: {matlab_cmd}")
    except Exception as e:
        print(f"✗ Failed to set MATLAB command: {e}")
        return
    
    # Step 2: Test SPM path
    print("\n2. Testing SPM path...")
    try:
        spm_path = '/home/common/matlab/spm12'
        if os.path.exists(spm_path):
            print(f"✓ SPM path exists: {spm_path}")
        else:
            print(f"✗ SPM path does not exist: {spm_path}")
            return
            
        MatlabCommand.set_default_paths(spm_path)
        print("✓ SPM paths set in MatlabCommand")
    except Exception as e:
        print(f"✗ Failed to set SPM paths: {e}")
        return
    
    # Step 3: Test SPM command initialization
    print("\n3. Testing SPM command initialization...")
    try:
        from nipype.interfaces import spm
        #spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=False)
        nipype_utils = os.path.dirname(spm_docs.__file__)
        spm.SPMCommand.set_mlab_paths(
            matlab_cmd=matlab_cmd,
            use_mcr=False,
            paths=[spm_path, nipype_utils]
        )
        print("✓ SPM command paths set")
    except Exception as e:
        print(f"✗ Failed to initialize SPM command: {e}")
        return
    
    # Step 4: Test SPM version check
    print("\n4. Testing SPM version check...")
    try:
        # Create a simple SPM command to test version checking
        test_cmd = spm.Info()
        version = test_cmd.version()
        print(f"✓ SPM version detected: {version}")
    except Exception as e:
        print(f"✗ Failed to get SPM version: {e}")
        print("  This is likely the source of your error")
    
    # Step 5: Test Level1Design creation
    print("\n5. Testing Level1Design node creation...")
    try:
        from nipype.interfaces.spm.model import Level1Design
        from nipype import Node
        
        # Try creating the node
        level1design = Node(Level1Design(), name='test_level1design')
        print("✓ Level1Design node created successfully")
        
        # Try setting some basic inputs
        level1design.inputs.timing_units = 'secs'
        level1design.inputs.interscan_interval = 1.0
        print("✓ Basic inputs set successfully")
        
    except Exception as e:
        print(f"✗ Failed to create Level1Design node: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Test running a simple MATLAB command
    print("\n6. Testing MATLAB execution...")
    try:
        from nipype.interfaces.matlab import MatlabCommand
        mlab = MatlabCommand()
        mlab.inputs.script = "disp('Hello from MATLAB'); ver;"
        
        print("  Running MATLAB test command...")
        result = mlab.run()
        
        if result.runtime.returncode == 0:
            print("✓ MATLAB execution successful")
            print("\nMATLAB output:")
            print("-" * 40)
            print(result.runtime.stdout[:500])  # First 500 chars
            print("-" * 40)
        else:
            print(f"✗ MATLAB execution failed with code: {result.runtime.returncode}")
            print(f"Error: {result.runtime.stderr}")
    except Exception as e:
        print(f"✗ Failed to run MATLAB command: {e}")
    
    print("\n" + "=" * 60)
    print("Diagnostic test complete")
    print("=" * 60)

if __name__ == "__main__":
    test_spm_setup()