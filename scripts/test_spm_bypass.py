#!/usr/bin/env python
"""
Test script to verify the version bypass fix works
"""

import os
import sys

# Add the bypass at the very beginning
print("Testing SPM version bypass fix...")

# Method 1: Try environment variable approach first
# os.environ['NIPYPE_NO_ET'] = '1'
# #os.environ['FORCE_SPMMCR'] = '1'
# import nipype
# nipype.config.set('execution', 'stop_on_first_crash', 'false')
# nipype.config.set('execution', 'check_version', 'false')

from nipype.interfaces import base

# Monkey patch to bypass version checking
original_check_version = base.BaseInterface._check_version_requirements

# def patched_check_version(self, trait_object, name, value):
#     """Bypass version checking for SPM interfaces"""
#     # If this is an SPM interface, skip version checking
#     if hasattr(self, '__module__') and 'spm' in self.__module__:
#         return None
#     # Otherwise use original version checking
#     return original_check_version(self, trait_object, name, value)

# base.BaseInterface._check_version_requirements = patched_check_version

# Method 2: Import the patched runner
try:
    # Import your fixed SPM runner
    #from analysis.glm.spm_glm_runner import SPMGLMRunner, SPMLevel1DesignNoVersionCheck
    from nipype import Node
    from nipype.interfaces.spm.model import Level1Design
    
    class SPMLevel1DesignNoVersionCheck(Level1Design):
        """Level1Design node that bypasses version checking"""
        
        @property
        def version(self):
            """Return a dummy version to bypass checking"""
            return "12.0"
        
        def _check_version_requirements(self, trait_object, name, value):
            """Skip version checking"""
            return None
    
    print("\n1. Testing custom Level1Design node creation...")
    try:
        # Try with custom node
        level1design = Node(SPMLevel1DesignNoVersionCheck(), name='test_level1design')
        level1design.inputs.timing_units = 'secs'
        level1design.inputs.interscan_interval = 1.0
        
        print("✓ Custom Level1Design node created successfully!")
        
    except Exception as e:
        print(f"✗ Failed to create custom Level1Design node: {e}")
        
    # print("\n2. Testing SPMGLMRunner initialization...")
    # try:
    #     # Initialize runner with version bypass
    #     runner = SPMGLMRunner(
    #         tr=1.0,
    #         matlab_cmd='/opt/matlab/R2022b/bin/matlab -nojvm -nodisplay',
    #         spm_path='/home/common/matlab/spm12',
    #         use_version_bypass=True
    #     )
        
    #     print("✓ SPMGLMRunner initialized successfully with version bypass!")
        
    # except Exception as e:
    #     print(f"✗ Failed to initialize SPMGLMRunner: {e}")
        
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    print("Make sure the path to your project is correct")
    
print("\nTest complete!")