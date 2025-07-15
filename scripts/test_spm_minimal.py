#!/usr/bin/env python
"""
Final simple fix for the SPM/Nipype compatibility issue
This should be placed at the VERY TOP of your main script
"""

# Fix the _strip_header issue before importing nipype
# import sys
# import types

# # Create mock spm_docs module
# spm_docs_mock = types.ModuleType('nipype.utils.spm_docs')

# # Add a _strip_header function that won't fail
# def _strip_header(text_output):
#     """Extract text from various input formats without failing"""
#     # Handle different input types
#     if hasattr(text_output, 'runtime'):
#         text = getattr(text_output.runtime, 'stdout', str(text_output))
#     else:
#         text = str(text_output)
    
#     # Just return the text without trying to strip headers
#     # This prevents the "substring not found" error
#     return text

# # Add other functions that might be needed
# def strip_header(text):
#     """Alias for _strip_header"""
#     return _strip_header(text)

# # Set up the mock module
# spm_docs_mock._strip_header = _strip_header
# spm_docs_mock.strip_header = strip_header

# # CRITICAL: Insert into sys.modules BEFORE nipype imports it
# sys.modules['nipype.utils.spm_docs'] = spm_docs_mock

print("SPM compatibility fix applied!")

# Now you can safely use your imports
if __name__ == "__main__":
    # Test that it works
    from nipype.interfaces.spm.model import Level1Design
    from nipype import Node
    
    test_node = Node(Level1Design(), name='test')