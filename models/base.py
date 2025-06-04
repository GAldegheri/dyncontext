class BaseGLMModel:
    """Base class for GLM models"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        
    def specify_model(self, events_file, behavior=None):
        """Generate model specification"""
        raise NotImplementedError
    
    def generate_contrasts(self):
        """Generate contrast definitions"""
        raise NotImplementedError