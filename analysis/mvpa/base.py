class MVPAAnalysis:
    """Base class for MVPA analyses"""
    def __init__(self, name, description, roi=None):
        self.name = name
        self.description = description
        self.roi = roi
        self.classifier = None
        self.data = None
        self.results = None
        
    def load_data(self, subject, task, model, data_format='betas'):
        """Load data for analysis"""
        raise NotImplementedError
    
    def set_classifier(self, classifier):
        """Set the classifier to use"""
        self.classifier = classifier
        
    def preprocess(self):
        """Preprocess data (z-scoring, etc.)"""
        raise NotImplementedError
    
    def run(self):
        """Run the analysis"""
        raise NotImplementedError
    
    def evaluate(self):
        """Evaluate performance"""
        raise NotImplementedError