from analysis.mvpa.base import MVPAAnalysis

class ProximalShapeDecoder(MVPAAnalysis):
    """Decoder for proximal shape (wide vs. narrow)"""
    
    def __init__(self, roi):
        super().__init__(
            name="proximal_shape_decoder",
            description="Decodes wide vs. narrow proximal shape projections",
            roi=roi
        )