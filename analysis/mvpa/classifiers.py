import numpy as np
import pandas as pd
from typing import Literal
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import logging
import ipdb

logger = logging.getLogger(__name__)

class LinearClassifier:
    """
    Linear classifier with support for different evaluation metrics
    """
    def __init__(self, C=1.0, kernel='linear', random_state=None):
        
        """
        Parameters:
        -----------
        C : float
            Regularization parameter
        kernel : str
            Kernel type for SVM
        random_state : int
            Random state for reproducibility
        """
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.classifier = None
        self.scaler = None
        self.is_fitted = False
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Required for scikit-learn compatibility and clone() function.
        
        Parameters:
        -----------
        deep : bool
            If True, return parameters for this estimator and contained
            subobjects that are estimators.
            
        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'C': self.C,
            'kernel': self.kernel,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Required for scikit-learn compatibility and clone() function.
        
        Parameters:
        -----------
        **params : dict
            Estimator parameters to set.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        valid_params = self.get_params().keys()
        
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}. "
                               f"Valid parameters are: {list(valid_params)}")
            setattr(self, key, value)
            
        # Reset fitted state when parameters change
        self.is_fitted = False
        self.classifier = None
        self.scaler = None
        
        return self
        
    def fit(self, X, y):
        """
        Fit the classifier
        
        Parameters:
        -----------
        X : np.ndarray
            Training data (n_samples x n_features)
        y : np.ndarray
            Training labels
        """
        self.classifier = SVC(
            kernel=self.kernel, 
            C=self.C, 
            random_state=self.random_state
        )
        self.scaler = StandardScaler()
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit classifier
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict(self, X):
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted yet")
        
        test_scaler = StandardScaler() 
        X_scaled = test_scaler.fit_transform(X)
        return self.classifier.predict(X_scaled)
    
    def decision_function(self, X):
        """Get decision function values (distance from boundary)"""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted yet")
            
        test_scaler = StandardScaler() 
        X_scaled = test_scaler.fit_transform(X)
        return self.classifier.decision_function(X_scaled)
    
    def score(self, X, y):
        """
        Score the classifier using different metrics
        
        Parameters:
        -----------
        X : np.ndarray
            Test data
        y : np.ndarray
            True labels
            
        Returns:
        --------
        pd.DataFrame : Individual results with columns:
        - sample_idx, true_label, predicted_label, correct, classifier_info
        """
        
        predictions = self.predict(X)
        
        classifier_info = self._compute_classifier_information(X, y)
        
        results_df = pd.DataFrame({
            'sample_idx': range(len(X)),
            'true_label': y,
            'predicted_label': predictions,
            'correct': predictions == y,
            'classifier_info': classifier_info
        })
    
        return results_df
    
    def _compute_classifier_information(self, X, y):
        # Get raw decision function values
        distances = self.decision_function(X)
        
        # Normalize by weight norm
        w_norm = np.linalg.norm(self.classifier.coef_)
        scaled_distances = distances / w_norm
        
        # Z-score the distances
        zscore_distances = (scaled_distances - np.mean(scaled_distances)) / np.std(scaled_distances)
        
        # Identify which label corresponds to negative decision values
        # (this is the SVM's "negative" class)
        neg_label = self.classifier.classes_[0]
        # Flip sign for the negative label so correct predictions are always positive
        classifier_infos = np.where(y == neg_label, -zscore_distances, zscore_distances)
        
        return classifier_infos
    

class CrossValidationClassifier:
    """
    Classifier with built-in cross-validation support
    """
    
    def __init__(self, base_classifier=None, cv_method='leave_one_out', n_splits=5):
        """
        Parameters:
        -----------
        base_classifier : LinearClassifier, optional
            Base classifier to use. If None, creates default LinearClassifier
        cv_method : str
            Cross-validation method ('leave_one_out', 'stratified_kfold')
        n_splits : int
            Number of splits for k-fold CV
        """
        self.base_classifier = base_classifier or LinearClassifier()
        self.cv_method = cv_method
        self.n_splits = n_splits
        
    def cross_validate(self, X, y):
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Data (n_samples x n_features)
        y : np.ndarray
            Labels
            
        Returns:
        --------
        dict : Results including mean, std, and individual fold scores
        """
        # Setup cross-validation
        if self.cv_method == 'leave_one_out':
            cv = LeaveOneOut()
        elif self.cv_method == 'stratified_kfold':
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")
        
        all_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone and fit classifier
            clf = clone(self.base_classifier)
            clf.fit(X_train, y_train)
            
            fold_results = clf.score(X_test, y_test)
            fold_results['fold'] = fold_idx
            fold_results['original_sample_idx'] = test_idx # Map back to original data
            
            all_results.append(fold_results)
            
        return pd.concat(all_results, ignore_index=True)
        

class TrainTestClassifier:
    """
    Classifier for train/test (cross-decoding) scenarios
    """
    
    def __init__(self, base_classifier=None):
        """
        Parameters:
        -----------
        base_classifier : LinearClassifier, optional
            Base classifier to use
        """
        self.base_classifier = base_classifier or LinearClassifier()
        
    
    def train_test(self, X_train, y_train, X_test, y_test):
        """
        Train on one dataset and test on another
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training data
        y_train : np.ndarray
            Training labels
        X_test : np.ndarray
            Test data
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        dict : Results including score and other metrics
        """
        # Clone and fit classifier
        clf = clone(self.base_classifier)
        clf.fit(X_train, y_train) 
        
        # Score on test set
        results_df = clf.score(X_test, y_test)
        
        return results_df
        
class ViewSpecificClassifier:
    """
    Classifier for view-specific analyses (background-matched decoding)
    """
    
    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier or LinearClassifier()
        
    def background_matched_decode(self, train_data_dict, test_data_dict):
        """
        Perform background-matched decoding
        
        Parameters:
        -----------
        train_data_dict : dict
            Dictionary with keys as background types, values as (X, y) tuples
        test_data_dict : dict
            Dictionary with keys as background types, values as (X, y) tuples
            
        Returns:
        --------
        dict : Results averaged across backgrounds
        """
        all_results = []
        
        for background in train_data_dict.keys():
            if background not in test_data_dict:
                logger.warning(f"Background {background} not found in test data")
                continue
            
            X_train, y_train = train_data_dict[background]
            X_test, y_test = test_data_dict[background]
            
            # Train and test on this background
            classifier = TrainTestClassifier(self.base_classifier)
            result_df = classifier.train_test(X_train, y_train, X_test, y_test)
            result_df['background'] = background
            
            all_results.append(result_df)
        
        if not all_results:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['sample_idx', 'true_label', 'predicted_label', 
                                   'correct', 'classifier_info', 'background'])
            
        return pd.concat(all_results, ignore_index=True)