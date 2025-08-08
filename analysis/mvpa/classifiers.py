import numpy as np
import pandas as pd
from typing import Literal
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import logging

logger = logging.getLogger(__name__)

MetricType = Literal['accuracy', 'classifier_information']


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
            
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)
    
    def decision_function(self, X):
        """Get decision function values (distance from boundary)"""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        return self.classifier.decision_function(X_scaled)
    
    def score(self, X, y, metric: MetricType = 'accuracy'):
        """
        Score the classifier using different metrics
        
        Parameters:
        -----------
        X : np.ndarray
            Test data
        y : np.ndarray
            True labels
        metric : str
            Metric to use ('accuracy', 'classifier_information', 'distance')
            
        Returns:
        --------
        float : Score according to the specified metric
        """
        if metric == 'accuracy':
            return self._compute_accuracy(X, y)
        elif metric == 'classifier_information':
            return self._compute_classifier_information(X, y)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
    def _compute_accuracy(self, X, y):
        """Compute classification accuracy"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def _compute_classifier_information(self, X, y):
        """
        Compute classifier information based on distance from decision boundary.
        This is the measure used in the paper.
        
        The classifier information is computed as the mean distance from the
        decision boundary, properly signed according to the true class labels.
        """
        distances = self.decision_function(X)
        
        # Convert labels to -1/+1 format for SVM
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("Classifier information requires binary classification")
        
        # Map labels to -1/+1
        y_mapped = np.where(y == unique_labels[0], -1, 1)
        
        # Compute signed distances (positive when prediction matches true label)
        signed_distances = distances * y_mapped
        
        # Return mean signed distance (classifier information)
        return np.mean(signed_distances)
    

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
        
    def cross_validate(self, X, y, metric: MetricType = 'classifier_information'):
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Data (n_samples x n_features)
        y : np.ndarray
            Labels
        metric : str
            Metric to use for evaluation
            
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
        
        scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
            # Clone and fit classifier
            clf = clone(self.base_classifier)
            clf.fit(X_train, y_train)
            
            # Score on test set
            score = clf.score(X_test, y_test, metric=metric)
            scores.append(score)
        
        return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'individual_scores': scores,
                'metric': metric
        }
        

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
        
    
    def train_test(self, X_train, y_train, X_test, y_test, 
                   metric: MetricType = 'classifier_information'):
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
        metric : str
            Metric to use for evaluation
            
        Returns:
        --------
        dict : Results including score and other metrics
        """
        # Clone and fit classifier
        clf = clone(self.base_classifier)
        clf.fit(X_train, y_train) 
        
        # Score on test set
        score = clf.score(X_test, y_test, metric=metric)
        
        # Also compute accuracy for comparison
        accuracy = clf.score(X_test, y_test, metric='accuracy')
        
        return {
            'score': score,
            'accuracy': accuracy,
            'metric': metric,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
class ViewSpecificClassifier:
    """
    Classifier for view-specific analyses (background-matched decoding)
    """
    
    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier or LinearClassifier()
        
    def background_matched_decode(self, train_data_dict, test_data_dict,
                                  metric: MetricType = 'classifier_information'):
        """
        Perform background-matched decoding
        
        Parameters:
        -----------
        train_data_dict : dict
            Dictionary with keys as background types, values as (X, y) tuples
        test_data_dict : dict
            Dictionary with keys as background types, values as (X, y) tuples
        metric : str
            Metric to use
            
        Returns:
        --------
        dict : Results averaged across backgrounds
        """
        background_results = []
        
        for background in train_data_dict.keys():
            if background not in test_data_dict:
                logger.warning(f"Background {background} not found in test data")
                continue
            
            X_train, y_train = train_data_dict[background]
            X_test, y_test = test_data_dict[background]
            
            # Train and test on this background
            classifier = TrainTestClassifier(self.base_classifier)
            result = classifier.train_test(X_train, y_train, X_test, y_test, metric)
            result['background'] = background
            
            background_results.append(result)
        
        if not background_results:
            return {'mean_score': np.nan, 'backgrounds': []}
        
        # Average across backgrounds
        scores = [r['score'] for r in background_results]
        accuracies = [r['accuracy'] for r in background_results]
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_accuracy': np.mean(accuracies),
            'individual_scores': scores,
            'background_results': background_results,
            'metric': metric
        }
      
  
class SplitTrialsClassifier:
    """
    Classifier that handles trials split into multiple groups
    (e.g., the splits of congruent trials in Experiment 1)
    """
    
    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier or LinearClassifier()
        
    def decode_with_splits(self, X_splits, y_splits, X_other, y_other,
                           metric: MetricType = 'classifier_information'):
        """
        Decode split trials against other condition
        
        Parameters:
        -----------
        X_splits : list of np.ndarray
            Data for each split
        y_splits : list of np.ndarray
            Labels for each split
        X_other : np.ndarray
            Data for comparison condition
        y_other : np.ndarray
            Labels for comparison condition
        metric : str
            Metric to use
            
        Returns:
        --------
        dict : Results averaged across splits
        """
        split_results = []
        
        for i, (X_split, y_split) in enumerate(zip(X_splits, y_splits)):
            # Combine split data with other condition
            X_combined = np.vstack([X_split, X_other])
            y_combined = np.concatenate([y_split, y_other])
            
            classifier = TrainTestClassifier(self.base_classifier)
            result = classifier.cross_validate(X_combined, y_combined, metric)
            result['split'] = i + 1
            
            split_results.append(result)
            
        # Average across splits
        scores = [r['mean_score'] for r in split_results]