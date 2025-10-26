#!/usr/bin/env python3
"""
Stacking Ensemble Implementation
===============================

Implements the stacking meta-learner using LightGBM.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb

class StackingEnsemble:
    """Implements stacking ensemble with LightGBM meta-learner."""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize stacking ensemble.
        
        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random state for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = None
        self.cv_scores = []
        self.feature_importance = None
        
    def train_stacking_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[lgb.LGBMClassifier, float, List[float]]:
        """
        Train stacking meta-learner with cross-validation.
        
        Args:
            X: Feature matrix of shape (N, num_models)
            y: Target labels of shape (N,)
            
        Returns:
            Tuple of (final_model, overall_acc, fold_scores)
        """
        print(f"Training Stacking meta-learner with {self.n_folds}-fold cross-validation...")
        
        # Create cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Store predictions for each fold
        oof_predictions = np.zeros(len(X))
        fold_scores = []
        feature_importances = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Training fold {fold+1}...")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create LightGBM model
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=self.random_state,
                n_estimators=100
            )
            
            # Train model
            lgb_model.fit(X_train, y_train)
            
            # Predict validation set
            val_pred = lgb_model.predict_proba(X_val)[:, 1]
            oof_predictions[val_idx] = val_pred
            
            # Calculate accuracy
            val_pred_binary = (val_pred >= 0.5).astype(int)
            fold_acc = accuracy_score(y_val, val_pred_binary)
            fold_scores.append(fold_acc)
            feature_importances.append(lgb_model.feature_importances_)
            
            print(f"    Fold {fold+1} accuracy: {fold_acc:.4f}")
        
        # Calculate overall performance
        oof_pred_binary = (oof_predictions >= 0.5).astype(int)
        overall_acc = accuracy_score(y, oof_pred_binary)
        
        print(f"Stacking overall accuracy: {overall_acc:.4f}")
        print(f"Fold accuracies: {[f'{acc:.4f}' for acc in fold_scores]}")
        
        # Train final model
        print("Training final Stacking model...")
        final_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=self.random_state,
            n_estimators=100
        )
        
        final_model.fit(X, y)
        
        # Store results
        self.meta_learner = final_model
        self.cv_scores = fold_scores
        self.feature_importance = np.mean(feature_importances, axis=0)
        
        return final_model, overall_acc, fold_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained meta-learner.
        
        Args:
            X: Feature matrix of shape (N, num_models)
            
        Returns:
            Prediction probabilities of shape (N,)
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained yet. Call train_stacking_model first.")
        
        return self.meta_learner.predict_proba(X)[:, 1]
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Feature matrix of shape (N, num_models)
            threshold: Classification threshold
            
        Returns:
            Binary predictions of shape (N,)
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate the stacking ensemble.
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        pred_probs = self.predict(X)
        pred_binary = (pred_probs >= threshold).astype(int)
        
        acc = accuracy_score(y, pred_binary)
        f1 = f1_score(y, pred_binary)
        
        return {
            'accuracy': acc,
            'f1_score': f1,
            'threshold': threshold
        }
    
    def get_feature_importance(self, model_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance for each model.
        
        Args:
            model_names: List of model names
            
        Returns:
            Dictionary mapping model names to importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train the model first.")
        
        return dict(zip(model_names, self.feature_importance))
    
    def save_model(self, filepath: str):
        """Save the trained meta-learner."""
        if self.meta_learner is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save LightGBM model
        self.meta_learner.booster_.save_model(filepath)
        
        # Save metadata
        metadata = {
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance.tolist()
        }
        
        metadata_path = filepath.replace('.txt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load a trained meta-learner."""
        # Load LightGBM model
        self.meta_learner = lgb.LGBMClassifier()
        self.meta_learner.booster_ = lgb.Booster(model_file=filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.txt', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.n_folds = metadata['n_folds']
            self.random_state = metadata['random_state']
            self.cv_scores = metadata['cv_scores']
            self.feature_importance = np.array(metadata['feature_importance'])

def compare_ensemble_methods(X: np.ndarray, y: np.ndarray, model_names: List[str]) -> Dict[str, float]:
    """
    Compare different ensemble methods.
    
    Args:
        X: Feature matrix of shape (N, num_models)
        y: True labels
        model_names: List of model names
        
    Returns:
        Dictionary with results for each method
    """
    results = {}
    
    # 1. Simple average
    simple_avg = X.mean(axis=1)
    simple_pred = (simple_avg >= 0.5).astype(int)
    results['simple_average'] = accuracy_score(y, simple_pred)
    
    # 2. Performance weighted (assuming equal weights for now)
    weights = np.ones(X.shape[1]) / X.shape[1]
    weighted_avg = np.sum(X * weights, axis=1)
    weighted_pred = (weighted_avg >= 0.5).astype(int)
    results['performance_weighted'] = accuracy_score(y, weighted_pred)
    
    # 3. Stacking
    stacking = StackingEnsemble()
    stacking.train_stacking_model(X, y)
    stacking_pred = stacking.predict_binary(X)
    results['stacking'] = accuracy_score(y, stacking_pred)
    
    return results

if __name__ == "__main__":
    # Test stacking ensemble
    print("Testing Stacking Ensemble...")
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.rand(1000, 5)  # 1000 samples, 5 models
    y = np.random.randint(0, 2, 1000)  # Binary labels
    
    # Test stacking
    stacking = StackingEnsemble(n_folds=3)
    model, acc, scores = stacking.train_stacking_model(X, y)
    
    print(f"Test accuracy: {acc:.4f}")
    print(f"Fold scores: {scores}")
    
    # Test prediction
    pred = stacking.predict_binary(X)
    print(f"Prediction shape: {pred.shape}")
