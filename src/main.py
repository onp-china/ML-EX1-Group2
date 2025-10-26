#!/usr/bin/env python3
"""
Main Stacking Ensemble Program
==============================

Complete pipeline for training and testing the stacking ensemble.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_loader import ModelLoader, set_seed
from data_loader import DataManager
from stacking_ensemble import StackingEnsemble, compare_ensemble_methods

class StackingPipeline:
    """Complete pipeline for stacking ensemble."""
    
    def __init__(self, 
                 models_dir: str = "models",
                 data_dir: str = "data", 
                 results_dir: str = "results",
                 device: str = "auto"):
        """
        Initialize the stacking pipeline.
        
        Args:
            models_dir: Directory containing model files
            data_dir: Directory containing data files
            results_dir: Directory for output results
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = device
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.model_loader = ModelLoader(models_dir, device)
        self.data_manager = DataManager(data_dir)
        self.stacking = StackingEnsemble()
        
        # Results storage
        self.results = {}
    
    def run_complete_pipeline(self, batch_size: int = 64) -> Dict:
        """
        Run the complete stacking pipeline.
        
        Args:
            batch_size: Batch size for data loading
            
        Returns:
            Dictionary with all results
        """
        print("="*60)
        print("STACKING ENSEMBLE PIPELINE")
        print("="*60)
        print(f"Models directory: {self.models_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Results directory: {self.results_dir}")
        print(f"Device: {self.model_loader.device}")
        print("="*60)
        
        # Step 1: Load all models
        print("\n=== Step 1: Loading All Models ===")
        models, model_names, accuracies = self.model_loader.load_all_models()
        
        if len(models) == 0:
            raise ValueError("No models loaded successfully!")
        
        self.results['model_info'] = {
            'num_models': len(models),
            'model_names': model_names,
            'accuracies': accuracies,
            'best_single_model': max(accuracies)
        }
        
        # Step 2: Load validation data
        print("\n=== Step 2: Loading Validation Data ===")
        val_loader = self.data_manager.load_val_data(batch_size)
        print(f"Validation set size: {len(val_loader.dataset)}")
        
        # Step 3: Get model predictions on validation set
        print("\n=== Step 3: Getting Model Predictions ===")
        X_val = self.model_loader.get_model_predictions(models, val_loader)
        print(f"Feature matrix shape: {X_val.shape}")
        
        # Get validation labels
        y_val = []
        for batch in val_loader:
            _, _, labels = batch
            y_val.append(labels.numpy())
        y_val = np.concatenate(y_val).astype(int)
        print(f"Labels shape: {y_val.shape}")
        print(f"Class distribution: {np.bincount(y_val)}")
        
        # Step 4: Train stacking model
        print("\n=== Step 4: Training Stacking Model ===")
        meta_learner, stacking_acc, fold_scores = self.stacking.train_stacking_model(X_val, y_val)
        
        self.results['validation_results'] = {
            'stacking_accuracy': stacking_acc,
            'fold_scores': fold_scores,
            'mean_fold_score': np.mean(fold_scores),
            'std_fold_score': np.std(fold_scores)
        }
        
        # Step 5: Compare ensemble methods
        print("\n=== Step 5: Comparing Ensemble Methods ===")
        ensemble_comparison = compare_ensemble_methods(X_val, y_val, model_names)
        
        for method, acc in ensemble_comparison.items():
            print(f"{method}: {acc:.4f}")
        
        self.results['ensemble_comparison'] = ensemble_comparison
        
        # Step 6: Test on public test set
        print("\n=== Step 6: Testing on Public Test Set ===")
        test_loader = self.data_manager.load_test_public_data(batch_size)
        print(f"Public test set size: {len(test_loader.dataset)}")
        
        # Get predictions on test set
        X_test = self.model_loader.get_model_predictions(models, test_loader)
        print(f"Test feature matrix shape: {X_test.shape}")
        
        # Apply stacking model
        test_pred_probs = self.stacking.predict(X_test)
        test_pred_binary = self.stacking.predict_binary(X_test)
        
        # Load true labels
        y_test_true = self.data_manager.load_test_public_labels()
        print(f"True labels shape: {y_test_true.shape}")
        print(f"True labels distribution: {np.bincount(y_test_true)}")
        
        # Calculate test metrics
        from sklearn.metrics import accuracy_score, f1_score
        test_acc = accuracy_score(y_test_true, test_pred_binary)
        test_f1 = f1_score(y_test_true, test_pred_binary)
        
        print(f"Public Test Accuracy: {test_acc:.4f}")
        print(f"Public Test F1 Score: {test_f1:.4f}")
        
        self.results['public_test_results'] = {
            'accuracy': test_acc,
            'f1_score': test_f1,
            'num_samples': len(y_test_true)
        }
        
        # Step 7: Individual model performance on test set
        print("\n=== Step 7: Individual Model Performance ===")
        individual_accs = []
        for i, model_name in enumerate(model_names):
            model_pred = (X_test[:, i] >= 0.5).astype(int)
            model_acc = accuracy_score(y_test_true, model_pred)
            individual_accs.append(model_acc)
            print(f"{model_name}: {model_acc:.4f}")
        
        self.results['individual_model_test_accs'] = individual_accs
        
        # Step 8: Save results and predictions
        print("\n=== Step 8: Saving Results ===")
        self.save_results()
        self.save_predictions(test_pred_binary, test_pred_probs)
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Best Individual Model: {max(individual_accs):.4f}")
        print(f"Stacking Ensemble: {test_acc:.4f}")
        print(f"Improvement: +{test_acc - max(individual_accs):.4f}")
        print(f"Validation Accuracy: {stacking_acc:.4f}")
        print(f"Public Test Accuracy: {test_acc:.4f}")
        print("="*60)
        
        return self.results
    
    def save_results(self):
        """Save all results to JSON file."""
        results_file = os.path.join(self.results_dir, "stacking_results.json")
        
        # Add timestamp
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['pipeline_info'] = {
            'models_dir': self.models_dir,
            'data_dir': self.data_dir,
            'device': str(self.model_loader.device)
        }
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    def save_predictions(self, predictions: np.ndarray, probabilities: np.ndarray):
        """Save predictions to CSV file."""
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'id': [f'PUB_{i:07d}' for i in range(len(predictions))],
            'prediction': predictions,
            'probability': probabilities
        })
        
        # Save predictions
        pred_file = os.path.join(self.results_dir, "public_test_predictions.csv")
        pred_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")
        
        # Save probabilities only (for submission)
        submission_df = pred_df[['id', 'prediction']].copy()
        submission_file = os.path.join(self.results_dir, "pred_public.csv")
        submission_df.to_csv(submission_file, index=False)
        print(f"Submission file saved to: {submission_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Stacking Ensemble Pipeline')
    parser.add_argument('--models_dir', default='models', help='Directory containing model files')
    parser.add_argument('--data_dir', default='data', help='Directory containing data files')
    parser.add_argument('--results_dir', default='results', help='Directory for output results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create and run pipeline
    pipeline = StackingPipeline(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=args.device
    )
    
    try:
        results = pipeline.run_complete_pipeline(args.batch_size)
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
