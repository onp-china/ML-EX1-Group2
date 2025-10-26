#!/usr/bin/env python3
"""
Data Loader for Stacking Ensemble
=================================

Handles data loading and preprocessing for the stacking ensemble.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
try:
    from .augmentation import get_augmentation_transforms
except ImportError:
    from augmentation import get_augmentation_transforms

class PairNPZDataset(Dataset):
    """
    Dataset for loading paired MNIST data from NPZ files.
    """
    
    def __init__(self, npz_path: str, is_train: bool = True, use_augmentation: bool = False, 
                 labels_path: str = None, augmentation_type: str = 'none', **aug_kwargs):
        """
        Initialize dataset.
        
        Args:
            npz_path: Path to NPZ file
            is_train: Whether this is training data
            use_augmentation: Whether to use data augmentation
            labels_path: Path to separate labels file (for test_public)
            augmentation_type: Type of augmentation ('none', 'randaugment', 'autoaugment')
            **aug_kwargs: Additional arguments for augmentation
        """
        self.npz_path = npz_path
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.labels_path = labels_path
        self.augmentation_type = augmentation_type
        self.aug_kwargs = aug_kwargs
        
        # Load data
        data = np.load(npz_path)
        self.x = data['x']  # Shape: (N, 28, 56)
        
        # Load labels
        if 'y' in data:
            # Labels are in the NPZ file
            self.y = data['y']  # Shape: (N,)
        elif labels_path is not None:
            # Labels are in a separate file (e.g., test_public)
            import pandas as pd
            labels_df = pd.read_csv(labels_path)
            # Sort by id to match the order in NPZ file
            labels_df = labels_df.sort_values('id').reset_index(drop=True)
            self.y = labels_df['label'].values
        else:
            raise ValueError("No labels found in NPZ file and no labels_path provided")
        
        print(f"Dataset loaded: {len(self.x)} samples")
        print(f"Class distribution: {np.bincount(self.y)}")
        
        # Normalize pixel values to [0, 1]
        self.x = self.x.astype(np.float32) / 255.0
        
        # Reshape to add channel dimension: (N, 1, 28, 56)
        self.x = self.x.reshape(-1, 1, 28, 56)
        
        # 初始化数据增强
        if self.use_augmentation and self.is_train:
            self.augmentation = get_augmentation_transforms(
                self.augmentation_type, **self.aug_kwargs
            )
        else:
            self.augmentation = None
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (left_image, right_image, label)
        """
        # Split the 28x56 image into two 28x28 images
        full_image = self.x[idx]  # Shape: (1, 28, 56)
        left_image = full_image[:, :, :28]   # Shape: (1, 28, 28)
        right_image = full_image[:, :, 28:]  # Shape: (1, 28, 28)
        
        # Convert to tensors
        left_tensor = torch.from_numpy(left_image)
        right_tensor = torch.from_numpy(right_image)
        
        # Apply data augmentation if enabled
        if self.augmentation is not None:
            left_tensor, right_tensor = self.augmentation(left_tensor, right_tensor)
        
        label = self.y[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return left_tensor, right_tensor, label_tensor

class DataManager:
    """Manages data loading for the stacking ensemble."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data manager.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        
        # Data file paths
        self.train_path = os.path.join(data_dir, "train.npz")
        self.val_path = os.path.join(data_dir, "val.npz")
        self.test_public_path = os.path.join(data_dir, "test_public.npz")
        self.test_public_labels_path = os.path.join(data_dir, "test_public_labels.csv")
    
    def load_train_data(self, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
        """Load training data."""
        dataset = PairNPZDataset(self.train_path, is_train=True, use_augmentation=False)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    def load_val_data(self, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
        """Load validation data."""
        dataset = PairNPZDataset(self.val_path, is_train=False, use_augmentation=False)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    def load_test_public_data(self, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
        """Load public test data."""
        dataset = PairNPZDataset(
            self.test_public_path, 
            is_train=False, 
            use_augmentation=False,
            labels_path=self.test_public_labels_path
        )
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    def load_test_public_labels(self) -> np.ndarray:
        """Load public test labels."""
        import pandas as pd
        df = pd.read_csv(self.test_public_labels_path)
        return df['label'].values
    
    def get_data_info(self) -> dict:
        """Get information about the datasets."""
        info = {}
        
        # Training data info
        train_data = np.load(self.train_path)
        info['train_samples'] = len(train_data['x'])
        info['train_classes'] = np.bincount(train_data['y'])
        
        # Validation data info
        val_data = np.load(self.val_path)
        info['val_samples'] = len(val_data['x'])
        info['val_classes'] = np.bincount(val_data['y'])
        
        # Public test data info
        test_data = np.load(self.test_public_path)
        info['test_public_samples'] = len(test_data['x'])
        
        return info

def create_data_loaders(data_dir: str = "data", batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test.
    
    Args:
        data_dir: Directory containing data files
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_manager = DataManager(data_dir)
    
    train_loader = data_manager.load_train_data(batch_size)
    val_loader = data_manager.load_val_data(batch_size)
    test_loader = data_manager.load_test_public_data(batch_size)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test data loading
    data_manager = DataManager()
    info = data_manager.get_data_info()
    print("Data info:", info)
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a single batch
    for batch in val_loader:
        xa, xb, y = batch
        print(f"Batch shapes: xa={xa.shape}, xb={xb.shape}, y={y.shape}")
        break
