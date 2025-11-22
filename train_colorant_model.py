#!/usr/bin/env python3
"""
Natural Food Colorant Formulation Model Training Pipeline
For: Kraft Heinz Jello Natural Color Project

This script trains a model to predict:
1. Optimal colorant combinations for target colors
2. Stability predictions (pH, temperature, time)
3. Formulation adjustments for real-world conditions

Author: Frank (QUNEU)
Purpose: Train QMAT (Quantum-Enhanced Materials AI) for colorant formulation
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class ColorantModelConfig:
    """Configuration for colorant formulation model - REGRESSION VERSION"""
    
    # GCP Configuration
    GCP_PROJECT_ID = "majestic-layout-461420-k4"
    GCP_BUCKET_NAME = "eurofins"
    
    # Data Paths
    DATA_DIR = Path("./colorant_data_expanded")
    CHECKPOINT_DIR = Path("./checkpoints_colorant_v2_regression")
    OUTPUT_DIR = Path("./outputs_colorant_v2_regression")
    
    # GCS paths
    GCS_DATA_PREFIX = "expanded_data/"
    
    # Model Architecture - REGRESSION OUTPUT
    INPUT_DIM = 256
    HIDDEN_DIMS = [128, 64, 32]
    OUTPUT_DIM = 8  # Predict 8 continuous properties instead of classification
    DROPOUT = 0.4
    
    # Output targets (what we're predicting)
    TARGET_FEATURES = [
        'pH_stability_3', 'pH_stability_5', 'pH_stability_7',
        'k_4C', 'light_stability', 'oxygen_sensitivity',
        'MolecularWeight', 'Complexity'
    ]
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = 10
    WEIGHT_DECAY = 1e-4
    
    # Data weighting
    DATA_SOURCE_WEIGHTS = {
        'pubchem_extended': 3.0,
        'literature_extended': 2.5,
        'synthetic_interpolated': 1.5,
        'synthetic_perturbed': 1.2,
        'synthetic_distribution': 1.0
    }
    
    # Early Stopping
    PATIENCE = 25
    MIN_DELTA = 1e-5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 10
# ==============================================================================
# DATA LOADER FROM GCS
# ==============================================================================

class ExpandedGCSDataLoader:
    """Load expanded data from GCS"""
    
    def __init__(self, project_id: str, bucket_name: str, local_dir: Path):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.local_dir = local_dir
        self.local_dir.mkdir(exist_ok=True, parents=True)
        
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
        
        logging.info(f"✅ Connected to GCS: gs://{bucket_name}/")
    
    def load_combined_data(self) -> pd.DataFrame:
        """Load the pre-combined dataset"""
        logging.info("📥 Loading combined dataset from GCS...")
        
        gcs_path = "expanded_data/combined_all_data.csv"
        local_path = self.local_dir / "combined_all_data.csv"
        
        blob = self.bucket.blob(gcs_path)
        
        if not blob.exists():
            raise FileNotFoundError(
                f"Combined dataset not found at gs://{self.bucket_name}/{gcs_path}\n"
                "Please run collect_more_data.py first!"
            )
        
        blob.download_to_filename(str(local_path))
        df = pd.read_csv(local_path)
        
        logging.info(f"✅ Loaded combined data: {len(df)} samples")
        
        return df

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

class ColorantDataProcessor:
    """Process and feature engineer colorant data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def process_pubchem_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from PubChem data"""
        logging.info("🔬 Processing PubChem chemical data...")
        
        features = pd.DataFrame()
        
        # Molecular properties
        if 'MolecularWeight' in df.columns:
            features['mol_weight'] = pd.to_numeric(df['MolecularWeight'], errors='coerce')
        
        if 'XLogP' in df.columns:
            features['logp'] = pd.to_numeric(df['XLogP'], errors='coerce')
        
        if 'TPSA' in df.columns:
            features['tpsa'] = pd.to_numeric(df['TPSA'], errors='coerce')
        
        if 'Complexity' in df.columns:
            features['complexity'] = pd.to_numeric(df['Complexity'], errors='coerce')
        
        if 'HBondDonorCount' in df.columns:
            features['h_donors'] = pd.to_numeric(df['HBondDonorCount'], errors='coerce')
        
        if 'HBondAcceptorCount' in df.columns:
            features['h_acceptors'] = pd.to_numeric(df['HBondAcceptorCount'], errors='coerce')
        
        # Add colorant name as categorical
        if 'colorant_name' in df.columns:
            features['colorant_name'] = df['colorant_name']
        
        # Fill NaN values
        features = features.fillna(features.mean(numeric_only=True))
        
        logging.info(f"  ✅ Extracted {len(features.columns)} features from PubChem")
        return features
    
    # Add this to ColorantDataProcessor class:

    def augment_data(self, X: np.ndarray, y: np.ndarray, n_augmented: int = 50):
        """Create synthetic variations by adding noise"""
        X_aug = []
        y_aug = []
    
        for _ in range(n_augmented):
        # Random sample
            idx = np.random.randint(len(X))
        
        # Add small noise (5% std)
            X_noisy = X[idx] + np.random.normal(0, 0.05, X[idx].shape)
            y_noisy = y[idx] + np.random.normal(0, 0.05, y[idx].shape)
        
            X_aug.append(X_noisy)
            y_aug.append(y_noisy)
    
    # Concatenate
        X_combined = np.vstack([X, np.array(X_aug)])
        y_combined = np.vstack([y, np.array(y_aug)])
    
        return X_combined, y_combined


# Now you have 113 samples instead of 13!
    def process_fda_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from FDA data"""
        logging.info("📋 Processing FDA regulatory data...")
        
        features = pd.DataFrame()
        
        # Color encoding (RGB approximation)
        color_map = {
            'Red': [255, 0, 0],
            'Orange': [255, 165, 0],
            'Yellow': [255, 255, 0],
            'Green': [0, 255, 0],
            'Blue': [0, 0, 255],
            'Purple': [128, 0, 128],
            'Red to Purple': [192, 0, 128],
            'Orange/Yellow': [255, 200, 0],
            'Yellow to Orange': [255, 200, 0],
            'Yellow to Brown': [200, 150, 50],
        }
        
        if 'color' in df.columns:
            colors = df['color'].map(lambda c: color_map.get(c, [128, 128, 128]))
            features['color_r'] = [c[0] for c in colors]
            features['color_g'] = [c[1] for c in colors]
            features['color_b'] = [c[2] for c in colors]
        
        # Regulatory features
        features['fda_approved'] = 1  # All FDA data is approved
        
        if 'year_approved' in df.columns:
            features['approval_year'] = pd.to_numeric(df['year_approved'], errors='coerce')
            # Normalize to 0-1 range
            features['approval_recency'] = (features['approval_year'] - 1900) / 125
        
        # Add colorant name
        if 'name' in df.columns:
            features['colorant_name'] = df['name']
        
        features = features.fillna(0)
        
        logging.info(f"  ✅ Extracted {len(features.columns)} features from FDA")
        return features
    
    def process_literature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract kinetics features from literature"""
        logging.info("📚 Processing literature kinetics data...")
        
        features = pd.DataFrame()
        
        # Degradation kinetics
        if 'degradation_rate_constant_k' in df.columns:
            features['k_constant'] = pd.to_numeric(df['degradation_rate_constant_k'], errors='coerce')
        
        if 'activation_energy_Ea' in df.columns:
            features['activation_energy'] = pd.to_numeric(df['activation_energy_Ea'], errors='coerce')
        
        if 'temperature_C' in df.columns:
            features['test_temp'] = pd.to_numeric(df['temperature_C'], errors='coerce')
        
        if 'pH' in df.columns:
            features['ph'] = pd.to_numeric(df['pH'], errors='coerce')
        
        # Stability indicators
        stability_map = {'poor': 0, 'moderate': 0.5, 'good': 1}
        if 'light_stability' in df.columns:
            features['light_stability'] = df['light_stability'].map(lambda x: stability_map.get(str(x).lower(), 0.5))
        
        sensitivity_map = {'low': 0, 'moderate': 0.5, 'high': 1}
        if 'oxygen_sensitivity' in df.columns:
            features['oxygen_sensitivity'] = df['oxygen_sensitivity'].map(lambda x: sensitivity_map.get(str(x).lower(), 0.5))
        
        # Add colorant name
        if 'colorant_name' in df.columns:
            features['colorant_name'] = df['colorant_name']
        
        features = features.fillna(features.mean(numeric_only=True))
        
        logging.info(f"  ✅ Extracted {len(features.columns)} features from literature")
        return features
    
    def merge_all_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all data sources on colorant name"""
        logging.info("🔗 Merging all data sources...")
        
        # Process each dataset
        pubchem_features = self.process_pubchem_data(data['pubchem_colorants'])
        fda_features = self.process_fda_data(data['fda_approved_colorants'])
        literature_features = self.process_literature_data(data['literature_kinetics_data'])
        
        # Merge on colorant_name
        merged = pubchem_features.copy()
        
        if 'colorant_name' in merged.columns:
            # Merge FDA data
            if 'colorant_name' in fda_features.columns:
                merged = merged.merge(
                    fda_features,
                    on='colorant_name',
                    how='left',
                    suffixes=('', '_fda')
                )
            
            # Merge literature data
            if 'colorant_name' in literature_features.columns:
                merged = merged.merge(
                    literature_features,
                    on='colorant_name',
                    how='left',
                    suffixes=('', '_lit')
                )
        
        # Drop colorant_name (categorical, handled separately)
        if 'colorant_name' in merged.columns:
            colorant_names = merged['colorant_name']
            merged = merged.drop('colorant_name', axis=1)
        
        # Fill remaining NaN
        merged = merged.fillna(0)
        
        logging.info(f"  ✅ Merged dataset: {len(merged)} rows, {len(merged.columns)} features")
        
        return merged, colorant_names if 'colorant_names' in locals() else None
    
    def create_training_data(self, merged_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create X, y for training"""
        logging.info("🎯 Creating training data...")
        
        # For now, we'll create synthetic targets for demonstration
        # In real scenario, this would come from Eurofins experiments
        
        X = merged_df.values
        
        # Synthetic target: predict optimal mixture ratios
        # This is a placeholder - real targets come from experiments
        y = np.random.randn(len(X), ColorantModelConfig.OUTPUT_DIM)
        
        logging.info(f"  X shape: {X.shape}")
        logging.info(f"  y shape: {y.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # In create_training_data(), add:
        X_scaled, y = self.create_training_data(merged_df)

# Add augmentation
        X_scaled, y = self.augment_data(X_scaled, y, n_augmented=100)
        
        self.feature_names = merged_df.columns.tolist()
        
        return X_scaled, y

# ==============================================================================
# DATASET
# ==============================================================================

class ColorantDataset(Dataset):
    """PyTorch Dataset for colorant data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class ColorantFormulationModel(nn.Module):
    """Neural network for colorant formulation prediction"""
    
    def __init__(self, config: ColorantModelConfig):
        super().__init__()
        
        # Build network
        layers = []
        input_dim = config.INPUT_DIM
        
        for hidden_dim in config.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.OUTPUT_DIM))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ==============================================================================
# TRAINER
# ==============================================================================

class ColorantModelTrainer:
    """Training pipeline for colorant model"""
    
    def __init__(self, config: ColorantModelConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Setup directories
        config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        
        # Logging
        log_file = config.OUTPUT_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def load_and_prepare_data(self):
        """Load data from GCS and prepare for training"""
        self.logger.info("="*70)
        self.logger.info("LOADING AND PREPARING DATA")
        self.logger.info("="*70)
        
        # Load from GCS
        gcs_loader = GCSDataLoader(
            self.config.GCP_PROJECT_ID,
            self.config.GCP_BUCKET_NAME,
            self.config.DATA_DIR
        )
        
        data = gcs_loader.download_data()
        
        # Process data
        processor = ColorantDataProcessor()
        merged_df, colorant_names = processor.merge_all_data(data)
        X, y = processor.create_training_data(merged_df)
        
        # Update input dimension based on actual features
        self.config.INPUT_DIM = X.shape[1]
        self.logger.info(f"📊 Updated INPUT_DIM: {self.config.INPUT_DIM}")
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.logger.info(f"  Train set: {len(X_train)} samples")
        self.logger.info(f"  Val set: {len(X_val)} samples")
        
        # Create datasets
        train_dataset = ColorantDataset(X_train, y_train)
        val_dataset = ColorantDataset(X_val, y_val)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        
        self.logger.info(f"✅ Data loaded and prepared")
        
        return processor
    
    def build_model(self):
        """Build and initialize model"""
        self.logger.info("="*70)
        self.logger.info("BUILDING MODEL")
        self.logger.info("="*70)
        
        self.model = ColorantFormulationModel(self.config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Device: {self.device}")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        self.logger.info("✅ Model built successfully")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.LOG_INTERVAL == 0:
                self.logger.info(
                    f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.CHECKPOINT_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"  💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self):
        """Full training loop"""
        self.logger.info("="*70)
        self.logger.info("STARTING TRAINING")
        self.logger.info("="*70)
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            self.logger.info(f"\n📊 Epoch {epoch}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Logging
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}")
            self.logger.info(f"  LR: {current_lr:.6f}")
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                self.logger.info(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
        
        self.logger.info("\n✅ Training complete!")
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        save_path = self.config.OUTPUT_DIR / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 Training plots saved: {save_path}")

# ==============================================================================
# DATA PROCESSOR - REGRESSION VERSION
# ==============================================================================

class RegressionColorantDataProcessor:
    """Process data for regression (predict properties, not classes)"""
    
    def __init__(self, config: ColorantModelConfig):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.input_features = [
            'MolecularWeight', 'XLogP', 'TPSA', 'Complexity',
            'HBondDonorCount', 'HBondAcceptorCount',
            'RotatableBondCount', 'HeavyAtomCount'
        ]
    
    def process_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process data for regression
        Returns: (input_features, target_properties, sample_weights)
        """
        logging.info("=" * 70)
        logging.info("🔧 PROCESSING DATA FOR REGRESSION")
        logging.info("=" * 70)
        
        # Extract input features
        input_data = self._extract_input_features(df)
        
        # Extract target properties
        target_data = self._extract_targets(df)
        
        # Create sample weights
        sample_weights = self._create_sample_weights(df)
        
        logging.info("=" * 70)
        logging.info(f"✅ PROCESSED DATA SUMMARY")
        logging.info("=" * 70)
        logging.info(f"  Total samples: {len(input_data)}")
        logging.info(f"  Input dimension: {input_data.shape[1]}")
        logging.info(f"  Output dimension: {target_data.shape[1]}")
        logging.info(f"  Data sources: {df['dataset_source'].value_counts().to_dict()}")
        logging.info("=" * 70)
        
        return input_data, target_data, sample_weights
    
    def _extract_input_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract input features"""
        logging.info("\n🔧 Extracting input features...")
        
        feature_arrays = []
        
        # Numeric features
        available_features = [f for f in self.input_features if f in df.columns]
        if available_features:
            numeric_data = df[available_features].values
            numeric_data = np.nan_to_num(numeric_data, nan=0.0)
            numeric_data = self.feature_scaler.fit_transform(numeric_data)
            feature_arrays.append(numeric_data)
            logging.info(f"  ✅ Numeric features: {len(available_features)}")
        
        # One-hot encode data source
        if 'dataset_source' in df.columns:
            for source in df['dataset_source'].unique():
                feature_arrays.append((df['dataset_source'] == source).astype(float).values.reshape(-1, 1))
            logging.info(f"  ✅ Encoded data sources")
        
        # Combine
        features = np.hstack(feature_arrays)
        
        # Pad to INPUT_DIM
        if features.shape[1] < self.config.INPUT_DIM:
            padding = np.zeros((features.shape[0], self.config.INPUT_DIM - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > self.config.INPUT_DIM:
            features = features[:, :self.config.INPUT_DIM]
        
        logging.info(f"  ✅ Final input shape: {features.shape}")
        
        return features
    
    def _extract_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target properties"""
        logging.info("\n🎯 Extracting target properties...")
        
        target_features = [f for f in self.config.TARGET_FEATURES if f in df.columns]
        
        if not target_features:
            raise ValueError("No target features found in data!")
        
        targets = df[target_features].values
        targets = np.nan_to_num(targets, nan=0.0)
        targets = self.target_scaler.fit_transform(targets)
        
        logging.info(f"  ✅ Target properties: {target_features}")
        logging.info(f"  ✅ Target shape: {targets.shape}")
        
        return targets
    
    def _create_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Create sample weights"""
        if 'dataset_source' not in df.columns:
            return np.ones(len(df))
        
        weights = np.ones(len(df))
        
        for source, weight in self.config.DATA_SOURCE_WEIGHTS.items():
            mask = df['dataset_source'] == source
            weights[mask] = weight
        
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
# ==============================================================================
# DATASET
# ==============================================================================

class RegressionColorantDataset(Dataset):
    """PyTorch Dataset for regression"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 weights: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.features[idx], self.targets[idx], self.weights[idx]
        return self.features[idx], self.targets[idx]
    
# ==============================================================================
# MODEL - REGRESSION VERSION
# ==============================================================================

class ColorantRegressionModel(nn.Module):
    """Neural network for colorant property prediction (regression)"""
    
    def __init__(self, config: ColorantModelConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.INPUT_DIM
        
        for hidden_dim in config.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT)
            ])
            in_dim = hidden_dim
        
        # Output layer for regression (no activation)
        layers.append(nn.Linear(in_dim, config.OUTPUT_DIM))
        
        self.network = nn.Sequential(*layers)
        self.num_params = sum(p.numel() for p in self.parameters())
        
        logging.info(f"  📊 Model parameters: {self.num_params:,}")
    
    def forward(self, x):
        return self.network(x)
    
class RegressionTrainer:
    """Trainer for regression model"""
    
    def __init__(self, config: ColorantModelConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """Train the model"""
        
        logging.info("=" * 70)
        logging.info("🚀 STARTING REGRESSION TRAINING")
        logging.info("=" * 70)
        
        model = ColorantRegressionModel(self.config).to(self.device)
        logging.info(f"  Model: {model.num_params:,} parameters")
        logging.info(f"  Device: {self.device}")
        
        # MSE loss for regression
        criterion = nn.MSELoss(reduction='none')
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            
            val_loss = self._validate_epoch(model, val_loader, criterion)
            self.val_losses.append(val_loss)
            
            scheduler.step()
            
            if epoch % self.config.LOG_INTERVAL == 0:
                logging.info(
                    f"Epoch {epoch:3d}/{self.config.NUM_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
            
            if val_loss < self.best_val_loss - self.config.MIN_DELTA:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(model, epoch, 'best')
                logging.info(f"  💾 Saved best model (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.PATIENCE:
                logging.info(f"  ⏹️  Early stopping at epoch {epoch}")
                break
            
            if epoch % self.config.SAVE_INTERVAL == 0:
                self._save_checkpoint(model, epoch, f'epoch_{epoch}')
        
        self._plot_training_curves()
        
        logging.info("=" * 70)
        logging.info("✅ TRAINING COMPLETE")
        logging.info(f"  Best validation loss: {self.best_val_loss:.4f}")
        logging.info("=" * 70)
        
        return model
    
    def _train_epoch(self, model: nn.Module, loader: DataLoader,
                     criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        total_loss = 0
        
        for batch_data in loader:
            if len(batch_data) == 3:
                features, targets, weights = batch_data
                features = features.to(self.device)
                targets = targets.to(self.device)
                weights = weights.to(self.device)
            else:
                features, targets = batch_data
                features = features.to(self.device)
                targets = targets.to(self.device)
                weights = None
            
            outputs = model(features)
            loss = criterion(outputs, targets).mean(dim=1)  # Average over output dims
            
            if weights is not None:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, model: nn.Module, loader: DataLoader,
                       criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data in loader:
                if len(batch_data) == 3:
                    features, targets, _ = batch_data
                else:
                    features, targets = batch_data
                
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(features)
                loss = criterion(outputs, targets).mean()
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _save_checkpoint(self, model: nn.Module, epoch: int, name: str):
        checkpoint_path = self.config.CHECKPOINT_DIR / f"model_{name}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
    
    def _plot_training_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Val Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Loss (Regression)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.config.OUTPUT_DIR / 'training_curves.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logging.info(f"  📊 Saved training curves to {plot_path}")
# ==============================================================================
# MAIN
# ==============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_colorant_v2_regression.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 70)
    logging.info("🎨 COLORANT MODEL - REGRESSION VERSION")
    logging.info("Predicting colorant properties (not classification)")
    logging.info("=" * 70)
    
    config = ColorantModelConfig()
    
    # Load data
    data_loader = ExpandedGCSDataLoader(
        config.GCP_PROJECT_ID,
        config.GCP_BUCKET_NAME,
        config.DATA_DIR
    )
    
    try:
        df = data_loader.load_combined_data()
    except FileNotFoundError as e:
        logging.error(str(e))
        return
    
    # Process data
    processor = RegressionColorantDataProcessor(config)
    features, targets, weights = processor.process_data(df)
    
    # Split data - NO STRATIFICATION NEEDED
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        features, targets, weights,
        test_size=0.2,
        random_state=42
    )
    
    logging.info(f"\n  Train: {len(X_train)} samples")
    logging.info(f"  Val:   {len(X_val)} samples")
    
    # Create datasets
    train_dataset = RegressionColorantDataset(X_train, y_train, w_train)
    val_dataset = RegressionColorantDataset(X_val, y_val, w_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Train
    trainer = RegressionTrainer(config)
    model = trainer.train(train_loader, val_loader)
    
    # Upload to GCS
    logging.info("\n📤 Uploading to GCS...")
    client = storage.Client(project=config.GCP_PROJECT_ID)
    bucket = client.bucket(config.GCP_BUCKET_NAME)
    
    model_blob = bucket.blob('models/colorant_model_v2_regression_best.pt')
    model_blob.upload_from_filename(str(config.CHECKPOINT_DIR / 'model_best.pt'))
    logging.info(f"  ☁️  Uploaded model")
    
    logging.info("\n✅ COMPLETE!")

if __name__ == "__main__":
    main()