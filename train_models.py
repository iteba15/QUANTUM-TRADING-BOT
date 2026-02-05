#!/usr/bin/env python3
"""
ML Model Training Script
Trains LSTM + Transformer models on collected data
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
from ml_engine import EnsemblePredictor, MarketDataset
from data_aggregator import MarketSnapshot
import torch

class ModelTrainer:
    """
    Handles training of ML models on collected data
    """
    
    def __init__(self, data_dir: str = 'training_data'):
        self.data_dir = Path(data_dir)
        self.predictor = EnsemblePredictor()
    
    def load_training_data(self, symbol: str) -> Dict:
        """Load collected data for a symbol"""
        pickle_file = self.data_dir / f"{symbol}_snapshots.pkl"
        
        if not pickle_file.exists():
            raise FileNotFoundError(f"No training data found for {symbol}")
        
        print(f"\n[LOAD] Loading {symbol} data...")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        snapshots = data['snapshots']
        labels = data['labels']
        
        # Filter out unlabeled samples
        labeled_data = [
            (snap, label) for snap, label in zip(snapshots, labels)
            if label is not None
        ]
        
        if not labeled_data:
            raise ValueError(f"No labeled data for {symbol}")
        
        snapshots, labels = zip(*labeled_data)
        
        print(f"  [OK] Loaded {len(snapshots)} labeled samples")
        print(f"  [OK] Labels: {sum(labels)/len(labels)*100:.1f}% UP, {(1-sum(labels)/len(labels))*100:.1f}% DOWN")
        
        return {
            'snapshots': list(snapshots),
            'labels': list(labels),
            'metadata': data['metadata']
        }
    
    def split_data(self, snapshots: List, labels: List, 
                   train_ratio: float = 0.8) -> Dict:
        """Split data into train/validation sets"""
        
        # Split chronologically (important for time series!)
        split_idx = int(len(snapshots) * train_ratio)
        
        train_snapshots = snapshots[:split_idx]
        train_labels = labels[:split_idx]
        
        val_snapshots = snapshots[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"\n[DATA] Data Split:")
        print(f"  Training: {len(train_snapshots)} samples")
        print(f"  Validation: {len(val_snapshots)} samples")
        
        return {
            'train_snapshots': train_snapshots,
            'train_labels': train_labels,
            'val_snapshots': val_snapshots,
            'val_labels': val_labels
        }
    
    def train_model(self, symbol: str, epochs: int = 50, 
                   batch_size: int = 32, learning_rate: float = 0.001):
        """
        Complete training pipeline for a symbol
        """
        
        print("="*70)
        print(f"TRAINING ML MODELS FOR {symbol}")
        print("="*70)
        
        # Load data
        data = self.load_training_data(symbol)
        
        # Check minimum samples
        min_samples = 100
        if len(data['snapshots']) < min_samples:
            print(f"\n[ERROR] Need at least {min_samples} samples to train")
            print(f"   Current: {len(data['snapshots'])} samples")
            print(f"   Collect more data with: python collect_training_data.py")
            return False
        
        # Split data
        split = self.split_data(data['snapshots'], data['labels'])
        
        # Check if we have GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[INFO] Using device: {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Incremental Training: Load existing model if available
        model_path = f"models/{symbol}_trained.pth"
        if Path(model_path).exists():
            print(f"\n[INFO] Found existing model for {symbol}. Loading for fine-tuning...")
            try:
                self.predictor.load_models(model_path)
                print(f"   [OK] Loaded weights from {model_path}")
            except Exception as e:
                print(f"   [WARN] Could not load existing weights: {e}")
                print("   Starting fresh training.")
        
        # Train
        print(f"\n[START] Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        self.predictor.train_models(
            train_snapshots=split['train_snapshots'],
            val_snapshots=split['val_snapshots'],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save trained model
        model_path = f"models/{symbol}_trained.pth"
        Path("models").mkdir(exist_ok=True)
        self.predictor.save_models(model_path)
        
        print(f"\n[OK] Training complete!")
        print(f"   Model saved to: {model_path}")
        
        # Test on validation set
        self.evaluate_model(split['val_snapshots'], split['val_labels'])
        
        return True
    
    def evaluate_model(self, val_snapshots: List, val_labels: List):
        """Evaluate model on validation set"""
        
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        correct = 0
        total = 0
        confidences = []
        
        # Need at least 20 snapshots for sequence
        if len(val_snapshots) < 20:
            print("Not enough validation data for evaluation")
            return
        
        for i in range(20, len(val_snapshots)):
            # Get sequence
            sequence = val_snapshots[i-20:i]
            true_label = val_labels[i]
            
            # Predict
            prediction = self.predictor.predict(sequence)
            predicted_label = 1.0 if prediction.probability_up > 0.5 else 0.0
            
            # Check
            if predicted_label == true_label:
                correct += 1
            total += 1
            confidences.append(prediction.confidence)
        
        accuracy = correct / total * 100 if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        print(f"\n[RESULTS] Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Correct predictions: {correct}/{total}")
        print(f"  Average confidence: {avg_confidence:.2%}")
        
        if accuracy > 55:
            print(f"\n[OK] GOOD: Above random (50%)")
        elif accuracy > 50:
            print(f"\n[WARN] MARGINAL: Slightly above random")
        else:
            print(f"\n[ERROR] POOR: At or below random - collect more data")
        
        return accuracy
    
    def train_all_symbols(self, symbols: List[str] = ['BTC', 'ETH', 'SOL', 'XRP'],
                         epochs: int = 50):
        """Train models for all symbols"""
        
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        results = {}
        
        for symbol in symbols:
            try:
                success = self.train_model(symbol, epochs=epochs)
                results[symbol] = 'SUCCESS' if success else 'FAILED'
            except FileNotFoundError:
                print(f"\n[WARN] No data collected for {symbol} - skipping")
                results[symbol] = 'NO DATA'
            except Exception as e:
                print(f"\n[ERROR] Error training {symbol}: {e}")
                results[symbol] = 'ERROR'
        
        # Summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        for symbol, result in results.items():
            icon = "[OK]" if result == 'SUCCESS' else "[FAIL]"
            print(f"  {icon} {symbol}: {result}")

def quick_start_guide():
    """Print quick start guide"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              ML MODEL TRAINING QUICK START                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    STEP 1: Collect Training Data
    ──────────────────────────────
    Run the data collector for 8-24 hours:
    
        python collect_training_data.py
    
    This will create files in training_data/:
        - BTC_snapshots.pkl
        - ETH_snapshots.pkl
        - SOL_snapshots.pkl
        - XRP_snapshots.pkl
    
    Minimum: 1000 samples (8 hours)
    Recommended: 5000 samples (42 hours)
    
    
    STEP 2: Train Models
    ────────────────────
    Once you have data, train models:
    
        python train_models.py --symbol BTC --epochs 50
    
    Or train all at once:
    
        python train_models.py --all --epochs 50
    
    Training time (RTX 4070 Ti Super):
        - 1000 samples: ~10-15 minutes
        - 5000 samples: ~30-40 minutes
        - 10000 samples: ~60-90 minutes
    
    
    STEP 3: Use Trained Models
    ───────────────────────────
    Models are saved to models/ directory:
        - BTC_trained.pth
        - ETH_trained.pth
        - etc.
    
    The quantum_predictor.py will automatically load them!
    
    
    STEP 4: Keep Collecting Data
    ─────────────────────────────
    ML models need retraining as markets change:
        - Retrain weekly with new data
        - Keep data collector running 24/7
        - Append new data to existing files
    
    
    ═══════════════════════════════════════════════════════════════════
    
    💡 PRO TIP: Start trading with physics-only mode while collecting
               data. Train models after 1-2 days of collection.
    
    ═══════════════════════════════════════════════════════════════════
    """)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models on collected data')
    parser.add_argument('--symbol', type=str, help='Symbol to train (BTC, ETH, SOL, XRP)')
    parser.add_argument('--all', action='store_true', help='Train all symbols')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--guide', action='store_true', help='Show quick start guide')
    
    args = parser.parse_args()
    
    if args.guide:
        quick_start_guide()
        sys.exit(0)
    
    trainer = ModelTrainer()
    
    if args.all:
        trainer.train_all_symbols(epochs=args.epochs)
    elif args.symbol:
        trainer.train_model(
            args.symbol,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    else:
        print("Error: Specify --symbol or --all")
        print("Example: python train_models.py --symbol BTC --epochs 50")
        print("         python train_models.py --all")
        print("\nFor help: python train_models.py --guide")
