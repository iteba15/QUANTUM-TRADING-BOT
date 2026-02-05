#!/usr/bin/env python3
"""
ML Prediction Engine - GPU Accelerated
Uses LSTM, Transformers, and ensemble methods for time series prediction
Optimized for RTX 4070 Ti Super
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path

@dataclass
class PredictionResult:
    """ML model prediction output"""
    probability_up: float
    probability_down: float
    confidence: float
    model_scores: Dict[str, float]  # Individual model predictions
    features_importance: Dict[str, float]

class MarketDataset(Dataset):
    """PyTorch dataset for market snapshots"""
    
    def __init__(self, snapshots: List, sequence_length: int = 20):
        self.snapshots = snapshots
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.snapshots) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of snapshots
        sequence = self.snapshots[idx:idx + self.sequence_length]
        
        # Extract features
        features = self._extract_features(sequence)
        
        # Target: 1 if price went up, 0 if down
        target_price = self.snapshots[idx + self.sequence_length].price
        current_price = self.snapshots[idx + self.sequence_length - 1].price
        target = 1.0 if target_price > current_price else 0.0
        
        return torch.FloatTensor(features), torch.FloatTensor([target])
    
    def _extract_features(self, sequence) -> np.ndarray:
        """Extract feature vector from sequence of snapshots"""
        features = []
        
        if not sequence:
            return np.array([])
            
        base_price = sequence[0].price if sequence[0].price > 0 else 1.0
        
        for snapshot in sequence:
            # Normalize features to reasonable range (approx -1 to 1 or 0 to 1)
            norm_price = (snapshot.price / base_price) - 1.0  # Percentage stats from start
            norm_cvd = snapshot.cvd / 1e9 if snapshot.cvd else 0.0 # Billions
            norm_oi = snapshot.open_interest / 1e9 if snapshot.open_interest else 0.0 # Billions
            
            # Handle potential NaNs in source data
            row = [
                norm_price,
                norm_cvd,
                norm_oi,
                snapshot.open_interest_change_pct if snapshot.open_interest_change_pct else 0.0,
                (snapshot.funding_rate * 1000) if snapshot.funding_rate else 0.0,
                snapshot.long_short_ratio if snapshot.long_short_ratio else 1.0,
                snapshot.volume_imbalance if snapshot.volume_imbalance else 1.0,
                (snapshot.volume_delta / 1e6) if snapshot.volume_delta else 0.0, # Millions
            ]
            
            # Final NaN check safety
            row = [0.0 if np.isnan(x) or np.isinf(x) else x for x in row]
            features.append(row)
        
        return np.array(features)

class LSTMPredictor(nn.Module):
    """
    LSTM network for time series prediction
    Captures temporal patterns in market data
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        out = F.relu(self.fc1(context_vector))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        
        return out

class TransformerPredictor(nn.Module):
    """
    Transformer network with multi-head attention
    Better at capturing long-range dependencies
    """
    
    def __init__(self, input_size: int = 8, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.3):
        super(TransformerPredictor, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Output layers
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        
        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EnsemblePredictor:
    """
    Ensemble of LSTM and Transformer models
    Combines predictions with weighted voting
    """
    
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize models
        self.lstm_model = LSTMPredictor().to(self.device)
        self.transformer_model = TransformerPredictor().to(self.device)
        
        # Model weights (can be tuned based on validation performance)
        self.model_weights: Dict[str, float] = {
            'lstm': 0.5,
            'transformer': 0.5
        }
        
        self.trained = False
    
    def train_models(self, 
                    train_snapshots: List,
                    val_snapshots: List,
                    epochs: int = 50,
                    batch_size: int = 32,
                    learning_rate: float = 0.001):
        """
        Train both models on historical data
        """
        
        print("\n" + "=" * 70)
        print("TRAINING ML MODELS")
        print("=" * 70)
        
        # Create datasets
        train_dataset = MarketDataset(train_snapshots, sequence_length=20)
        val_dataset = MarketDataset(val_snapshots, sequence_length=20)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizers
        criterion = nn.BCELoss()
        lstm_optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
        transformer_optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=learning_rate)
        
        # Learning rate schedulers
        lstm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, patience=5, factor=0.5)
        transformer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train LSTM
            lstm_train_loss = self._train_epoch(
                self.lstm_model, train_loader, criterion, lstm_optimizer
            )
            
            # Train Transformer
            transformer_train_loss = self._train_epoch(
                self.transformer_model, train_loader, criterion, transformer_optimizer
            )
            
            # Validation
            lstm_val_loss = self._validate(self.lstm_model, val_loader, criterion)
            transformer_val_loss = self._validate(self.transformer_model, val_loader, criterion)
            
            # Update learning rates
            lstm_scheduler.step(lstm_val_loss)
            transformer_scheduler.step(transformer_val_loss)
            
            # Combined validation loss
            combined_val_loss = (lstm_val_loss + transformer_val_loss) / 2
            
            if epoch % 5 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  LSTM     - Train Loss: {lstm_train_loss:.4f}, Val Loss: {lstm_val_loss:.4f}")
                print(f"  Transformer - Train Loss: {transformer_train_loss:.4f}, Val Loss: {transformer_val_loss:.4f}")
            
            # Save best model
            if combined_val_loss < best_val_loss:
                best_val_loss = combined_val_loss
                self.save_models('best_model.pth')
        
        print(f"\n[OK] Training complete. Best validation loss: {best_val_loss:.4f}")
        self.trained = True
    
    def _train_epoch(self, model, data_loader, criterion, optimizer):
        """Single training epoch"""
        model.train()
        total_loss = 0
        
        for features, targets in data_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _validate(self, model, data_loader, criterion):
        """Validation pass"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def predict(self, recent_snapshots: List) -> PredictionResult:
        """
        Make prediction on recent market data
        """
        if not self.trained:
            print("[WARN] Models not trained yet. Using random predictions.")
        
        # Prepare data
        dataset = MarketDataset(recent_snapshots, sequence_length=20)
        if len(dataset) == 0:
            return self._create_neutral_prediction("Insufficient data")
        
        # Get last sequence
        features, _ = dataset[-1]
        features = features.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get predictions from both models
        self.lstm_model.eval()
        self.transformer_model.eval()
        
        with torch.no_grad():
            lstm_pred = self.lstm_model(features).cpu().item()
            transformer_pred = self.transformer_model(features).cpu().item()
        
        # Ensemble prediction
        ensemble_pred = (
            lstm_pred * self.model_weights['lstm'] +
            transformer_pred * self.model_weights['transformer']
        )
        
        # Calculate confidence (based on model agreement)
        model_agreement = 1.0 - abs(lstm_pred - transformer_pred)
        confidence = model_agreement * min(abs(ensemble_pred - 0.5) * 2, 1.0)
        
        return PredictionResult(
            probability_up=ensemble_pred,
            probability_down=1.0 - ensemble_pred,
            confidence=confidence,
            model_scores={
                'lstm': lstm_pred,
                'transformer': transformer_pred,
                'ensemble': ensemble_pred
            },
            features_importance=self._calculate_feature_importance(features)
        )
    
    def _calculate_feature_importance(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Estimate feature importance using gradient-based method
        """
        feature_names = [
            'price', 'cvd', 'open_interest', 'oi_change',
            'funding_rate', 'ls_ratio', 'volume_imbalance', 'volume_delta'
        ]
        
        # For now, return uniform importance
        # In production, use proper feature importance calculation
        return {name: 1.0/len(feature_names) for name in feature_names}
    
    def _create_neutral_prediction(self, reason: str) -> PredictionResult:
        """Return neutral prediction when analysis cannot be performed"""
        return PredictionResult(
            probability_up=0.5,
            probability_down=0.5,
            confidence=0.0,
            model_scores={'lstm': 0.5, 'transformer': 0.5, 'ensemble': 0.5},
            features_importance={}
        )
    
    def save_models(self, filepath: str):
        """Save model weights"""
        torch.save({
            'lstm_state_dict': self.lstm_model.state_dict(),
            'transformer_state_dict': self.transformer_model.state_dict(),
            'model_weights': self.model_weights
        }, filepath)
        print(f"[OK] Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.lstm_model.load_state_dict(checkpoint['lstm_state_dict'])
        self.transformer_model.load_state_dict(checkpoint['transformer_state_dict'])
        self.model_weights = checkpoint['model_weights']
        self.trained = True
        print(f"[OK] Models loaded from {filepath}")

# Test script
if __name__ == "__main__":
    print("GPU-Accelerated ML Prediction Engine")
    print("=" * 70)
    
    # Check CUDA availability
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create predictor
    predictor = EnsemblePredictor()
    
    # Generate mock training data
    print("\nGenerating mock training data...")
    from data_aggregator import MarketSnapshot
    import time
    
    mock_snapshots: List[MarketSnapshot] = []
    for i in range(1000):
        snapshot = MarketSnapshot(
            timestamp=time.time() - (1000 - i) * 60,
            symbol='BTC',
            price=95000 + np.random.randn() * 500,
            volume_24h=1000000,
            cvd=np.random.randn() * 100,
            open_interest=5000000000 + np.random.randn() * 100000000,
            open_interest_change_pct=np.random.randn() * 5,
            funding_rate=np.random.randn() * 0.01,
            long_short_ratio=1.0 + np.random.randn() * 0.3,
            liquidation_cluster_above=96000,
            liquidation_cluster_below=94000,
            liquidation_strength_above=1000000000,
            liquidation_strength_below=800000000,
            volume_delta=np.random.randn() * 50,
            volume_imbalance=1.0 + np.random.randn() * 0.5
        )
        mock_snapshots.append(snapshot)
    
    # Split train/val
    split_idx = int(len(mock_snapshots) * 0.8)
    train_data = mock_snapshots[:split_idx]
    val_data = mock_snapshots[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Train models (use small epochs for demo)
    predictor.train_models(
        train_snapshots=train_data,
        val_snapshots=val_data,
        epochs=10,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Make prediction
    print("\n" + "=" * 70)
    print("MAKING PREDICTION")
    print("=" * 70)
    
    result = predictor.predict(mock_snapshots[-30:])
    
    print(f"\nProbability UP: {result.probability_up:.1%}")
    print(f"Probability DOWN: {result.probability_down:.1%}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"\nModel Scores:")
    for model, score in result.model_scores.items():
        print(f"  {model}: {score:.1%}")
