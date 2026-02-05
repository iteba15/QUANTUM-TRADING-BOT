import numpy as np
import time
import pickle
from pathlib import Path
from data_aggregator import MarketSnapshot

def generate_synthetic_snapshots(count=5000, symbol='BTC'):
    """
    Generate synthetic market data with realistic patterns
    For testing ONLY - do not use for real trading!
    """
    print(f"Generating {count} synthetic snapshots for {symbol}...")
    
    snapshots = []
    base_price = 75000 if symbol == 'BTC' else 3000
    
    price = base_price
    
    # Current time
    end_timestamp = time.time()
    
    for i in range(count):
        # Time going backwards from now
        timestamp = end_timestamp - (count - i) * 60 # 1 minute intervals
        
        # Random walk with momentum
        momentum = np.random.randn() * 0.001
        random_shock = np.random.randn() * 0.002
        
        price = price * (1 + momentum + random_shock)
        
        # Synthetic features
        snapshot = MarketSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            volume_24h=np.random.uniform(1e9, 5e9),
            cvd=np.random.randn() * 1000,
            open_interest=np.random.uniform(5e9, 10e9),
            open_interest_change_pct=np.random.randn() * 5,
            funding_rate=np.random.randn() * 0.01,
            long_short_ratio=np.random.uniform(1.5, 3.0),
            liquidation_cluster_above=price * 1.05,
            liquidation_cluster_below=price * 0.95,
            liquidation_strength_above=np.random.uniform(1e8, 1e9),
            liquidation_strength_below=np.random.uniform(1e8, 1e9),
            volume_delta=np.random.randn() * 100,
            volume_imbalance=np.random.uniform(0.8, 1.2)
        )
        snapshots.append(snapshot)
    
    return snapshots

if __name__ == "__main__":
    data_dir = Path('training_data')
    data_dir.mkdir(exist_ok=True)
    
    symbols = ['BTC', 'ETH']
    
    for symbol in symbols:
        snapshots = generate_synthetic_snapshots(2000, symbol)
        
        # Create labels
        labels = []
        prediction_window = 15
        
        for i in range(len(snapshots)):
            if i + prediction_window < len(snapshots):
                future_price = snapshots[i + prediction_window].price
                current_price = snapshots[i].price
                labels.append(1.0 if future_price > current_price else 0.0)
            else:
                labels.append(None)
        
        # Save
        pickle_file = data_dir / f"{symbol}_snapshots.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'snapshots': snapshots,
                'labels': labels,
                'metadata': {
                    'symbol': symbol,
                    'total_samples': len(snapshots),
                    'source': 'synthetic'
                }
            }, f)
            
        print(f"[OK] Generated and saved to {pickle_file}")
