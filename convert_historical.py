import pandas as pd
from data_aggregator import MarketSnapshot
import pickle
from pathlib import Path
import os
import time

def csv_to_snapshots(csv_file, symbol):
    """Convert CSV to MarketSnapshot objects"""
    print(f"Converting {csv_file} to snapshots...")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return []
    
    snapshots = []
    
    # Calculate simple price changes for simulated features
    df['price_change'] = df['close'].pct_change()
    
    for i, row in df.iterrows():
        # Create snapshot (missing some fields, but price is key)
        # We simulate some fields to avoid ML model crashing on None/Zero
        
        # Simulate simple CVD based on volume direction
        close_price = float(row['close'])
        open_price = float(row['open'])
        volume = float(row['volume'])
        
        # Simple heuristic: Green candle = positive CVD, Red candle = negative CVD
        fake_cvd = volume if close_price >= open_price else -volume
        
        snapshot = MarketSnapshot(
            timestamp=pd.to_datetime(row['timestamp']).timestamp(),
            symbol=symbol,
            price=close_price,
            volume_24h=volume * 24 * 60, # Rough estimate
            cvd=fake_cvd,  # Approximated
            open_interest=0,  # Missing
            open_interest_change_pct=0,
            funding_rate=0.0001, # Default positive funding
            long_short_ratio=1.0,
            liquidation_cluster_above=close_price * 1.05,
            liquidation_cluster_below=close_price * 0.95,
            liquidation_strength_above=0,
            liquidation_strength_below=0,
            volume_delta=fake_cvd,
            volume_imbalance=1.0
        )
        snapshots.append(snapshot)
    
    return snapshots

def process_all_historical_files():
    data_dir = Path('training_data')
    data_dir.mkdir(exist_ok=True)
    
    symbols = ['BTC', 'ETH', 'SOL', 'XRP']
    
    for symbol in symbols:
        csv_file = f"{symbol}_historical.csv"
        
        if not os.path.exists(csv_file):
            print(f"Skipping {symbol} (no CSV found)")
            continue
            
        snapshots = csv_to_snapshots(csv_file, symbol)
        
        if not snapshots:
            continue
            
        # Create labels
        print(f"Labeling {len(snapshots)} snapshots for {symbol}...")
        labels = []
        prediction_window = 30 # 30 minutes (if 1m interval)
        
        for i in range(len(snapshots)):
            if i + prediction_window < len(snapshots):
                future_price = snapshots[i + prediction_window].price
                current_price = snapshots[i].price
                # 1 if UP, 0 if DOWN
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
                    'collection_start': snapshots[0].timestamp,
                    'collection_end': snapshots[-1].timestamp,
                    'total_samples': len(snapshots),
                    'labeled_samples': len([l for l in labels if l is not None]),
                    'prediction_window_minutes': 15, # Approx
                    'source': 'historical_csv'
                }
            }, f)
            
        print(f"[OK] Saved {pickle_file}")

if __name__ == "__main__":
    process_all_historical_files()
