#!/usr/bin/env python3
"""
Training Data Collector
Runs 24/7 collecting market snapshots for ML training
"""

import time
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from data_aggregator import DataAggregator, MarketSnapshot
import signal
import sys

class TrainingDataCollector:
    """
    Collects and stores market snapshots for ML training
    """
    
    def __init__(self, symbols=['BTC', 'ETH', 'SOL', 'XRP']):
        self.symbols = symbols
        self.aggregator = DataAggregator()
        self.data_dir = Path('training_data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage
        self.snapshots = {symbol: [] for symbol in symbols}
        self.labels = {symbol: [] for symbol in symbols}
        
        # Config
        self.collection_interval = 30  # seconds
        self.save_interval = 300  # Save every 5 minutes
        self.prediction_window = 15  # minutes (predict 15min ahead)
        
        self.running = True
        self.last_save = time.time()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Handle shutdown gracefully"""
        print("\n\nShutting down gracefully...")
        self.running = False
        self.save_all_data()
        sys.exit(0)
    
    def collect(self):
        """Main collection loop"""
        print("="*70)
        print("TRAINING DATA COLLECTOR")
        print("="*70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Collection interval: {self.collection_interval}s")
        print(f"Prediction window: {self.prediction_window} minutes")
        print(f"Data directory: {self.data_dir}")
        print("\nStarting data aggregator...")
        
        self.aggregator.start()
        
        print("[OK] Ready. Collecting data...")
        print("Press Ctrl+C to stop and save\n")
        
        collection_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Collect snapshots for all symbols
                for symbol in self.symbols:
                    snapshot = self.aggregator.get_snapshot(symbol)
                    
                    if snapshot:
                        self.snapshots[symbol].append(snapshot)
                        
                        # Create label (will be filled in later)
                        self.labels[symbol].append(None)
                
                collection_count += 1
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Collected {collection_count} batches "
                      f"({elapsed/60:.1f} min elapsed) "
                      f"| Total snapshots: {sum(len(s) for s in self.snapshots.values())}")
                
                # Periodic save
                if time.time() - self.last_save > self.save_interval:
                    self.save_all_data()
                    self.last_save = time.time()
                
                # Wait
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error during collection: {e}")
                time.sleep(5)
    
    def label_data(self):
        """
        Create labels for supervised learning
        Label = 1 if price went up in next 15 min, 0 if down
        """
        print("\nLabeling data...")
        
        for symbol in self.symbols:
            snapshots = self.snapshots[symbol]
            labels = []
            
            # Calculate how many snapshots = prediction window
            # If collecting every 30s, 15min = 30 snapshots
            window_snapshots = int((self.prediction_window * 60) / self.collection_interval)
            
            for i in range(len(snapshots)):
                if i + window_snapshots < len(snapshots):
                    # Compare current price to price N snapshots ahead
                    current_price = snapshots[i].price
                    future_price = snapshots[i + window_snapshots].price
                    
                    # Label: 1 if price went up, 0 if down
                    label = 1.0 if future_price > current_price else 0.0
                    labels.append(label)
                else:
                    # Can't label (not enough future data)
                    labels.append(None)
            
            self.labels[symbol] = labels
            
            valid_labels = [l for l in labels if l is not None]
            if valid_labels:
                up_pct = sum(valid_labels) / len(valid_labels) * 100
                print(f"  {symbol}: {len(valid_labels)} labeled samples ({up_pct:.1f}% up)")
    
    def save_all_data(self):
        """Save collected data to disk"""
        print("\n[SAVE] Saving data...")
        
        # Label data first
        self.label_data()
        
        for symbol in self.symbols:
            if not self.snapshots[symbol]:
                continue
            
            # Save as pickle (preserves Python objects)
            pickle_file = self.data_dir / f"{symbol}_snapshots.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump({
                    'snapshots': self.snapshots[symbol],
                    'labels': self.labels[symbol],
                    'metadata': {
                        'symbol': symbol,
                        'collection_start': self.snapshots[symbol][0].timestamp,
                        'collection_end': self.snapshots[symbol][-1].timestamp,
                        'total_samples': len(self.snapshots[symbol]),
                        'labeled_samples': len([l for l in self.labels[symbol] if l is not None]),
                        'prediction_window_minutes': self.prediction_window
                    }
                }, f)
            
            # Also save metadata as JSON for easy inspection
            json_file = self.data_dir / f"{symbol}_metadata.json"
            with open(json_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'total_samples': len(self.snapshots[symbol]),
                    'labeled_samples': len([l for l in self.labels[symbol] if l is not None]),
                    'collection_start': datetime.fromtimestamp(
                        self.snapshots[symbol][0].timestamp
                    ).isoformat(),
                    'collection_end': datetime.fromtimestamp(
                        self.snapshots[symbol][-1].timestamp
                    ).isoformat(),
                    'prediction_window_minutes': self.prediction_window,
                    'first_price': self.snapshots[symbol][0].price,
                    'last_price': self.snapshots[symbol][-1].price,
                    'price_change_pct': (
                        (self.snapshots[symbol][-1].price - self.snapshots[symbol][0].price) /
                        self.snapshots[symbol][0].price * 100
                    )
                }, f, indent=2)
            
            print(f"  [OK] Saved {len(self.snapshots[symbol])} snapshots for {symbol}")
        
        print(f"[OK] Data saved to {self.data_dir}/")
    
    def load_existing_data(self):
        """Load previously collected data"""
        print("\n[LOAD] Loading existing data...")
        
        for symbol in self.symbols:
            pickle_file = self.data_dir / f"{symbol}_snapshots.pkl"
            
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.snapshots[symbol] = data['snapshots']
                    self.labels[symbol] = data['labels']
                    
                print(f"  [OK] Loaded {len(self.snapshots[symbol])} snapshots for {symbol}")
            else:
                print(f"  [WARN] No existing data for {symbol}")
    
    def get_collection_stats(self):
        """Print statistics about collected data"""
        print("\n" + "="*70)
        print("COLLECTION STATISTICS")
        print("="*70)
        
        for symbol in self.symbols:
            if not self.snapshots[symbol]:
                print(f"\n{symbol}: No data collected yet")
                continue
            
            snapshots = self.snapshots[symbol]
            labels = [l for l in self.labels[symbol] if l is not None]
            
            duration = snapshots[-1].timestamp - snapshots[0].timestamp
            hours = duration / 3600
            
            print(f"\n{symbol}:")
            print(f"  Total snapshots: {len(snapshots)}")
            print(f"  Labeled samples: {len(labels)}")
            print(f"  Duration: {hours:.1f} hours")
            print(f"  Start: {datetime.fromtimestamp(snapshots[0].timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  End: {datetime.fromtimestamp(snapshots[-1].timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            
            if labels:
                up_pct = sum(labels) / len(labels) * 100
                print(f"  Price went UP: {up_pct:.1f}% of time")
                print(f"  Price went DOWN: {100-up_pct:.1f}% of time")
            
            # Price statistics
            prices = [s.price for s in snapshots]
            print(f"  Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
            print(f"  Price change: {(prices[-1] - prices[0]) / prices[0] * 100:+.2f}%")

def estimate_training_time(samples_needed=1000):
    """Estimate how long to collect data"""
    collection_interval = 30  # seconds
    
    total_seconds = samples_needed * collection_interval
    hours = total_seconds / 3600
    
    print("\n" + "="*70)
    print("DATA COLLECTION TIME ESTIMATE")
    print("="*70)
    print(f"\nFor {samples_needed} samples:")
    print(f"  Collection interval: {collection_interval}s")
    print(f"  Total time needed: {hours:.1f} hours ({hours/24:.1f} days)")
    print(f"\nRecommended minimum: 1000 samples (8.3 hours)")
    print(f"Good training set: 5000 samples (41.7 hours / ~2 days)")
    print(f"Excellent training set: 10000 samples (83.3 hours / ~3.5 days)")
    print(f"\n[TIP] Start with 1000 samples, train model, collect more while trading")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data for ML models')
    parser.add_argument('--load', action='store_true', help='Load existing data first')
    parser.add_argument('--estimate', action='store_true', help='Show time estimates and exit')
    parser.add_argument('--backtest', action='store_true', help='Run backtest simulation')
    parser.add_argument('--days', type=int, default=1, help='Days to backtest')
    args = parser.parse_args()
    
    if args.backtest:
        from backtest import run_backtest
        run_backtest(args.days)
        sys.exit(0)
    
    if args.estimate:
        estimate_training_time()
        sys.exit(0)
    
    collector = TrainingDataCollector()
    
    if args.load:
        collector.load_existing_data()
    
    if args.stats:
        collector.get_collection_stats()
        sys.exit(0)
    
    # Start collection
    try:
        collector.collect()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        collector.save_all_data()
        collector.get_collection_stats()
