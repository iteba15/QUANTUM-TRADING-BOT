import pickle
import sys
from pathlib import Path
from ml_engine import EnsemblePredictor
from data_aggregator import MarketSnapshot
import numpy as np

def run_backtest(days=1, symbol='BTC'):
    """
    Run a simple backtest of the ML model on collected data
    """
    print("="*70)
    print(f"BACKTESTING ML MODEL ({symbol})")
    print("="*70)
    
    # Load data
    data_file = Path('training_data') / f"{symbol}_snapshots.pkl"
    if not data_file.exists():
        print(f"[ERROR] No data file found at {data_file}")
        print("Run collect_training_data.py or convert_historical.py first.")
        return

    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        
    snapshots = data['snapshots']
    labels = data['labels']
    
    if len(snapshots) < 100:
        print("[ERROR] Not enough data points to backtest.")
        return
        
    # Load model
    predictor = EnsemblePredictor()
    
    # Try to load trained model
    model_path = f"models/{symbol}_trained.pth"
    if Path(model_path).exists():
        print(f"Loading model from {model_path}...")
        predictor.load_models(model_path)
    else:
        print(f"[WARN] No trained model found at {model_path}. Predictions will be random.")
    
    # Simulation vars
    balance = 10000
    position = 0
    trades = 0
    wins = 0
    
    print(f"Starting Balance: ${balance:,.2f}")
    print(f"Data points: {len(snapshots)}")
    
    history_len = 20
    
    for i in range(history_len, len(snapshots) - 1):
        # We need a sequence for prediction
        sequence = snapshots[i-history_len:i]
        current_snap = snapshots[i]
        next_snap = snapshots[i+1]
        
        # Predict
        # Note: In real backtest we'd be careful about lookahead bias
        # Here we assume the model was trained on past data or we are testing capacity
        
        try:
            prediction = predictor.predict(sequence)
        except Exception:
            continue
            
        # Strategy: Buy if confidence > 60% UP, Sell if confidence > 60% DOWN
        
        action = None
        if prediction.probability_up > 0.6 and prediction.confidence > 0.5:
            action = 'BUY'
        elif prediction.probability_up < 0.4 and prediction.confidence > 0.5:
            action = 'SELL'
            
        if action == 'BUY' and position == 0:
            # Enter Long
            position = balance / current_snap.price
            entry_price = current_snap.price
            balance = 0
            trades += 1
            # print(f"  BUY @ {entry_price:.2f}")
            
        elif action == 'SELL' and position > 0:
            # Exit Long
            balance = position * current_snap.price
            profit = (current_snap.price - entry_price) / entry_price
            if profit > 0:
                wins += 1
            position = 0
            # print(f"  SELL @ {current_snap.price:.2f} (Profit: {profit*100:.2f}%)")

    # Final tally
    if position > 0:
        balance = position * snapshots[-1].price
        
    print("-" * 30)
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Total Trades: {trades}")
    print(f"Win Rate: {wins/trades*100:.1f}%" if trades > 0 else "Win Rate: N/A")
    print(f"Return: {(balance - 10000)/10000*100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_backtest()
