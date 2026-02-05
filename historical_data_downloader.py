import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def download_binance_history(symbol='BTCUSDT', interval='1m', days=7):
    """
    Download historical kline data from Binance
    """
    print(f"Downloading {days} days of {interval} data for {symbol}...")
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    
    while start_time < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': min(start_time + 1000 * 60000 * 1000, end_time),  # Request chunks (e.g. 1000 candles)
            'limit': 1500
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                break
                
            klines = response.json()
            
            if not klines:
                break
            
            all_klines.extend(klines)
            start_time = klines[-1][0] + 1  # Next timestamp
            
            print(f"  Downloaded {len(all_klines)} candles... (Latest: {datetime.fromtimestamp(klines[-1][0]/1000)})")
            time.sleep(0.1)  # Rate limit protection
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
    
    if not all_klines:
        print("No data downloaded.")
        return None

    # Convert to DataFrame
    # Binance columns: Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Keep useful columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
    
    for symbol in symbols:
        df = download_binance_history(symbol, '1m', days=7)
        if df is not None:
            filename = f'{symbol.replace("USDT", "")}_historical.csv'
            df.to_csv(filename, index=False)
            print(f"[OK] Saved {len(df)} candles to {filename}")
