#!/usr/bin/env python3
"""
Polymarket Quantum Predictor - Data Aggregator
Collects real-time data from multiple sources for physics-based analysis
"""

import requests
import websocket
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
from dataclasses import dataclass, asdict
import pandas as pd
from liquidation_estimator import LiquidationEstimator, BinanceFuturesData

@dataclass
class MarketSnapshot:
    """Single point-in-time market state"""
    timestamp: float
    symbol: str
    price: float
    volume_24h: float
    
    # Physics indicators
    cvd: float  # Cumulative Volume Delta
    open_interest: float
    open_interest_change_pct: float
    funding_rate: float
    long_short_ratio: float
    
    # Liquidation data
    liquidation_cluster_above: float  # Price of nearest cluster above
    liquidation_cluster_below: float  # Price of nearest cluster below
    liquidation_strength_above: float  # Size of cluster (USD)
    liquidation_strength_below: float
    
    # Volume profile
    volume_delta: float  # Buy volume - Sell volume
    volume_imbalance: float  # Ratio of buy/sell
    
    # Polymarket data
    polymarket_up_odds: Optional[float] = None
    polymarket_down_odds: Optional[float] = None
    polymarket_volume: Optional[float] = None

# CoinglassAPI class removed as we are using free alternatives (BinanceFutures)

class BinanceWebSocket:
    """Real-time price, volume, and trade data from Binance"""
    
    def __init__(self, symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]):
        self.symbols = symbols
        self.ws = None
        self.trade_history = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.current_prices = {symbol: 0.0 for symbol in symbols}
        self.cvd = {symbol: 0.0 for symbol in symbols}  # Cumulative Volume Delta
        self.running = False
        self.thread = None
    
    def calculate_cvd(self, symbol: str, trade: Dict):
        """Calculate Cumulative Volume Delta from trades"""
        # CVD = sum of (buy volume - sell volume)
        qty = float(trade['q'])
        is_buyer_maker = trade['m']  # True if buyer is maker (sell order)
        
        if is_buyer_maker:
            # Sell order (negative contribution)
            self.cvd[symbol] -= qty
        else:
            # Buy order (positive contribution)
            self.cvd[symbol] += qty
    
    def on_message(self, ws, message):
        """Handle incoming websocket messages"""
        try:
            data = json.loads(message)
            
            if 'e' in data and data['e'] == 'trade':
                symbol = data['s']
                trade = {
                    'price': float(data['p']),
                    'quantity': float(data['q']),
                    'time': data['T'],
                    'is_buyer_maker': data['m'],
                    'q': data['q'],
                    'm': data['m']
                }
                
                self.trade_history[symbol].append(trade)
                self.current_prices[symbol] = trade['price']
                self.calculate_cvd(symbol, trade)
                
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.running:
            print("Reconnecting...")
            time.sleep(5)
            self.start()
    
    def on_open(self, ws):
        print("WebSocket connection opened")
        # Subscribe to trade streams
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@trade" for symbol in self.symbols],
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))
    
    def start(self):
        """Start websocket connection in background thread"""
        self.running = True
        
        # Binance WebSocket URL
        ws_url = "wss://stream.binance.com:9443/ws"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop websocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
    
    def get_cvd(self, symbol: str, window_seconds: int = 300) -> float:
        """Get CVD change over time window (default 5 min)"""
        if symbol not in self.trade_history:
            return 0.0
        
        current_time = time.time() * 1000  # Binance uses milliseconds
        window_start = current_time - (window_seconds * 1000)
        
        # Get trades within window
        trades_in_window = [
            t for t in self.trade_history[symbol]
            if t['time'] >= window_start
        ]
        
        if not trades_in_window:
            return 0.0
        
        # Calculate CVD for window
        cvd_window = 0.0
        for trade in trades_in_window:
            qty = float(trade['quantity'])
            if trade['is_buyer_maker']:
                cvd_window -= qty
            else:
                cvd_window += qty
        
        return cvd_window
    
    def get_volume_imbalance(self, symbol: str, window_seconds: int = 300) -> float:
        """Calculate buy/sell volume ratio"""
        if symbol not in self.trade_history:
            return 0.0
        
        current_time = time.time() * 1000
        window_start = current_time - (window_seconds * 1000)
        
        trades_in_window = [
            t for t in self.trade_history[symbol]
            if t['time'] >= window_start
        ]
        
        if not trades_in_window:
            return 0.0
        
        buy_volume = sum(
            float(t['quantity']) for t in trades_in_window
            if not t['is_buyer_maker']
        )
        sell_volume = sum(
            float(t['quantity']) for t in trades_in_window
            if t['is_buyer_maker']
        )
        
        if sell_volume == 0:
            return 1.0 if buy_volume > 0 else 0.0
        
        return buy_volume / sell_volume

class PolymarketDataFetcher:
    """Fetch Polymarket odds and volume"""
    
    def __init__(self):
        self.gamma_api = "https://gamma-api.polymarket.com"
        self.clob_api = "https://clob.polymarket.com"
        self.active_markets = {}
    
    def find_15min_markets(self, symbols: List[str] = ["Bitcoin", "Ethereum", "Solana", "XRP"]) -> Dict:
        """Find active 15-minute markets for given symbols"""
        try:
            response = requests.get(
                f"{self.gamma_api}/markets",
                params={"active": "true", "closed": "false", "limit": 100}
            )
            response.raise_for_status()
            markets = response.json()
            
            # Filter for 15-min markets
            active_15min = {}
            for market in markets:
                question = market.get('question', '').lower()
                if '15 minute' in question:
                    for symbol in symbols:
                        if symbol.lower() in question:
                            active_15min[symbol] = market
                            break
            
            return active_15min
        except Exception as e:
            print(f"Error fetching Polymarket markets: {e}")
            return {}
    
    def get_market_odds(self, market_id: str) -> Dict:
        """Get current odds for a market"""
        try:
            # Get market details
            response = requests.get(f"{self.gamma_api}/markets/{market_id}")
            market = response.json()
            
            up_token = market['tokens'][0]['token_id']
            
            # Get orderbook
            response = requests.get(
                f"{self.clob_api}/book",
                params={"token_id": up_token}
            )
            book = response.json()
            
            best_bid = float(book['bids'][0]['price']) if book.get('bids') else 0
            best_ask = float(book['asks'][0]['price']) if book.get('asks') else 1
            mid = (best_bid + best_ask) / 2
            
            # Get volume
            volume = market.get('volume', 0)
            
            return {
                'up_odds': mid,
                'down_odds': 1 - mid,
                'volume': volume,
                'spread': best_ask - best_bid
            }
        except Exception as e:
            print(f"Error fetching market odds: {e}")
            return None

class DataAggregator:
    """Main aggregator that combines all data sources"""
    
    def __init__(self):
        self.symbols = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT',
            'SOL': 'SOLUSDT',
            'XRP': 'XRPUSDT'
        }
        
        # Use user's advanced liquidation estimator (70-80% accuracy!)
        self.liquidation_estimator = LiquidationEstimator()
        self.binance_futures = BinanceFuturesData()  # For direct API calls
        self.binance_ws = BinanceWebSocket(list(self.symbols.values()))
        self.polymarket = PolymarketDataFetcher()
        
        # Historical data storage
        self.snapshots = {symbol: deque(maxlen=1000) for symbol in self.symbols.keys()}
        
    def start(self):
        """Start all data streams"""
        print("Starting data aggregator...")
        self.binance_ws.start()
        print("Binance WebSocket connected")
        time.sleep(2)  # Allow time for connection
    
    def stop(self):
        """Stop all data streams"""
        self.binance_ws.stop()
    
    def get_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Get current market snapshot for a symbol"""
        binance_symbol = self.symbols.get(symbol)
        if not binance_symbol:
            return None
        
        try:
            # Get current price from Binance
            price = self.binance_ws.current_prices.get(binance_symbol, 0)
            
            if price == 0:
                # Fallback to REST API
                response = requests.get(
                    f"https://api.binance.com/api/v3/ticker/24hr",
                    params={"symbol": binance_symbol}
                )
                data = response.json()
                price = float(data['lastPrice'])
                volume_24h = float(data['volume'])
            else:
                # Get 24h volume
                response = requests.get(
                    f"https://api.binance.com/api/v3/ticker/24hr",
                    params={"symbol": binance_symbol}
                )
                volume_24h = float(response.json()['volume'])
            
            # Get CVD
            cvd = self.binance_ws.get_cvd(binance_symbol, window_seconds=300)
            volume_imbalance = self.binance_ws.get_volume_imbalance(binance_symbol, window_seconds=300)
            
            # Get Binance Futures data (FREE)
            oi_data = self.binance_futures.get_open_interest(binance_symbol)
            funding_data = self.binance_futures.get_funding_rate(binance_symbol)
            ls_ratio_data = self.binance_futures.get_long_short_ratio(binance_symbol)
            
            # Use advanced liquidation estimator (70-80% accuracy!)
            # This combines 4 methods: leverage-based, volume profile, S/R, funding rate
            liquidation_clusters = self.liquidation_estimator.estimate_clusters(binance_symbol)
            
            # Extract top clusters
            cluster_above = liquidation_clusters['above'][0] if liquidation_clusters['above'] else None
            cluster_below = liquidation_clusters['below'][0] if liquidation_clusters['below'] else None
            
            # Create snapshot
            snapshot = MarketSnapshot(
                timestamp=time.time(),
                symbol=symbol,
                price=price,
                volume_24h=volume_24h,
                cvd=cvd,
                open_interest=oi_data['open_interest_usd'],
                open_interest_change_pct=0,  # Would need historical tracking
                funding_rate=funding_data['funding_rate'] * 100,  # Convert to percentage
                long_short_ratio=ls_ratio_data['long_short_ratio'],
                liquidation_cluster_above=cluster_above.price if cluster_above else None,
                liquidation_cluster_below=cluster_below.price if cluster_below else None,
                liquidation_strength_above=cluster_above.strength_usd if cluster_above else 0,
                liquidation_strength_below=cluster_below.strength_usd if cluster_below else 0,
                volume_delta=cvd,
                volume_imbalance=volume_imbalance
            )
            
            # Store snapshot
            self.snapshots[symbol].append(snapshot)
            
            return snapshot
            
        except Exception as e:
            print(f"Error creating snapshot for {symbol}: {e}")
            return None
    
    def get_polymarket_odds(self, symbol: str) -> Optional[Dict]:
        """Get Polymarket odds for symbol's 15-min market"""
        markets = self.polymarket.find_15min_markets([symbol])
        if symbol in markets:
            return self.polymarket.get_market_odds(markets[symbol]['id'])
        return None
    
    def get_historical_snapshots(self, symbol: str, count: int = 10) -> List[MarketSnapshot]:
        """Get last N snapshots for a symbol"""
        if symbol not in self.snapshots:
            return []
        return list(self.snapshots[symbol])[-count:]
    
    def export_to_dataframe(self, symbol: str) -> pd.DataFrame:
        """Export snapshot history to pandas DataFrame"""
        if symbol not in self.snapshots or not self.snapshots[symbol]:
            return pd.DataFrame()
        
        data = [asdict(snapshot) for snapshot in self.snapshots[symbol]]
        return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    print("Polymarket Quantum Predictor - Data Aggregator")
    print("=" * 70)
    
    # Initialize with free APIs (no API keys needed!)
    aggregator = DataAggregator()
    
    # Start data collection
    aggregator.start()
    
    print("\nCollecting market data...")
    print("Waiting for data to populate (10 seconds)...")
    time.sleep(10)
    
    # Get snapshots for all symbols
    print("\n" + "=" * 70)
    print("CURRENT MARKET SNAPSHOTS")
    print("=" * 70)
    
    for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
        print(f"\n{symbol}:")
        snapshot = aggregator.get_snapshot(symbol)
        
        if snapshot:
            print(f"  Price: ${snapshot.price:,.2f}")
            print(f"  CVD (5min): {snapshot.cvd:,.0f}")
            print(f"  Open Interest: ${snapshot.open_interest:,.0f}")
            print(f"  OI Change (1h): {snapshot.open_interest_change_pct:+.2f}%")
            print(f"  Funding Rate: {snapshot.funding_rate:.4f}%")
            print(f"  Long/Short Ratio: {snapshot.long_short_ratio:.2f}")
            print(f"  Volume Imbalance: {snapshot.volume_imbalance:.2f}")
            
            if snapshot.liquidation_cluster_above:
                print(f"  Liquidation Cluster Above: ${snapshot.liquidation_cluster_above:,.2f}")
            if snapshot.liquidation_cluster_below:
                print(f"  Liquidation Cluster Below: ${snapshot.liquidation_cluster_below:,.2f}")
        else:
            print("  [No data available]")
    
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(60)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Data collection running...")
    except KeyboardInterrupt:
        print("\n\nStopping aggregator...")
        aggregator.stop()
        print("Stopped.")
