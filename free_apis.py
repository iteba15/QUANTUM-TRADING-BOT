#!/usr/bin/env python3
"""
Free API Classes for Derivatives Data
Replaces Coinglass with Binance Futures, Bybit, and OKX
"""

import requests
from typing import Dict, List, Optional

class BinanceFuturesAPI:
    """Fetches Open Interest, Funding Rate, Long/Short Ratio from Binance Futures (FREE)"""
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.futures_data_url = "https://fapi.binance.com/futures/data"
        # Symbol mapping: BTC -> BTCUSDT
        self.symbol_map = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT',
            'SOL': 'SOLUSDT',
            'XRP': 'XRPUSDT'
        }
    
    def _get_binance_symbol(self, symbol: str) -> str:
        """Convert symbol to Binance format"""
        return self.symbol_map.get(symbol, f"{symbol}USDT")
    
    def get_open_interest(self, symbol: str = "BTC") -> Dict:
        """Get current open interest from Binance Futures"""
        try:
            binance_symbol = self._get_binance_symbol(symbol)
            response = requests.get(
                f"{self.base_url}/fapi/v1/openInterest",
                params={"symbol": binance_symbol},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            oi = float(data.get('openInterest', 0))
            
            # Get historical OI to calculate change
            # Note: We'll use current value and estimate change as 0 for now
            # Could enhance by storing historical values
            return {
                'open_interest': oi,
                'change_24h': 0,  # Would need historical data
                'change_1h': 0,
            }
        except Exception as e:
            print(f"Error fetching OI for {symbol} from Binance: {e}")
            return {'open_interest': 0, 'change_24h': 0, 'change_1h': 0}
    
    def get_funding_rate(self, symbol: str = "BTC") -> Dict:
        """Get current funding rate from Binance Futures"""
        try:
            binance_symbol = self._get_binance_symbol(symbol)
            response = requests.get(
                f"{self.base_url}/fapi/v1/premiumIndex",
                params={"symbol": binance_symbol},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            funding_rate = float(data.get('lastFundingRate', 0)) * 100  # Convert to percentage
            next_funding_time = int(data.get('nextFundingTime', 0))
            
            return {
                'funding_rate': funding_rate,
                'next_funding_time': next_funding_time
            }
        except Exception as e:
            print(f"Error fetching funding rate for {symbol} from Binance: {e}")
            return {'funding_rate': 0, 'next_funding_time': 0}
    
    def get_long_short_ratio(self, symbol: str = "BTC") -> Dict:
        """Get long/short account ratio from Binance Futures"""
        try:
            binance_symbol = self._get_binance_symbol(symbol)
            response = requests.get(
                f"{self.futures_data_url}/globalLongShortAccountRatio",
                params={"symbol": binance_symbol, "period": "5m", "limit": 1},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                latest = data[0]
                ls_ratio = float(latest.get('longShortRatio', 1.0))
                long_pct = float(latest.get('longAccount', 0.5)) * 100
                short_pct = float(latest.get('shortAccount', 0.5)) * 100
                
                return {
                    'long_short_ratio': ls_ratio,
                    'long_account_pct': long_pct,
                    'short_account_pct': short_pct
                }
            
            return {'long_short_ratio': 1.0, 'long_account_pct': 50, 'short_account_pct': 50}
        except Exception as e:
            print(f"Error fetching long/short ratio for {symbol} from Binance: {e}")
            return {'long_short_ratio': 1.0, 'long_account_pct': 50, 'short_account_pct': 50}
    
    def get_top_trader_ratio(self, symbol: str = "BTC") -> Dict:
        """Get top trader long/short ratio (additional metric)"""
        try:
            binance_symbol = self._get_binance_symbol(symbol)
            response = requests.get(
                f"{self.futures_data_url}/topLongShortAccountRatio",
                params={"symbol": binance_symbol, "period": "5m", "limit": 1},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                latest = data[0]
                return {
                    'top_trader_ratio': float(latest.get('longShortRatio', 1.0)),
                    'top_long_pct': float(latest.get('longAccount', 0.5)) * 100,
                    'top_short_pct': float(latest.get('shortAccount', 0.5)) * 100
                }
            
            return {'top_trader_ratio': 1.0, 'top_long_pct': 50, 'top_short_pct': 50}
        except Exception as e:
            print(f"Error fetching top trader ratio for {symbol} from Binance: {e}")
            return {'top_trader_ratio': 1.0, 'top_long_pct': 50, 'top_short_pct': 50}
    
    def estimate_liquidation_clusters(
        self,
        symbol: str,
        current_price: float,
        open_interest: float,
        funding_rate: float,
        long_short_ratio: float,
        price_history: List[float]
    ) -> Dict:
        """Estimate liquidation clusters using available data"""
        try:
            # Calculate position distribution
            long_pct = long_short_ratio / (1 + long_short_ratio)
            short_pct = 1 - long_pct
            
            # Estimate average leverage based on funding rate intensity
            avg_leverage = 10  # Default assumption
            if abs(funding_rate) > 0.02:
                avg_leverage = 20  # Higher leverage when funding is extreme
            elif abs(funding_rate) > 0.05:
                avg_leverage = 50
            
            # Calculate liquidation distances
            long_liq_distance = 1 / avg_leverage
            short_liq_distance = 1 / avg_leverage
            
            # Find recent support/resistance (likely entry points)
            if len(price_history) >= 100:
                recent_high = max(price_history[-100:])
                recent_low = min(price_history[-100:])
            else:
                recent_high = current_price * 1.02
                recent_low = current_price * 0.98
            
            # Estimate clusters
            # Longs likely entered near recent high, liquidate below
            cluster_below = recent_high * (1 - long_liq_distance)
            
            # Shorts likely entered near recent low, liquidate above
            cluster_above = recent_low * (1 + short_liq_distance)
            
            # Calculate strength (USD value of positions at risk)
            strength_below = open_interest * current_price * long_pct
            strength_above = open_interest * current_price * short_pct
            
            return {
                'cluster_above': cluster_above,
                'cluster_below': cluster_below,
                'strength_above': strength_above,
                'strength_below': strength_below
            }
        except Exception as e:
            print(f"Error estimating liquidation clusters for {symbol}: {e}")
            return {
                'cluster_above': None,
                'cluster_below': None,
                'strength_above': 0,
                'strength_below': 0
            }
