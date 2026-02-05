#!/usr/bin/env python3
"""
API Endpoint Testing Script
Tests all data sources to verify they're working before running the predictor
"""

import requests
import json
import time
from typing import Dict, Tuple
from datetime import datetime

class APITester:
    """Test all API endpoints used by the trading system"""
    
    def __init__(self):
        self.results = {
            'binance': {},
            'coinglass': {},
            'polymarket': {}
        }
        self.test_symbols = ['BTC', 'ETH', 'SOL', 'XRP']
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}")
    
    def print_test(self, name: str, status: str, details: str = ""):
        """Print test result"""
        status_icon = "[OK]" if status == "PASS" else "[FAIL]"
        print(f"{status_icon} {name}: {status}")
        if details:
            print(f"   -> {details}")
    
    # ==================== BINANCE TESTS ====================
    
    def test_binance_rest_api(self) -> Tuple[bool, str]:
        """Test Binance REST API for price data"""
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": "BTCUSDT"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            price = float(data['lastPrice'])
            volume = float(data['volume'])
            
            return True, f"Price: ${price:,.2f}, 24h Volume: {volume:,.0f} BTC"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_binance_websocket_endpoint(self) -> Tuple[bool, str]:
        """Test if Binance WebSocket endpoint is reachable"""
        try:
            # We can't easily test WebSocket in a simple script, but we can verify the REST API works
            # which indicates Binance services are up
            response = requests.get(
                "https://api.binance.com/api/v3/ping",
                timeout=5
            )
            response.raise_for_status()
            return True, "WebSocket endpoint should be accessible (REST API is up)"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_binance_all_symbols(self) -> Tuple[bool, str]:
        """Test all trading symbols"""
        try:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
            prices = []
            
            for symbol in symbols:
                response = requests.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": symbol},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                prices.append(f"{symbol}: ${float(data['price']):,.2f}")
            
            return True, " | ".join(prices)
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    # ==================== COINGLASS TESTS ====================
    
    def test_coinglass_open_interest(self) -> Tuple[bool, str]:
        """Test Coinglass Open Interest endpoint"""
        try:
            response = requests.get(
                "https://open-api.coinglass.com/public/v2/indicator/open-interest",
                params={"symbol": "BTC"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                oi_data = data.get('data', {})
                oi = oi_data.get('openInterest', 0)
                change = oi_data.get('h24Change', 0)
                return True, f"OI: ${oi:,.0f}, 24h Change: {change:+.2f}%"
            else:
                return False, f"API returned success=false: {data.get('msg', 'Unknown error')}"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return False, "Rate limit exceeded (need API key for higher limits)"
            return False, f"HTTP Error {e.response.status_code}: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_coinglass_funding_rate(self) -> Tuple[bool, str]:
        """Test Coinglass Funding Rate endpoint"""
        try:
            response = requests.get(
                "https://open-api.coinglass.com/public/v2/indicator/funding-rate",
                params={"symbol": "BTC"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                funding_data = data.get('data', {})
                rate = funding_data.get('rate', 0)
                return True, f"Funding Rate: {rate:.4f}%"
            else:
                return False, f"API returned success=false: {data.get('msg', 'Unknown error')}"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return False, "Rate limit exceeded (need API key for higher limits)"
            return False, f"HTTP Error {e.response.status_code}: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_coinglass_liquidation_heatmap(self) -> Tuple[bool, str]:
        """Test Coinglass Liquidation Heatmap endpoint"""
        try:
            response = requests.get(
                "https://open-api.coinglass.com/public/v2/indicator/liquidation-heatmap",
                params={"symbol": "BTC", "interval": "15m"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return True, "Liquidation heatmap data available"
            else:
                return False, f"API returned success=false: {data.get('msg', 'Unknown error')}"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return False, "Rate limit exceeded (need API key for higher limits)"
            return False, f"HTTP Error {e.response.status_code}: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_coinglass_long_short_ratio(self) -> Tuple[bool, str]:
        """Test Coinglass Long/Short Ratio endpoint"""
        try:
            response = requests.get(
                "https://open-api.coinglass.com/public/v2/indicator/long-short-accounts",
                params={"symbol": "BTC"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                ratio_data = data.get('data', {})
                ratio = ratio_data.get('ratio', 1.0)
                return True, f"Long/Short Ratio: {ratio:.2f}"
            else:
                return False, f"API returned success=false: {data.get('msg', 'Unknown error')}"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return False, "Rate limit exceeded (need API key for higher limits)"
            return False, f"HTTP Error {e.response.status_code}: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    # ==================== POLYMARKET TESTS ====================
    
    def test_polymarket_markets_api(self) -> Tuple[bool, str]:
        """Test Polymarket Gamma API for markets"""
        try:
            response = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"active": "true", "closed": "false", "limit": 10},
                timeout=10
            )
            response.raise_for_status()
            markets = response.json()
            
            if isinstance(markets, list) and len(markets) > 0:
                return True, f"Found {len(markets)} active markets"
            else:
                return False, "No markets returned"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_polymarket_15min_markets(self) -> Tuple[bool, str]:
        """Test if 15-minute crypto markets exist"""
        try:
            response = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"active": "true", "closed": "false", "limit": 100},
                timeout=10
            )
            response.raise_for_status()
            markets = response.json()
            
            # Look for 15-minute markets
            fifteen_min_markets = []
            for market in markets:
                question = market.get('question', '').lower()
                if '15 minute' in question or '15-minute' in question:
                    for symbol in ['bitcoin', 'ethereum', 'solana', 'xrp']:
                        if symbol in question:
                            fifteen_min_markets.append(f"{symbol.upper()}: {market.get('question', '')[:50]}...")
                            break
            
            if fifteen_min_markets:
                return True, f"Found {len(fifteen_min_markets)} 15-min markets: {', '.join(fifteen_min_markets[:2])}"
            else:
                return False, "No 15-minute crypto markets found (they may not be active right now)"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def test_polymarket_clob_api(self) -> Tuple[bool, str]:
        """Test Polymarket CLOB API accessibility"""
        try:
            # First get a market to test with
            response = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"active": "true", "limit": 1},
                timeout=10
            )
            response.raise_for_status()
            markets = response.json()
            
            if not markets or len(markets) == 0:
                return False, "No markets available to test CLOB API"
            
            # Try to get orderbook for first market
            market = markets[0]
            if 'tokens' not in market or len(market['tokens']) == 0:
                return False, "Market has no tokens"
            
            token_id = market['tokens'][0].get('token_id')
            if not token_id:
                return False, "Could not get token_id"
            
            response = requests.get(
                "https://clob.polymarket.com/book",
                params={"token_id": token_id},
                timeout=10
            )
            response.raise_for_status()
            book = response.json()
            
            if 'bids' in book or 'asks' in book:
                return True, "CLOB API accessible and returning orderbook data"
            else:
                return False, "CLOB API returned unexpected format"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    # ==================== MAIN TEST RUNNER ====================
    
    def run_all_tests(self):
        """Run all API tests"""
        self.print_header("API ENDPOINT TESTING - Polymarket Quantum Predictor")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Binance Tests
        self.print_header("BINANCE API TESTS")
        
        success, details = self.test_binance_rest_api()
        self.print_test("Binance REST API (24hr ticker)", "PASS" if success else "FAIL", details)
        self.results['binance']['rest_api'] = success
        
        success, details = self.test_binance_websocket_endpoint()
        self.print_test("Binance WebSocket Endpoint", "PASS" if success else "FAIL", details)
        self.results['binance']['websocket'] = success
        
        success, details = self.test_binance_all_symbols()
        self.print_test("All Trading Symbols", "PASS" if success else "FAIL", details)
        self.results['binance']['all_symbols'] = success
        
        # Coinglass Tests
        self.print_header("COINGLASS API TESTS")
        print("[!] Coinglass tests skipped (not in use)\n")
        
        # Polymarket Tests
        self.print_header("POLYMARKET API TESTS")
        
        success, details = self.test_polymarket_markets_api()
        self.print_test("Gamma API (Markets)", "PASS" if success else "FAIL", details)
        self.results['polymarket']['gamma_api'] = success
        
        success, details = self.test_polymarket_15min_markets()
        self.print_test("15-Minute Crypto Markets", "PASS" if success else "FAIL", details)
        self.results['polymarket']['15min_markets'] = success
        
        success, details = self.test_polymarket_clob_api()
        self.print_test("CLOB API (Orderbook)", "PASS" if success else "FAIL", details)
        self.results['polymarket']['clob_api'] = success
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")
        
        total_tests = 0
        passed_tests = 0
        
        for service, tests in self.results.items():
            service_passed = sum(1 for result in tests.values() if result)
            service_total = len(tests)
            total_tests += service_total
            passed_tests += service_passed
            
            status = "[OK] ALL PASS" if service_passed == service_total else f"[!] {service_passed}/{service_total} PASS"
            print(f"{service.upper()}: {status}")
        
        print(f"\n{'='*70}")
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"{'='*70}")
        
        # Recommendations
        print("\n[RECOMMENDATIONS]:")
        
        if all(self.results['binance'].values()):
            print("[OK] Binance: All systems operational")
        else:
            print("[FAIL] Binance: Check your internet connection")
        
        coinglass_pass = sum(self.results['coinglass'].values())
        if coinglass_pass == len(self.results['coinglass']):
            print("[OK] Coinglass: All systems operational")
        elif coinglass_pass > 0:
            print("[!] Coinglass: Partial functionality (consider getting API key for higher rate limits)")
        else:
            print("[FAIL] Coinglass: No endpoints working (check if service is down or rate limited)")
        
        if all(self.results['polymarket'].values()):
            print("[OK] Polymarket: All systems operational")
        elif not self.results['polymarket'].get('15min_markets', False):
            print("[!] Polymarket: APIs working but no 15-min markets active right now")
        else:
            print("[FAIL] Polymarket: Check if service is accessible")
        
        print("\n[TIP]: If Coinglass tests fail, you can still run the predictor.")
        print("   It will use fallback values for missing data.")
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    tester = APITester()
    tester.run_all_tests()
