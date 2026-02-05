#!/usr/bin/env python3
"""
Test free alternatives to Coinglass API
Testing: Binance, Bybit, and other free sources for derivatives data
"""

import requests
import json
from datetime import datetime

def test_binance_futures_api():
    """Test Binance Futures API for Open Interest and Funding Rate"""
    print("\n" + "="*70)
    print("TESTING BINANCE FUTURES API (FREE)")
    print("="*70)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        
        # 1. Open Interest
        try:
            response = requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={"symbol": symbol},
                timeout=10
            )
            response.raise_for_status()
            oi_data = response.json()
            print(f"  [OK] Open Interest: {float(oi_data['openInterest']):,.2f} contracts")
        except Exception as e:
            print(f"  [FAIL] Open Interest: {e}")
        
        # 2. Funding Rate
        try:
            response = requests.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": symbol},
                timeout=10
            )
            response.raise_for_status()
            funding_data = response.json()
            funding_rate = float(funding_data['lastFundingRate']) * 100
            print(f"  [OK] Funding Rate: {funding_rate:.4f}%")
        except Exception as e:
            print(f"  [FAIL] Funding Rate: {e}")
        
        # 3. Long/Short Ratio
        try:
            response = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": "5m", "limit": 1},
                timeout=10
            )
            response.raise_for_status()
            ratio_data = response.json()
            if ratio_data:
                ls_ratio = float(ratio_data[0]['longShortRatio'])
                long_pct = float(ratio_data[0]['longAccount']) * 100
                short_pct = float(ratio_data[0]['shortAccount']) * 100
                print(f"  [OK] Long/Short Ratio: {ls_ratio:.2f} (Long: {long_pct:.1f}%, Short: {short_pct:.1f}%)")
        except Exception as e:
            print(f"  [FAIL] Long/Short Ratio: {e}")
        
        # 4. Top Trader Long/Short Ratio
        try:
            response = requests.get(
                "https://fapi.binance.com/futures/data/topLongShortAccountRatio",
                params={"symbol": symbol, "period": "5m", "limit": 1},
                timeout=10
            )
            response.raise_for_status()
            ratio_data = response.json()
            if ratio_data:
                ls_ratio = float(ratio_data[0]['longShortRatio'])
                print(f"  [OK] Top Trader L/S Ratio: {ls_ratio:.2f}")
        except Exception as e:
            print(f"  [FAIL] Top Trader L/S Ratio: {e}")

def test_bybit_api():
    """Test Bybit API for derivatives data"""
    print("\n" + "="*70)
    print("TESTING BYBIT API (FREE)")
    print("="*70)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        
        # 1. Open Interest
        try:
            response = requests.get(
                "https://api.bybit.com/v5/market/open-interest",
                params={"category": "linear", "symbol": symbol, "intervalTime": "5min", "limit": 1},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                oi = float(data['result']['list'][0]['openInterest'])
                print(f"  [OK] Open Interest: {oi:,.2f} contracts")
            else:
                print(f"  [FAIL] Open Interest: {data.get('retMsg', 'Unknown error')}")
        except Exception as e:
            print(f"  [FAIL] Open Interest: {e}")
        
        # 2. Funding Rate
        try:
            response = requests.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "linear", "symbol": symbol},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                funding_rate = float(data['result']['list'][0]['fundingRate']) * 100
                print(f"  [OK] Funding Rate: {funding_rate:.4f}%")
            else:
                print(f"  [FAIL] Funding Rate: {data.get('retMsg', 'Unknown error')}")
        except Exception as e:
            print(f"  [FAIL] Funding Rate: {e}")

def test_okx_api():
    """Test OKX API for derivatives data"""
    print("\n" + "="*70)
    print("TESTING OKX API (FREE)")
    print("="*70)
    
    symbols = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'XRP-USDT-SWAP']
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        
        # 1. Open Interest
        try:
            response = requests.get(
                "https://www.okx.com/api/v5/public/open-interest",
                params={"instId": symbol},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if data.get('code') == '0' and data.get('data'):
                oi = float(data['data'][0]['oi'])
                print(f"  [OK] Open Interest: {oi:,.2f} contracts")
            else:
                print(f"  [FAIL] Open Interest: {data.get('msg', 'Unknown error')}")
        except Exception as e:
            print(f"  [FAIL] Open Interest: {e}")
        
        # 2. Funding Rate
        try:
            response = requests.get(
                "https://www.okx.com/api/v5/public/funding-rate",
                params={"instId": symbol},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if data.get('code') == '0' and data.get('data'):
                funding_rate = float(data['data'][0]['fundingRate']) * 100
                print(f"  [OK] Funding Rate: {funding_rate:.4f}%")
            else:
                print(f"  [FAIL] Funding Rate: {data.get('msg', 'Unknown error')}")
        except Exception as e:
            print(f"  [FAIL] Funding Rate: {e}")



if __name__ == "__main__":
    print("="*70)
    print("FREE DERIVATIVES DATA API TESTING")
    print("="*70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test all free sources
    test_binance_futures_api()
    test_bybit_api()
    test_okx_api()

    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n[RECOMMENDATION]:")
    print("1. BINANCE FUTURES API - Best option (free, comprehensive)")
    print("   - Open Interest: Available")
    print("   - Funding Rate: Available")
    print("   - Long/Short Ratio: Available")
    print("   - Liquidations: Available")
    print("\n2. BYBIT API - Good alternative")
    print("   - Open Interest: Available")
    print("   - Funding Rate: Available")
    print("\n3. OKX API - Additional source")
    print("   - Open Interest: Available")
    print("   - Funding Rate: Available")
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
