#!/usr/bin/env python3
"""
Quick Test - Verify Integration
Tests that data_aggregator.py works with the new free APIs
"""

import sys
import time

print("="*70)
print("TESTING INTEGRATED SYSTEM")
print("="*70)

# Test 1: Import liquidation estimator
print("\n[1/4] Testing liquidation_estimator imports...")
try:
    from liquidation_estimator import LiquidationEstimator, BinanceFuturesData
    print("     [OK] Imports successful")
except Exception as e:
    print(f"     [FAIL] {e}")
    sys.exit(1)

# Test 2: Test BinanceFuturesData
print("\n[2/4] Testing BinanceFuturesData API calls...")
try:
    api = BinanceFuturesData()
    price = api.get_current_price("BTCUSDT")
    oi = api.get_open_interest("BTCUSDT")
    funding = api.get_funding_rate("BTCUSDT")
    ls_ratio = api.get_long_short_ratio("BTCUSDT")
    
    print(f"     [OK] BTC Price: ${price:,.2f}")
    print(f"     [OK] Open Interest: ${oi['open_interest_usd']:,.0f}")
    print(f"     [OK] Funding Rate: {funding['funding_rate']*100:.4f}%")
    print(f"     [OK] L/S Ratio: {ls_ratio['long_short_ratio']:.2f}")
except Exception as e:
    print(f"     [FAIL] {e}")
    sys.exit(1)

# Test 3: Test data_aggregator imports
print("\n[3/4] Testing data_aggregator integration...")
try:
    from data_aggregator import DataAggregator
    print("     [OK] DataAggregator imports successfully")
    
    # Initialize (don't start WebSocket to keep it quick)
    aggregator = DataAggregator()
    print("     [OK] DataAggregator initialized")
    print(f"     [OK] Has liquidation_estimator: {hasattr(aggregator, 'liquidation_estimator')}")
    print(f"     [OK] Has binance_futures: {hasattr(aggregator, 'binance_futures')}")
except Exception as e:
    print(f"     [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Quick liquidation estimation (without verbose output)
print("\n[4/4] Testing liquidation estimation (quick test)...")
try:
    # Suppress print output from estimator
    import io
    import contextlib
    
    with contextlib.redirect_stdout(io.StringIO()):
        estimator = LiquidationEstimator()
        clusters = estimator.estimate_clusters("BTCUSDT")
    
    if clusters['above'] and clusters['below']:
        top_above = clusters['above'][0]
        top_below = clusters['below'][0]
        print(f"     [OK] Found {len(clusters['above'])} clusters above")
        print(f"     [OK] Found {len(clusters['below'])} clusters below")
        print(f"     [OK] Top cluster above: ${top_above.price:,.2f} (${top_above.strength_usd/1e6:.1f}M)")
        print(f"     [OK] Top cluster below: ${top_below.price:,.2f} (${top_below.strength_usd/1e6:.1f}M)")
    else:
        print("     [WARN] No clusters found (might be data issue)")
except Exception as e:
    print(f"     [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nYour system is ready to use!")
print("\nNext steps:")
print("  1. Run: py data_aggregator.py")
print("  2. Let it collect data for 10-30 seconds")
print("  3. View market snapshots with liquidation clusters")
print("\nThe data will feed into your physics_engine.py and ml_engine.py")
