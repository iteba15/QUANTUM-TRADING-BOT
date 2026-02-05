# POLYMARKET QUANTUM PREDICTOR - DATA FLOW

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INPUT LAYER (FREE APIs)                     │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│   BINANCE FUTURES    │  │  BINANCE WEBSOCKET   │  │   POLYMARKET     │
│      (REST API)      │  │   (TICK-BY-TICK)     │  │   (REST API)     │
└──────────────────────┘  └──────────────────────┘  └──────────────────┘
         │                          │                        │
         ▼                          ▼                        ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│ • Open Interest      │  │ • Price (real-time)  │  │ • Market Odds    │
│ • Funding Rate       │  │ • Trade-by-trade     │  │ • Order Book     │
│ • Long/Short Ratio   │  │ • CVD calculation    │  │ • Volume         │
│ • Recent Liquidations│  │ • Volume imbalance   │  │ • Spread         │
│ • Price History      │  │ • Buy/Sell pressure  │  │                  │
└──────────────────────┘  └──────────────────────┘  └──────────────────┘

                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 DATA AGGREGATOR (data_aggregator.py)                │
│                                                                       │
│  Combines all data sources into unified MarketSnapshot objects      │
│                                                                       │
│  MarketSnapshot = {                                                  │
│    timestamp, symbol, price,                                         │
│    cvd, open_interest, oi_change_pct,                               │
│    funding_rate, long_short_ratio,                                   │
│    liquidation_clusters (estimated),                                 │
│    volume_delta, volume_imbalance,                                   │
│    polymarket_odds (optional)                                        │
│  }                                                                   │
│                                                                       │
│  Storage: Deque of last 1000 snapshots per symbol                   │
│  Update Rate: Every 30 seconds (configurable)                       │
└─────────────────────────────────────────────────────────────────────┘

                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                    ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│ LIQUIDATION ESTIMATOR           │  │  REAL-TIME SNAPSHOT             │
│ (liquidation_estimator.py)      │  │                                 │
│                                  │  │  Current market state           │
│ Methods:                         │  │  + Historical context           │
│ 1. Leverage-based (60% weight)  │  │                                 │
│ 2. Volume profile (70% weight)  │  │  Fed to analysis engines →     │
│ 3. Support/Resistance (75%)     │  │                                 │
│ 4. Funding rate (80% weight)    │  │                                 │
│                                  │  │                                 │
│ Output: Top 5 clusters above/   │  │                                 │
│         below current price      │  │                                 │
│         with confidence scores   │  │                                 │
└─────────────────────────────────┘  └─────────────────────────────────┘

                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                    ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│    PHYSICS ENGINE (CPU)         │  │   ML ENGINE (GPU - CUDA)        │
│    (physics_engine.py)          │  │   (ml_engine.py)                │
│                                  │  │                                 │
│ Analyzes Market Structure:       │  │ Pattern Recognition:            │
│                                  │  │                                 │
│ 1. Kinetic Energy (30%)         │  │ • LSTM Network (3 layers)       │
│    → CVD divergence detection   │  │   128 hidden units              │
│    → Buy/sell pressure          │  │   Attention mechanism           │
│    Score: 0-100                  │  │                                 │
│                                  │  │ • Transformer (4 layers)        │
│ 2. Potential Energy (25%)       │  │   8 attention heads             │
│    → OI flow analysis           │  │   Positional encoding           │
│    → Real vs fake moves         │  │                                 │
│    → Trap detection             │  │ • Ensemble Voting               │
│    Score: 0-100                  │  │   50% LSTM + 50% Transformer   │
│                                  │  │                                 │
│ 3. Field Strength (25%)         │  │ Inference Time: <10ms           │
│    → Liquidation magnet pull    │  │ Batch Processing: <50ms/32     │
│    → Cluster proximity          │  │                                 │
│    Score: 0-100                  │  │ Output:                         │
│                                  │  │ • Probability UP: 0-1           │
│ 4. Friction (20%)               │  │ • Probability DOWN: 0-1         │
│    → Funding rate pressure      │  │ • Confidence: 0-1               │
│    → Crowding detection         │  │ • Model agreement score         │
│    Score: 0-100                  │  │                                 │
│                                  │  │                                 │
│ Output: PhysicsScore             │  │                                 │
│ • Total: 0-100                   │  │                                 │
│ • Direction: UP/DOWN/NEUTRAL    │  │                                 │
│ • Confidence: 0-1               │  │                                 │
│ • Regime: ACCUMULATION,         │  │                                 │
│   DISTRIBUTION, TRAP, etc.      │  │                                 │
│ • Signals & Warnings            │  │                                 │
│ • True probability              │  │                                 │
└─────────────────────────────────┘  └─────────────────────────────────┘

                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│           SIGNAL GENERATOR (quantum_predictor.py)                   │
│                                                                       │
│  Combines Physics + ML:                                              │
│  combined_probability = physics * 0.60 + ml * 0.40                  │
│                                                                       │
│  Calculates Edge:                                                    │
│  edge = combined_probability - polymarket_odds                       │
│                                                                       │
│  Decision Logic:                                                     │
│  IF edge > 12% AND confidence > 65%:                                │
│    → TRADE                                                           │
│  ELSE:                                                               │
│    → WAIT                                                            │
│                                                                       │
│  Position Sizing (Kelly Criterion):                                 │
│  kelly_fraction = edge / (1 - entry_odds)                           │
│  position = kelly * 0.25 * confidence * bankroll                    │
│  Capped at 40% of bankroll                                          │
│                                                                       │
│  Risk Assessment:                                                    │
│  • Warning count                                                     │
│  • Regime classification                                             │
│  • Funding extremes                                                  │
│  • Confidence level                                                  │
│  → Risk Score: 0-1 (0=safe, 1=risky)                               │
└─────────────────────────────────────────────────────────────────────┘

                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FINAL OUTPUT                                  │
│                                                                       │
│  TradingSignal = {                                                   │
│    action: 'LONG' / 'SHORT' / 'WAIT',                               │
│    symbol: 'BTC' / 'ETH' / 'SOL' / 'XRP',                          │
│    timeframe: '15min' / '1hour' / '4hour',                          │
│    confidence: 0.0 - 1.0,                                            │
│    edge: -1.0 to +1.0,                                              │
│                                                                       │
│    probabilities: {                                                  │
│      physics: 0.0 - 1.0,                                            │
│      ml: 0.0 - 1.0,                                                 │
│      combined: 0.0 - 1.0,                                           │
│      market (polymarket): 0.0 - 1.0                                 │
│    },                                                                │
│                                                                       │
│    position_sizing: {                                                │
│      recommended_pct: 0.20 - 0.40,                                  │
│      position_size_usd: calculated,                                  │
│      expected_roi: calculated,                                       │
│      entry_odds: from polymarket                                     │
│    },                                                                │
│                                                                       │
│    risk_metrics: {                                                   │
│      risk_score: 0.0 - 1.0,                                         │
│      regime: MarketRegime enum,                                      │
│      warnings: List[str],                                            │
│      signals: List[str]                                              │
│    },                                                                │
│                                                                       │
│    expected_outcomes: {                                              │
│      if_win: {payout, profit, roi},                                 │
│      if_lose: {loss_amount}                                          │
│    }                                                                 │
│  }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## DETAILED DATA SPECIFICATIONS

### INPUT DATA (What We Collect)

#### 1. Binance Futures REST API
```python
Open Interest:
  - open_interest (BTC amount)
  - open_interest_usd (USD value)
  - Update: Every 30s

Funding Rate:
  - current_funding_rate (%)
  - predicted_funding_rate (%)
  - next_funding_time
  - Update: Every 30s

Long/Short Ratio:
  - long_short_ratio (e.g., 2.36)
  - long_account_pct (e.g., 70%)
  - short_account_pct (e.g., 30%)
  - Update: Every 5min

Klines (Candlesticks):
  - OHLC prices
  - Volume
  - Quote volume
  - Intervals: 1m, 5m, 15m, 1h, 4h
  - Historical: Last 100-1000 candles

Recent Liquidations (if accessible):
  - Side (long/short)
  - Price
  - Quantity
  - Timestamp
```

#### 2. Binance WebSocket (Real-time)
```python
Trade Stream:
  - price (exact)
  - quantity
  - timestamp (milliseconds)
  - is_buyer_maker (true/false)
  
Aggregated Trades:
  - Buy volume (cumulative)
  - Sell volume (cumulative)
  - Update: Real-time (milliseconds)

Calculated Metrics:
  - CVD (Cumulative Volume Delta)
    = sum(buy_volume - sell_volume)
  - Volume Imbalance
    = buy_volume / sell_volume
  - Update: Continuous
```

#### 3. Polymarket API
```python
Market Data:
  - question (market description)
  - condition_id (unique ID)
  - tokens[0] = "Yes" token
  - tokens[1] = "No" token

Order Book:
  - bids[] (array of {price, size})
  - asks[] (array of {price, size})
  - best_bid, best_ask
  - mid_price = (bid + ask) / 2
  
Market Metrics:
  - volume (total traded)
  - liquidity
  - spread
  - Update: Every 30-60s
```

---

### INTERMEDIATE DATA (What We Calculate)

#### MarketSnapshot (Primary Data Structure)
```python
{
  # Identifiers
  timestamp: 1738694400.123,
  symbol: 'BTC',
  
  # Price Data
  price: 95234.56,
  volume_24h: 28_500_000_000,
  
  # Volume Analysis
  cvd: 145_234.5,  # Cumulative volume delta
  volume_delta: 1_234.5,  # Recent change
  volume_imbalance: 1.45,  # Buy/sell ratio
  
  # Derivatives Data
  open_interest: 45_230_000_000,  # USD
  open_interest_change_pct: 8.2,  # %
  funding_rate: 0.0085,  # 0.85%
  long_short_ratio: 2.36,  # 2.36:1
  
  # Liquidation Estimates (NEW)
  liquidation_cluster_above: 98_500,  # Price
  liquidation_cluster_below: 92_800,
  liquidation_strength_above: 850_000_000,  # USD
  liquidation_strength_below: 1_200_000_000,
  liquidation_confidence_above: 0.75,  # 75%
  liquidation_confidence_below: 0.80,
  
  # Polymarket (optional)
  polymarket_up_odds: 0.58,
  polymarket_down_odds: 0.42,
  polymarket_volume: 125_000
}
```

#### PhysicsScore (Physics Engine Output)
```python
{
  # Overall
  total_score: 72.5,  # 0-100
  direction: 'UP',  # UP/DOWN/NEUTRAL
  confidence: 0.78,  # 0-1
  regime: MarketRegime.ACCUMULATION,
  
  # Component Scores
  kinetic_energy_score: 68.0,  # CVD analysis
  potential_energy_score: 75.0,  # OI analysis
  field_strength_score: 72.0,  # Liquidation pull
  friction_score: 65.0,  # Funding analysis
  
  # Interpretation
  signals: [
    "🟢 Bullish CVD Divergence",
    "🟢 Aggressive longs: OI +8.2%",
    "✓ Healthy funding: 0.008%"
  ],
  warnings: [
    "⚠ Near resistance cluster"
  ],
  
  # Edge Calculation
  true_probability: 0.725,  # Our estimate
  market_probability: 0.58,  # Polymarket
  edge: 0.145  # +14.5%
}
```

#### PredictionResult (ML Engine Output)
```python
{
  # Predictions
  probability_up: 0.68,
  probability_down: 0.32,
  confidence: 0.82,
  
  # Model Breakdown
  model_scores: {
    'lstm': 0.71,
    'transformer': 0.65,
    'ensemble': 0.68
  },
  
  # Feature Analysis
  features_importance: {
    'price': 0.15,
    'cvd': 0.25,
    'open_interest': 0.20,
    'oi_change': 0.15,
    'funding_rate': 0.10,
    'ls_ratio': 0.08,
    'volume_imbalance': 0.07
  }
}
```

---

### OUTPUT DATA (What You Get)

#### TradingSignal (Final Output)
```python
{
  # Identity
  timestamp: 1738694400.123,
  symbol: 'BTC',
  timeframe: TimeWindow.MIN_15,
  
  # CORE DECISION
  action: 'LONG',  # LONG/SHORT/WAIT
  confidence: 0.78,  # 78%
  edge: 0.145,  # +14.5% vs market
  
  # PROBABILITIES
  physics_probability: 0.725,
  ml_probability: 0.68,
  combined_probability: 0.708,  # 60% physics + 40% ML
  market_probability: 0.58,  # Polymarket odds
  
  # POSITION SIZING
  recommended_position_pct: 0.36,  # 36% of bankroll
  # For $5 bankroll = $1.80 position
  expected_roi: 0.102,  # +10.2% expected return
  
  # RISK ASSESSMENT
  risk_score: 0.18,  # 18% = Low risk 🟢
  regime: MarketRegime.ACCUMULATION,
  
  # DETAILED ANALYSIS
  physics_score: PhysicsScore{...},  # Full physics output
  ml_prediction: PredictionResult{...},  # Full ML output
  
  # EXPECTED OUTCOMES
  if_win: {
    payout_usd: 3.36,
    profit_usd: 1.41,
    roi_pct: 78.3
  },
  if_lose: {
    loss_usd: 1.80,
    loss_pct: 36.0
  },
  
  # STOP LOSS / TAKE PROFIT (optional)
  stop_loss: 93_800,  # Near liquidation cluster
  take_profit: 97_200  # Near resistance
}
```

---

## EXAMPLES OF COMPLETE DATA FLOW

### Example 1: Bull Trap Detection

**INPUT:**
```
Binance: Price +0.25% (UP)
Binance: CVD -15% (SELLING volume)
Binance: OI -8% (Positions closing)
Binance: Funding +0.08% (Crowded longs)
Liquidation Est: $850M cluster at $92,800 (-3%)
Polymarket: 65¢ odds (crowd thinks UP)
```

**PROCESSING:**
```
Physics Engine:
  - Kinetic: 35/100 (CVD divergence)
  - Potential: 30/100 (OI dropping)
  - Friction: 25/100 (Funding extreme)
  → Total: 32/100 → TRUE PROB: 32%

ML Engine:
  - LSTM: 28% UP
  - Transformer: 35% UP
  → Ensemble: 31% UP

Combined: 32% * 0.6 + 31% * 0.4 = 31.6%
Edge: 31.6% - 65% = -33.4%
```

**OUTPUT:**
```python
TradingSignal(
  action='SHORT',  # Fade the crowd!
  confidence=0.85,  # High confidence
  edge=-0.334,  # HUGE mispricing
  
  # Bet DOWN at 35¢ odds
  recommended_position_pct=0.40,  # Max position
  expected_roi=0.85,  # 85% ROI if correct
  
  risk_score=0.15,  # Low risk (clear trap)
  regime=MarketRegime.TRAP_BULL
)
```

### Example 2: Accumulation Signal

**INPUT:**
```
Binance: Price -0.08% (flat/down)
Binance: CVD +22% (BUYING volume)
Binance: OI +12% (New positions)
Binance: Funding -0.01% (Not crowded)
Liquidation Est: $1.2B cluster at $98,500 (+3.5%)
Polymarket: 45¢ odds (crowd thinks DOWN)
```

**PROCESSING:**
```
Physics Engine:
  - Kinetic: 85/100 (Bullish divergence!)
  - Potential: 88/100 (OI rising)
  - Field: 75/100 (Pull toward $98.5k)
  - Friction: 65/100 (Neutral funding)
  → Total: 80/100 → TRUE PROB: 80%

ML Engine:
  - LSTM: 76% UP
  - Transformer: 82% UP
  → Ensemble: 79% UP

Combined: 80% * 0.6 + 79% * 0.4 = 79.6%
Edge: 79.6% - 45% = +34.6%
```

**OUTPUT:**
```python
TradingSignal(
  action='LONG',  # Strong buy!
  confidence=0.92,  # Very high
  edge=0.346,  # MASSIVE edge
  
  # Bet UP at 55¢ odds (buy the "No" side at 45¢)
  recommended_position_pct=0.40,  # Max position
  expected_roi=0.62,  # 62% ROI
  
  risk_score=0.08,  # Very low risk
  regime=MarketRegime.ACCUMULATION
)
```

---

## UPDATE FREQUENCIES

```
Real-time (milliseconds):
  - Binance trades
  - CVD calculation
  - Price updates

Every 30 seconds:
  - Market snapshots
  - Physics analysis
  - ML predictions
  - Signal generation

Every 60 seconds:
  - Full market scan
  - Multi-asset comparison
  - Top opportunities ranking

Every 5 minutes:
  - Long/Short ratio update
  - Liquidation re-estimation

Every 15 minutes:
  - ML model re-calibration (optional)
  - Historical data export
```

---

## PERFORMANCE METRICS

**Data Collection:**
- WebSocket latency: <50ms
- REST API calls: 200-500ms
- Snapshot creation: ~100ms

**Analysis:**
- Physics engine: ~500ms
- ML inference (GPU): ~10ms
- Combined signal: ~600ms total

**Full Scan (4 assets):**
- Sequential: ~2.4 seconds
- With threading: ~800ms

**Memory Usage:**
- Per asset: ~50MB (1000 snapshots)
- ML models: ~500MB (loaded in VRAM)
- Total: ~1.5GB RAM, ~2GB VRAM

---

## DATA QUALITY INDICATORS

Each data point includes confidence scores:

```python
liquidation_confidence: 0.70  # 70% confident in estimate
physics_confidence: 0.85  # 85% confident in analysis
ml_confidence: 0.78  # 78% model agreement
overall_confidence: 0.80  # Final signal confidence
```

Warnings automatically reduce confidence:
- Each warning: -10% confidence
- Critical warnings: -20% confidence
- Multiple warnings: Consider skipping trade

---

**This is a production-grade data pipeline designed for real money trading.**
