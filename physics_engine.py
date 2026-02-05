#!/usr/bin/env python3
"""
Physics Engine - Market Structure Analysis
Implements the "force-based" model for crypto prediction
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    """Market structure classification"""
    ACCUMULATION = "accumulation"  # Smart money buying
    MARKUP = "markup"  # Trending up
    DISTRIBUTION = "distribution"  # Smart money selling
    MARKDOWN = "markdown"  # Trending down
    RANGING = "ranging"  # Sideways
    TRAP_BULL = "trap_bull"  # Bull trap forming
    TRAP_BEAR = "trap_bear"  # Bear trap forming

@dataclass
class PhysicsScore:
    """Output of physics analysis"""
    total_score: float  # 0-100
    confidence: float  # 0-1
    direction: str  # 'UP', 'DOWN', 'NEUTRAL'
    regime: MarketRegime
    
    # Component scores
    kinetic_energy_score: float  # CVD divergence
    potential_energy_score: float  # OI analysis
    field_strength_score: float  # Liquidation pull
    friction_score: float  # Funding rate
    
    # Signals
    signals: List[str]
    warnings: List[str]
    
    # Edge calculation
    true_probability: float  # Our calculated probability
    market_probability: Optional[float] = None  # Polymarket odds
    market_probability: Optional[float] = None  # Polymarket odds
    edge: Optional[float] = None  # Difference
    reasoning: List[str] = None  # Natural language explanation

class PhysicsEngine:
    """
    Analyzes market structure using physics-inspired model:
    - Kinetic Energy: CVD (volume + momentum)
    - Potential Energy: Open Interest (fuel for moves)
    - Field Strength: Liquidation clusters (magnetic pull)
    - Friction: Funding rates (resistance to trend)
    """
    
    def __init__(self):
        # Thresholds (tunable)
        self.cvd_divergence_threshold = 0.15  # 15% divergence
        self.oi_change_threshold = 5.0  # 5% OI change
        self.funding_extreme_positive = 0.05  # 0.05% = crowded long
        self.funding_extreme_negative = -0.05  # -0.05% = crowded short
        self.liquidation_pull_threshold = 0.005  # 0.5% price distance
        
    def analyze(self, 
                current_snapshot,
                historical_snapshots: List,
                polymarket_odds: Optional[Dict] = None) -> PhysicsScore:
        """
        Main analysis function
        Returns a PhysicsScore with all components
        """
        
        if not current_snapshot or len(historical_snapshots) < 3:
            return self._create_neutral_score("Insufficient data")
        
        # Calculate each physics component
        kinetic = self._analyze_kinetic_energy(current_snapshot, historical_snapshots)
        potential = self._analyze_potential_energy(current_snapshot, historical_snapshots)
        field = self._analyze_field_strength(current_snapshot)
        friction = self._analyze_friction(current_snapshot)
        
        # Detect regime
        regime = self._detect_regime(kinetic, potential, field, friction)
        
        # Combine scores
        total_score = (
            kinetic['score'] * 0.30 +  # 30% weight
            potential['score'] * 0.25 +  # 25% weight
            field['score'] * 0.25 +  # 25% weight
            friction['score'] * 0.20  # 20% weight
        )
        
        # Determine direction
        if total_score >= 60:
            direction = 'UP'
        elif total_score <= 40:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        # Calculate true probability
        true_probability = total_score / 100.0
        
        # Collect all signals and warnings
        signals = (
            kinetic['signals'] +
            potential['signals'] +
            field['signals'] +
            friction['signals']
        )
        
        warnings = (
            kinetic['warnings'] +
            potential['warnings'] +
            field['warnings'] +
            friction['warnings']
        )
        
        # Calculate confidence (reduced by warnings)
        base_confidence = min(abs(total_score - 50) / 50, 1.0)  # 0 at 50, 1 at 0 or 100
        confidence = base_confidence * (1 - len(warnings) * 0.1)  # Each warning reduces by 10%
        confidence = max(0.3, min(confidence, 1.0))  # Clamp to 0.3-1.0
        
        # Calculate edge vs Polymarket
        market_probability = None
        edge = None
        if polymarket_odds:
            market_probability = polymarket_odds.get('up_odds', 0.5)
            edge = true_probability - market_probability
        
        return PhysicsScore(
            total_score=total_score,
            confidence=confidence,
            direction=direction,
            regime=regime,
            kinetic_energy_score=kinetic['score'],
            potential_energy_score=potential['score'],
            field_strength_score=field['score'],
            friction_score=friction['score'],
            signals=signals,
            warnings=warnings,
            true_probability=true_probability,
            market_probability=market_probability,
            edge=edge,
            reasoning=self._generate_reasoning(total_score, regime, direction, kinetic, potential, field, friction)
        )

    def _generate_reasoning(self, total_score, regime, direction, kinetic, potential, field, friction) -> List[str]:
        """Generate human-readable reasoning for the score"""
        reasons = []
        
        # 1. Regime Context
        if regime == MarketRegime.MARKUP:
            reasons.append("Market is in MARKUP phase (Strong Uptrend). All physics align nicely.")
        elif regime == MarketRegime.ACCUMULATION:
            reasons.append("Smart money is ACCUMULATING (Price flat, Buying pressure rising). Breakout likely.")
        elif regime == MarketRegime.DISTRIBUTION:
            reasons.append("Smart money is DISTRIBUTING (Price flat, Selling pressure rising). Dump likely.")
        elif regime == MarketRegime.MARKDOWN:
            reasons.append("Market is in MARKDOWN phase (Strong Downtrend). Catching a falling knife is dangerous.")
        elif regime == MarketRegime.TRAP_BULL:
            reasons.append("WARNING: Potential BULL TRAP detected. Price rising on weak/divergent metrics.")
        elif regime == MarketRegime.RANGING:
            reasons.append("Market is RANGING (Sideways). No clear trend.")
            
        # 2. Key Drivers
        # Kinetic (CVD)
        if kinetic['score'] > 60:
            reasons.append("Buying momentum is strong (CVD Divergence or Trend).")
        elif kinetic['score'] < 40:
            reasons.append("Selling pressure is dominant (CVD trending down).")
            
        # Potential (OI)
        if potential['score'] > 60:
            reasons.append("Open Interest is rising, fueling the move (High Conviction).")
        elif potential['score'] < 40:
            reasons.append("Open Interest is contracting, suggesting a squeeze or lack of interest.")
            
        # Field (Liquidation)
        if field['score'] > 60:
            reasons.append("Price is being magnetically pulled to a liquidation cluster ABOVE.")
        elif field['score'] < 40:
            reasons.append("Price is being magnetically pulled to a liquidation cluster BELOW.")
            
        # Friction (Funding)
        if friction['score'] < 40:
            reasons.append("Funding is crowded (Everyone is on one side). Reversal risk.")
        elif friction['score'] > 60:
            reasons.append("Funding is favorable (Short squeeze potential or healthy market).")
            
        return reasons
    
    def _analyze_kinetic_energy(self, current, history) -> Dict:
        """
        CVD Analysis - Force = Mass × Acceleration
        Detects divergences between price and volume
        """
        signals = []
        warnings = []
        score = 50  # Neutral start
        
        # Get price change
        if len(history) >= 2:
            price_start = history[0].price
            price_now = current.price
            if price_start > 0:
                price_change_pct = ((price_now - price_start) / price_start) * 100
            else:
                price_change_pct = 0
        else:
            return {'score': 50, 'signals': [], 'warnings': ['Insufficient history']}
        
        # Get CVD change
        cvd_current = current.cvd
        cvd_history = [s.cvd for s in history]
        
        if len(cvd_history) >= 2:
            cvd_trend = np.polyfit(range(len(cvd_history)), cvd_history, 1)[0]  # Linear trend
        else:
            cvd_trend = 0
        
        # === DIVERGENCE DETECTION ===
        
        # Bullish Divergence: Price down, CVD up (accumulation)
        if price_change_pct < -0.1 and cvd_trend > 0:
            score += 20
            signals.append(f"[BULL] Bullish CVD Divergence: Price {price_change_pct:.2f}% but CVD rising (accumulation)")
        
        # Bearish Divergence: Price up, CVD down (distribution)
        elif price_change_pct > 0.1 and cvd_trend < 0:
            score -= 20
            signals.append(f"[BEAR] Bearish CVD Divergence: Price {price_change_pct:.2f}% but CVD falling (distribution)")
        
        # Confirmation: Price and CVD agree
        elif (price_change_pct > 0.1 and cvd_trend > 0) or (price_change_pct < -0.1 and cvd_trend < 0):
            if price_change_pct > 0:
                score += 10
                signals.append(f"[OK] CVD confirms uptrend: Price {price_change_pct:.2f}%, CVD rising")
            else:
                score -= 10
                signals.append(f"[OK] CVD confirms downtrend: Price {price_change_pct:.2f}%, CVD falling")
        
        # Volume imbalance
        vol_imbalance = current.volume_imbalance
        if vol_imbalance > 1.5:
            score += 10
            signals.append(f"[OK] Strong buy pressure: {vol_imbalance:.2f}x more buyers")
        elif vol_imbalance < 0.67 and vol_imbalance > 0:
            score -= 10
            signals.append(f"[OK] Strong sell pressure: {1/vol_imbalance:.2f}x more sellers")
        
        # Warnings
        if abs(price_change_pct) < 0.05:
            warnings.append("[WARN] Low momentum: Price barely moving")
        
        if abs(cvd_current) < 10:
            warnings.append("[WARN] Low volume: CVD near zero")
        
        return {
            'score': max(0, min(100, score)),
            'signals': signals,
            'warnings': warnings
        }
    
    def _analyze_potential_energy(self, current, history) -> Dict:
        """
        Open Interest Analysis - Fuel for moves
        OI ↑ + Price ↑ = Real move
        OI ↓ + Price ↑ = Fake move (short covering)
        """
        signals = []
        warnings = []
        score = 50
        
        oi_change = current.open_interest_change_pct
        
        # Get price change
        if len(history) >= 2:
            price_start = history[0].price
            price_now = current.price
            if price_start > 0:
                price_change_pct = ((price_now - price_start) / price_start) * 100
            else:
                price_change_pct = 0
        else:
            price_change_pct = 0
        
        # === OI + PRICE ANALYSIS ===
        
        # Strong bullish: OI rising with price
        if oi_change > self.oi_change_threshold and price_change_pct > 0.1:
            score += 25
            signals.append(f"[BULL] Aggressive longs: OI +{oi_change:.1f}%, Price +{price_change_pct:.2f}%")
        
        # TRAP: Price up but OI down (short squeeze, will reverse)
        elif price_change_pct > 0.1 and oi_change < -self.oi_change_threshold:
            score -= 20
            warnings.append(f"[ALERT] BULL TRAP: Price +{price_change_pct:.2f}% but OI -{abs(oi_change):.1f}% (short covering)")
        
        # Strong bearish: OI rising with falling price
        elif oi_change > self.oi_change_threshold and price_change_pct < -0.1:
            score -= 25
            signals.append(f"[BEAR] Aggressive shorts: OI +{oi_change:.1f}%, Price {price_change_pct:.2f}%")
        
        # TRAP: Price down but OI down (long liquidation, may reverse)
        elif price_change_pct < -0.1 and oi_change < -self.oi_change_threshold:
            score += 15
            warnings.append(f"[WARN] Possible capitulation: Price {price_change_pct:.2f}%, OI -{abs(oi_change):.1f}%")
        
        # Flat OI = low conviction
        if abs(oi_change) < 2.0:
            warnings.append(f"[WARN] Flat OI: {oi_change:+.1f}% (low conviction)")
        
        return {
            'score': max(0, min(100, score)),
            'signals': signals,
            'warnings': warnings
        }
    
    def _analyze_field_strength(self, current) -> Dict:
        """
        Liquidation Cluster Analysis - Magnetic pull
        Price is attracted to high-liquidity zones
        """
        signals = []
        warnings = []
        score = 50
        
        price = current.price
        cluster_above = current.liquidation_cluster_above
        cluster_below = current.liquidation_cluster_below
        
        if price <= 0:
            return {
                'score': 50,
                'signals': [],
                'warnings': ['[ERR] Invalid price data']
            }
        
        # Calculate pull strength
        if cluster_above:
            distance_above = ((cluster_above - price) / price)
            pull_strength_above = current.liquidation_strength_above / 1e9  # Normalize to billions
            
            if distance_above < 0.01:  # Within 1%
                score += 15 * min(pull_strength_above, 2.0)  # Cap at 30 points
                signals.append(f"[PULL UP] Strong pull upward: ${cluster_above:,.0f} ({distance_above*100:.2f}% away, ${pull_strength_above:.1f}B liq)")
        else:
            distance_above = 1.0 # Default large distance
        
        if cluster_below:
            distance_below = ((price - cluster_below) / price)
            pull_strength_below = current.liquidation_strength_below / 1e9
            
            if distance_below < 0.01:  # Within 1%
                score -= 15 * min(pull_strength_below, 2.0)
                signals.append(f"[PULL DOWN] Strong pull downward: ${cluster_below:,.0f} ({distance_below*100:.2f}% away, ${pull_strength_below:.1f}B liq)")
        else:
            distance_below = 1.0 # Default large distance
        
        # Warning if between two strong clusters (chop zone)
        if cluster_above and cluster_below:
            if distance_above < 0.02 and distance_below < 0.02:
                warnings.append("[WARN] Trapped between liquidation clusters - expect chop")
        
        return {
            'score': max(0, min(100, score)),
            'signals': signals,
            'warnings': warnings
        }
    
    def _analyze_friction(self, current) -> Dict:
        """
        Funding Rate Analysis - Market sentiment friction
        High positive funding = crowded longs = danger
        High negative funding = crowded shorts = reversal potential
        """
        signals = []
        warnings = []
        score = 50
        
        funding = current.funding_rate
        ls_ratio = current.long_short_ratio
        
        # === FUNDING RATE ANALYSIS ===
        
        # Extremely positive funding = too many longs = danger
        if funding > self.funding_extreme_positive:
            score -= 20
            warnings.append(f"[ALERT] CROWDED LONG: Funding +{funding*100:.3f}% (likely to dump)")
        
        # Negative funding = shorts piling in = squeeze potential
        elif funding < self.funding_extreme_negative:
            score += 15
            signals.append(f"[BULL] SHORT SQUEEZE SETUP: Funding {funding*100:.3f}% (shorts crowded)")
        
        # Neutral funding = healthy
        elif abs(funding) < 0.01:
            signals.append(f"[OK] Healthy funding: {funding*100:.3f}% (not crowded)")
        
        # === LONG/SHORT RATIO ===
        
        # Too many longs
        if ls_ratio > 2.0:
            score -= 10
            warnings.append(f"[WARN] L/S Ratio {ls_ratio:.2f}: Longs crowded")
        
        # Too many shorts
        elif ls_ratio < 0.5:
            score += 10
            signals.append(f"[OK] L/S Ratio {ls_ratio:.2f}: Shorts vulnerable")
        
        return {
            'score': max(0, min(100, score)),
            'signals': signals,
            'warnings': warnings
        }
    
    def _detect_regime(self, kinetic, potential, field, friction) -> MarketRegime:
        """Classify current market structure"""
        
        # Bull trap: Price up, CVD down, OI down
        if (kinetic['score'] < 40 and 
            potential['score'] < 40 and 
            "TRAP" in str(potential['warnings'])):
            return MarketRegime.TRAP_BULL
        
        # Bear trap: Price down, CVD up, OI changing
        if (kinetic['score'] > 60 and 
            potential['score'] < 50 and
            "capitulation" in str(potential['warnings']).lower()):
            return MarketRegime.TRAP_BEAR
        
        # Accumulation: CVD up, price flat/down
        if kinetic['score'] > 60 and "Divergence" in str(kinetic['signals']):
            return MarketRegime.ACCUMULATION
        
        # Distribution: CVD down, price flat/up
        if kinetic['score'] < 40 and "Divergence" in str(kinetic['signals']):
            return MarketRegime.DISTRIBUTION
        
        # Markup: All bullish
        if (kinetic['score'] > 60 and 
            potential['score'] > 60 and 
            friction['score'] > 50):
            return MarketRegime.MARKUP
        
        # Markdown: All bearish
        if (kinetic['score'] < 40 and 
            potential['score'] < 40 and 
            friction['score'] < 50):
            return MarketRegime.MARKDOWN
        
        # Default: Ranging
        return MarketRegime.RANGING
    
    def _create_neutral_score(self, reason: str) -> PhysicsScore:
        """Return neutral score when analysis cannot be performed"""
        return PhysicsScore(
            total_score=50,
            confidence=0.0,
            direction='NEUTRAL',
            regime=MarketRegime.RANGING,
            kinetic_energy_score=50,
            potential_energy_score=50,
            field_strength_score=50,
            friction_score=50,
            signals=[],
            warnings=[reason],
            true_probability=0.5,
            market_probability=None,
            edge=None,
            reasoning=[reason]
        )

# Example usage
if __name__ == "__main__":
    from data_aggregator import DataAggregator, MarketSnapshot
    import time
    
    print("Physics Engine Test")
    print("=" * 70)
    
    # Create mock data
    mock_snapshot = MarketSnapshot(
        timestamp=time.time(),
        symbol='BTC',
        price=95000,
        volume_24h=1000000,
        cvd=500,  # Positive CVD
        open_interest=5000000000,
        open_interest_change_pct=8.5,  # Rising OI
        funding_rate=0.01,  # 0.01% = neutral
        long_short_ratio=1.2,
        liquidation_cluster_above=96000,
        liquidation_cluster_below=94000,
        liquidation_strength_above=1000000000,
        liquidation_strength_below=800000000,
        volume_delta=500,
        volume_imbalance=1.8  # More buyers
    )
    
    mock_history = [
        MarketSnapshot(
            timestamp=time.time() - 300,
            symbol='BTC',
            price=94800,
            volume_24h=980000,
            cvd=200,
            open_interest=4900000000,
            open_interest_change_pct=5.0,
            funding_rate=0.008,
            long_short_ratio=1.1,
            liquidation_cluster_above=96000,
            liquidation_cluster_below=94000,
            liquidation_strength_above=1000000000,
            liquidation_strength_below=800000000,
            volume_delta=200,
            volume_imbalance=1.3
        )
    ]
    
    mock_polymarket = {
        'up_odds': 0.58,  # Market thinks 58% chance up
        'down_odds': 0.42
    }
    
    # Run analysis
    engine = PhysicsEngine()
    result = engine.analyze(mock_snapshot, mock_history, mock_polymarket)
    
    print(f"\nRESULT:")
    print(f"Total Score: {result.total_score:.1f}/100")
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Regime: {result.regime.value.upper()}")
    print(f"\nComponent Scores:")
    print(f"  Kinetic Energy (CVD): {result.kinetic_energy_score:.1f}")
    print(f"  Potential Energy (OI): {result.potential_energy_score:.1f}")
    print(f"  Field Strength (Liq): {result.field_strength_score:.1f}")
    print(f"  Friction (Funding): {result.friction_score:.1f}")
    
    print(f"\n[SIGNALS]:")
    for signal in result.signals:
        print(f"  {signal}")
    
    if result.warnings:
        print(f"\n[WARNINGS]:")
        for warning in result.warnings:
            print(f"  {warning}")
    
    print(f"\n[EDGE] EDGE ANALYSIS:")
    print(f"  True Probability: {result.true_probability:.1%}")
    print(f"  Market Probability: {result.market_probability:.1%}")
    print(f"  EDGE: {result.edge:+.1%}")
    
    if abs(result.edge) > 0.15:
        print(f"\n[TARGET] STRONG SIGNAL: Edge > 15% - TRADE RECOMMENDED")
    elif abs(result.edge) > 0.10:
        print(f"\n[TIP] MODERATE SIGNAL: Edge > 10% - Consider trading")
    else:
        print(f"\n[SKIP] NO EDGE: Market fairly priced - SKIP")
