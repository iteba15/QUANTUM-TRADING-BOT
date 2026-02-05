#!/usr/bin/env python3
"""
Quantum Predictor - Main Orchestrator
Combines Physics Engine + ML Predictions + Polymarket Odds
Multi-timeframe, multi-asset prediction system
"""

import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Import our modules
from data_aggregator import DataAggregator, MarketSnapshot
from physics_engine import PhysicsEngine, PhysicsScore, MarketRegime
from ml_engine import EnsemblePredictor, PredictionResult

class TimeWindow(Enum):
    """Prediction time windows"""
    MIN_15 = "15min"
    HOUR_1 = "1hour"
    HOUR_4 = "4hour"

@dataclass
class TradingSignal:
    """Final trading signal output"""
    timestamp: float
    symbol: str
    timeframe: TimeWindow
    
    # Core signal
    action: str  # 'LONG', 'SHORT', 'WAIT'
    confidence: float  # 0-1
    edge: float  # Our probability - Market probability
    
    # Probabilities
    physics_probability: float
    ml_probability: float
    combined_probability: float
    market_probability: float
    
    # Position sizing
    recommended_position_pct: float  # % of bankroll
    expected_roi: float
    
    # Supporting data
    physics_score: PhysicsScore
    ml_prediction: PredictionResult
    regime: MarketRegime
    
    # Risk metrics
    risk_score: float  # 0-1 (higher = riskier)
    stop_loss: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: List[str] = None  # Natural language reasoning

class QuantumPredictor:
    """
    Main orchestrator for the complete prediction system
    """
    
    def __init__(self, 
                 symbols: List[str] = ['BTC', 'ETH', 'SOL', 'XRP'],
                 coinglass_api_key: Optional[str] = None,
                 use_ml: bool = True):
        
        self.symbols = symbols
        self.use_ml = use_ml
        
        # Initialize components
        print("Initializing Quantum Predictor...")
        self.data_aggregator = DataAggregator()  # No longer needs API key
        self.physics_engine = PhysicsEngine()
        
        if use_ml:
            self.predictors = {}
            print("\n[ML] Loading models...")
            for symbol in symbols:
                predictor = EnsemblePredictor()
                model_path = f"models/{symbol}_trained.pth"
                try:
                    # We need a way to check if file exists without importing Path if not imported
                    # But EnsemblePredictor.load_models handles the loading.
                    # Let's just try to load it.
                    predictor.load_models(model_path)
                    self.predictors[symbol] = predictor
                    print(f"  [OK] Loaded {symbol} model")
                except Exception:
                    print(f"  [WARN] No model for {symbol} (will use physics only)")
                    self.predictors[symbol] = None
        else:
            self.predictors = {s: None for s in symbols}
        
        # Configuration
        self.min_confidence = 0.65  # Minimum confidence to trade
        self.min_edge = 0.12  # Minimum edge vs market (12%)
        self.max_position_pct = 0.40  # Max 40% of bankroll per trade
        
        # Historical signals for tracking
        self.signal_history = {symbol: [] for symbol in symbols}
        
    def start(self):
        """Initialize and start data collection"""
        print("Starting data aggregator...")
        self.data_aggregator.start()
        print("[OK] Data streams active")
        
        # Wait for initial data
        print("Waiting for initial data (10 seconds)...")
        time.sleep(10)
        
    def stop(self):
        """Stop all data streams"""
        self.data_aggregator.stop()
    
    def analyze_symbol(self, 
                      symbol: str, 
                      timeframe: TimeWindow = TimeWindow.MIN_15,
                      bankroll: float = 5.0) -> Optional[TradingSignal]:
        """
        Perform complete analysis for a symbol
        Returns trading signal if edge exists, None otherwise
        """
        
        print(f"\n{'='*70}")
        print(f"ANALYZING {symbol} - {timeframe.value}")
        print(f"{'='*70}")
        
        # Get current snapshot
        current_snapshot = self.data_aggregator.get_snapshot(symbol)
        if not current_snapshot:
            print(f"[ERROR] No data available for {symbol}")
            return None
        
        # Get historical snapshots
        history_length = self._get_history_length(timeframe)
        historical_snapshots = self.data_aggregator.get_historical_snapshots(
            symbol, count=history_length
        )
        
        if len(historical_snapshots) < 10:
            print(f"[ERROR] Insufficient historical data (have {len(historical_snapshots)}, need 10+)")
            return None
        
        # Get Polymarket odds
        polymarket_odds = self.data_aggregator.get_polymarket_odds(symbol)
        
        # === PHYSICS ANALYSIS ===
        print("\n[PHYSICS] Running Physics Analysis...")
        physics_result = self.physics_engine.analyze(
            current_snapshot,
            historical_snapshots,
            polymarket_odds
        )
        
        self._print_physics_results(physics_result)
        
        # === ML PREDICTION ===
        ml_result = None
        predictor = self.predictors.get(symbol)
        
        if self.use_ml and predictor and predictor.trained:
            print(f"\n[ML] Running {symbol} Model Prediction...")
            ml_result = predictor.predict(
                historical_snapshots + [current_snapshot]
            )
            self._print_ml_results(ml_result)
        
        # === COMBINE SIGNALS ===
        signal = self._generate_signal(
            symbol=symbol,
            timeframe=timeframe,
            current_snapshot=current_snapshot,
            physics_result=physics_result,
            ml_result=ml_result,
            polymarket_odds=polymarket_odds,
            bankroll=bankroll
        )
        
        # Store signal
        if signal:
            self.signal_history[symbol].append(signal)
        
        return signal
    
    def _generate_signal(self,
                        symbol: str,
                        timeframe: TimeWindow,
                        current_snapshot,
                        physics_result: PhysicsScore,
                        ml_result: Optional[PredictionResult],
                        polymarket_odds: Optional[Dict],
                        bankroll: float) -> Optional[TradingSignal]:
        """
        Combine physics + ML to generate final signal
        """
        
        print("\n" + "="*70)
        print("SIGNAL GENERATION")
        print("="*70)
        
        # Calculate combined probability
        if ml_result:
            # Weight: 60% physics, 40% ML
            combined_prob = (
                physics_result.true_probability * 0.60 +
                ml_result.probability_up * 0.40
            )
            # Combined confidence
            combined_confidence = (
                physics_result.confidence * 0.60 +
                ml_result.confidence * 0.40
            )
        else:
            combined_prob = physics_result.true_probability
            combined_confidence = physics_result.confidence
        
        # Get market probability
        if polymarket_odds:
            market_prob = polymarket_odds['up_odds']
        else:
            print("[WARN] No Polymarket odds available - cannot calculate edge")
            market_prob = 0.5  # Assume neutral if no market
        
        # Calculate edge
        edge = combined_prob - market_prob
        
        print(f"\n[DATA] PROBABILITIES:")
        print(f"  Physics Model: {physics_result.true_probability:.1%}")
        if ml_result:
            print(f"  ML Model: {ml_result.probability_up:.1%}")
        print(f"  Combined: {combined_prob:.1%}")
        print(f"  Polymarket: {market_prob:.1%}")
        print(f"  EDGE: {edge:+.1%}")
        print(f"  Confidence: {combined_confidence:.1%}")
        
        # Decision criteria
        if combined_confidence < self.min_confidence:
            print(f"\n[SKIP] Confidence {combined_confidence:.1%} < {self.min_confidence:.0%}")
            return None
        
        if abs(edge) < self.min_edge:
            print(f"\n[SKIP] Edge {abs(edge):.1%} < {self.min_edge:.0%}")
            return None
        
        # Determine action
        if edge > 0:
            action = 'LONG'
            entry_odds = polymarket_odds['up_odds'] if polymarket_odds else 0.5
        else:
            action = 'SHORT'
            entry_odds = polymarket_odds['down_odds'] if polymarket_odds else 0.5
        
        # Position sizing (Kelly Criterion with conservative fraction)
        if entry_odds < 1 and entry_odds > 0:
            kelly_fraction = abs(edge) / (1 - entry_odds)
        else:
            kelly_fraction = 0
            
        kelly_fraction = min(kelly_fraction, 1.0)  # Cap at 100%
        
        # Use 25% of Kelly (conservative)
        position_pct = kelly_fraction * 0.25 * combined_confidence
        position_pct = min(position_pct, self.max_position_pct)
        position_pct = max(position_pct, 0.20)  # Minimum 20% if trading
        
        position_size = bankroll * position_pct
        
        # Expected ROI
        if entry_odds > 0:
            expected_win = position_size / entry_odds
            expected_roi = ((expected_win - position_size) / position_size) * combined_prob
        else:
            expected_win = 0
            expected_roi = 0
        
        # Risk assessment
        risk_score = self._calculate_risk_score(
            physics_result, 
            combined_confidence, 
            edge,
            current_snapshot
        )
        
        # Create signal
        signal = TradingSignal(
            timestamp=time.time(),
            symbol=symbol,
            timeframe=timeframe,
            action=action,
            confidence=combined_confidence,
            edge=edge,
            physics_probability=physics_result.true_probability,
            ml_probability=ml_result.probability_up if ml_result else combined_prob,
            combined_probability=combined_prob,
            market_probability=market_prob,
            recommended_position_pct=position_pct,
            expected_roi=expected_roi,
            physics_score=physics_result,
            ml_prediction=ml_result,
            regime=physics_result.regime,
            risk_score=risk_score,
            reasoning=physics_result.reasoning
        )
        
        self._print_signal(signal, position_size, bankroll)
        
        return signal
    
    def _calculate_risk_score(self, 
                             physics: PhysicsScore,
                             confidence: float,
                             edge: float,
                             snapshot) -> float:
        """
        Calculate risk score based on multiple factors
        0 = low risk, 1 = high risk
        """
        risk = 0.0
        
        # Warning count increases risk
        risk += len(physics.warnings) * 0.15
        
        # Low confidence increases risk
        if confidence < 0.70:
            risk += (0.70 - confidence)
        
        # Extreme funding = crowded = risky
        if abs(snapshot.funding_rate) > 0.05:
            risk += 0.20
        
        # Trap regimes = risky
        if physics.regime in [MarketRegime.TRAP_BULL, MarketRegime.TRAP_BEAR]:
            risk += 0.25
        
        return min(risk, 1.0)
    
    def _get_history_length(self, timeframe: TimeWindow) -> int:
        """Get number of snapshots needed for timeframe"""
        if timeframe == TimeWindow.MIN_15:
            return 30  # 15 min / 30 sec per snapshot
        elif timeframe == TimeWindow.HOUR_1:
            return 60
        elif timeframe == TimeWindow.HOUR_4:
            return 240
        return 30
    
    def _print_physics_results(self, result: PhysicsScore):
        """Pretty print physics analysis"""
        print(f"\n  Total Score: {result.total_score:.1f}/100")
        print(f"  Direction: {result.direction}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Regime: {result.regime.value.upper()}")
        
        print(f"\n  Component Scores:")
        print(f"    Kinetic (CVD): {result.kinetic_energy_score:.1f}")
        print(f"    Potential (OI): {result.potential_energy_score:.1f}")
        print(f"    Field (Liq): {result.field_strength_score:.1f}")
        print(f"    Friction (Fund): {result.friction_score:.1f}")
        
        if result.signals:
            print(f"\n  [SIGNALS]:")
            for signal in result.signals[:5]:  # Top 5
                print(f"    {signal}")
        
        if result.warnings:
            print(f"\n  [WARNINGS]:")
            for warning in result.warnings[:3]:  # Top 3
                print(f"    {warning}")
    
    def _print_ml_results(self, result: PredictionResult):
        """Pretty print ML prediction"""
        print(f"\n  UP Probability: {result.probability_up:.1%}")
        print(f"  DOWN Probability: {result.probability_down:.1%}")
        print(f"  Confidence: {result.confidence:.1%}")
        
        print(f"\n  Model Scores:")
        for model, score in result.model_scores.items():
            print(f"    {model}: {score:.1%}")
    
    def _print_signal(self, signal: TradingSignal, position_size: float, bankroll: float):
        """Pretty print final trading signal"""
        print(f"\n{'='*70}")
        print(f"[TARGET] TRADING SIGNAL")
        print(f"{'='*70}")
        
        print(f"\n  ACTION: {signal.action}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Edge: {signal.edge:+.1%}")
        print(f"  Risk Score: {signal.risk_score:.1%} {'[LOW]' if signal.risk_score < 0.3 else '[MED]' if signal.risk_score < 0.6 else '[HIGH]'}")
        
        print(f"\n  POSITION SIZING:")
        print(f"    Bankroll: ${bankroll:.2f}")
        print(f"    Position: ${position_size:.2f} ({signal.recommended_position_pct*100:.1f}%)")
        print(f"    Entry Odds: {signal.market_probability:.3f}")
        print(f"    Expected ROI: {signal.expected_roi:+.1%}")
        
        win_amount = position_size / signal.market_probability
        profit = win_amount - position_size
        
        print(f"\n  IF WIN:")
        print(f"    Payout: ${win_amount:.2f}")
        print(f"    Profit: +${profit:.2f}")
        
        print(f"\n  IF LOSE:")
        print(f"    Loss: -${position_size:.2f}")
        
        print(f"\n  REGIME: {signal.regime.value.upper()}")
        
        print(f"\n{'='*70}")
    
    def scan_all_markets(self, 
                        timeframes: List[TimeWindow] = [TimeWindow.MIN_15],
                        bankroll: float = 5.0) -> List[TradingSignal]:
        """
        Scan all symbols and timeframes for trading opportunities
        """
        print("\n" + "="*70)
        print("SCANNING ALL MARKETS")
        print("="*70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timeframes: {', '.join([tf.value for tf in timeframes])}")
        print(f"Bankroll: ${bankroll:.2f}")
        
        all_signals = []
        
        for symbol in self.symbols:
            for timeframe in timeframes:
                signal = self.analyze_symbol(symbol, timeframe, bankroll)
                if signal:
                    all_signals.append(signal)
        
        # Sort by edge (best opportunities first)
        all_signals.sort(key=lambda s: abs(s.edge), reverse=True)
        
        print("\n" + "="*70)
        print(f"SCAN COMPLETE - Found {len(all_signals)} opportunities")
        print("="*70)
        
        return all_signals
    
    def get_top_signal(self, 
                      timeframes: List[TimeWindow] = [TimeWindow.MIN_15],
                      bankroll: float = 5.0) -> Optional[TradingSignal]:
        """Get the single best trading opportunity right now"""
        signals = self.scan_all_markets(timeframes, bankroll)
        return signals[0] if signals else None

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("POLYMARKET QUANTUM PREDICTOR v2.0")
    print("GPU-Accelerated Multi-Asset Prediction System")
    print("="*70)
    
    # Configuration
    SYMBOLS = ['BTC', 'ETH', 'SOL', 'XRP']
    TIMEFRAMES = [TimeWindow.MIN_15]  # Start with 15-min
    BANKROLL = 5.0
    SCAN_INTERVAL = 60  # Scan every 60 seconds
    
    # Initialize predictor
    predictor = QuantumPredictor(
        symbols=SYMBOLS,
        coinglass_api_key=None,  # Add your key here
        use_ml=False  # Set to True once models are trained
    )
    
    # Start data collection
    predictor.start()
    
    print("\n[OK] System ready. Starting market scan...")
    print(f"Configuration:")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Timeframes: {', '.join([tf.value for tf in TIMEFRAMES])}")
    print(f"  Bankroll: ${BANKROLL:.2f}")
    print(f"  Scan Interval: {SCAN_INTERVAL}s")
    print(f"  Min Confidence: {predictor.min_confidence:.0%}")
    print(f"  Min Edge: {predictor.min_edge:.0%}")
    
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        while True:
            # Scan all markets
            signals = predictor.scan_all_markets(TIMEFRAMES, BANKROLL)
            
            if signals:
                print(f"\n{'='*70}")
                print(f"TOP OPPORTUNITIES ({datetime.now().strftime('%H:%M:%S')})")
                print(f"{'='*70}\n")
                
                for i, signal in enumerate(signals[:3], 1):  # Top 3
                    print(f"{i}. {signal.symbol} - {signal.action}")
                    print(f"   Edge: {signal.edge:+.1%} | Confidence: {signal.confidence:.1%}")
                    print(f"   Position: ${signal.recommended_position_pct * BANKROLL:.2f}")
                    print()
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No trading opportunities found.")
            
            # Wait before next scan
            time.sleep(SCAN_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nStopping predictor...")
        predictor.stop()
        print("[OK] Stopped.")
