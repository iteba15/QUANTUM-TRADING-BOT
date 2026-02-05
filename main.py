#!/usr/bin/env python3
"""
Polymarket Quantum Predictor - Main Trading Bot
Automated trading system combining physics + ML analysis
"""

import time
import sys
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import all components
from data_aggregator import DataAggregator, MarketSnapshot
from physics_engine import PhysicsEngine, PhysicsScore, MarketRegime
from ml_engine import EnsemblePredictor, PredictionResult
from quantum_predictor import QuantumPredictor, TradingSignal, TimeWindow
from collect_training_data import TrainingDataCollector
from threading import Thread

@dataclass
class TradingOpportunity:
    """Ranked trading opportunity"""
    signal: TradingSignal
    snapshot: MarketSnapshot
    physics: PhysicsScore
    ml: PredictionResult
    score: float  # Combined score for ranking

class QuantumTradingBot:
    """
    Main trading bot that orchestrates all components
    Runs continuously, generates signals, displays opportunities
    """
    
    def __init__(self, 
                 symbols: List[str] = ['BTC', 'ETH', 'SOL', 'XRP'],
                 update_interval: int = 30,
                 min_edge: float = 0.12,  # 12% minimum edge
                 min_confidence: float = 0.65,  # 65% minimum confidence
                 auto_trade: bool = False,
                 enable_data_collection: bool = True):  # NEW: Auto-collect training data
        
        self.symbols = symbols
        self.update_interval = update_interval
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.auto_trade = auto_trade
        self.enable_data_collection = enable_data_collection
        
        # Initialize components
        print("Initializing Quantum Trading Bot...")
        print("="*70)
        
        self.aggregator = DataAggregator()
        self.physics = PhysicsEngine()
        self.aggregator = DataAggregator()
        self.physics = PhysicsEngine()
        # self.ml removed - QuantumPredictor handles it now
        self.predictor = QuantumPredictor(symbols=symbols, use_ml=True)
        
        # Initialize training data collector
        if self.enable_data_collection:
            self.data_collector = TrainingDataCollector(symbols=symbols)
            self.collection_thread = None
            print("[OK] Training Data Collector initialized")
        
        # Track signals
        self.signal_history = []
        self.opportunities_found = 0
        self.cycles_completed = 0
        
        # Validation state
        self.previous_snapshots = {}  # symbol -> snapshot
        self.previous_signals = {}    # symbol -> signal
        
        print("[OK] Data Aggregator initialized")
        print("[OK] Physics Engine initialized")
        print("[OK] ML Engine initialized")
        print("[OK] Quantum Predictor initialized")
        print("="*70)
    
    def start(self):
        """Start the trading bot"""
        print("\n" + "="*70)
        print("STARTING QUANTUM TRADING BOT")
        print("="*70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Update Interval: {self.update_interval}s")
        print(f"Min Edge: {self.min_edge:.0%}")
        print(f"Min Confidence: {self.min_confidence:.0%}")
        print(f"Auto-Trade: {'ENABLED' if self.auto_trade else 'DISABLED (Display Only)'}")
        print("="*70)
        
        # Start data collection
        print("\nStarting data aggregator...")
        self.aggregator.start()
        
        # ML models are loaded inside QuantumPredictor now
        # We don't need to load them here manually
        
        # Start background data collection
        if self.enable_data_collection:
            print("\nStarting training data collection...")
            self.collection_thread = Thread(
                target=self._run_data_collection,
                daemon=True
            )
            self.collection_thread.start()
            print("[OK] Data collection running in background")
            print("[INFO] Training data will be saved to training_data/")
        
        # Initial data collection
        print(f"\nCollecting initial data ({self.update_interval}s)...")
        time.sleep(self.update_interval)
        
        print("\n" + "="*70)
        print("BOT IS NOW RUNNING")
        print("="*70)
        print("Press Ctrl+C to stop\n")
        
        # Main loop
        try:
            while True:
                self._run_analysis_cycle()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            self._shutdown()
    
    def _run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        self.cycles_completed += 1
        
        print(f"\n{'='*70}")
        print(f"CYCLE #{self.cycles_completed} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        print(f"CYCLE #{self.cycles_completed} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # 1. Validate previous cycle
        if self.previous_signals:
            print("[VALIDATION] Checking previous predictions...")
            for symbol, prev_signal in self.previous_signals.items():
                current_snapshot = self.aggregator.get_snapshot(symbol)
                prev_snapshot = self.previous_snapshots.get(symbol)
                
                if current_snapshot and prev_snapshot:
                    price_change = (current_snapshot.price - prev_snapshot.price) / prev_snapshot.price
                    
                    # Determine result
                    result = "NEUTRAL"
                    if prev_signal.action == 'LONG':
                        if price_change > 0: result = "SUCCESS"
                        elif price_change < 0: result = "FAIL"
                    elif prev_signal.action == 'SHORT':
                        if price_change < 0: result = "SUCCESS"
                        elif price_change > 0: result = "FAIL"
                        
                    print(f"  {symbol}: {prev_signal.action} @ ${prev_snapshot.price:.2f} -> ${current_snapshot.price:.2f} ({price_change:+.2%}) [{result}]")
            print(f"{'-'*70}")
            
        # Reset previous state for this new cycle
        self.previous_signals = {}
        self.previous_snapshots = {}

        opportunities = []
        
        # Analyze each symbol
        for symbol in self.symbols:
            try:
                opportunity = self._analyze_symbol(symbol)
                if opportunity:
                    opportunities.append(opportunity)
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                continue
        
        # Display results
        if opportunities:
            self._display_opportunities(opportunities)
            
            # Execute trades if enabled
            if self.auto_trade:
                self._execute_best_opportunity(opportunities)
                
            # Store top opportunity for validation next cycle
            # We only validate the "best" one if we are just watching, 
            # OR we can validate all signals that met criteria.
            # Let's validate ALL opportunities found:
            for opp in opportunities:
                self._store_for_validation(opp.signal.symbol, opp.signal, opp.snapshot)
        else:
            print("\n[INFO] No trading opportunities found this cycle")
            print(f"      Waiting {self.update_interval}s for next cycle...")
    
    def _analyze_symbol(self, symbol: str) -> Optional[TradingOpportunity]:
        """Analyze a single symbol and generate signal"""
        
        # Get latest snapshot
        snapshot = self.aggregator.get_snapshot(symbol)
        if not snapshot:
            print(f"[SKIP] {symbol}: No data available")
            return None
        
        print(f"\n[ANALYZING] {symbol} @ ${snapshot.price:,.2f}")
        
        # Get historical snapshots for physics analysis
        historical_snapshots = self.aggregator.get_historical_snapshots(symbol, count=30)
        
        # Run physics analysis
        physics_result = self.physics.analyze(
            current_snapshot=snapshot,
            historical_snapshots=historical_snapshots
        )
        print(f"  Physics: {physics_result.total_score:.0f}/100 ({physics_result.direction})")
        
        # Run ML prediction via Quantum Predictor
        # We don't need to run it separately here anymore
        ml_result = None
        
        # Generate trading signal using QuantumPredictor's method
        signal = self.predictor.analyze_symbol(
            symbol=symbol,
            timeframe=TimeWindow.MIN_15,
            bankroll=100.0  # Default bankroll for signal generation
        )
        
        # Check if it meets our criteria
        if not signal:
            return None
            
        # Extract ML result from signal if available
        if signal.ml_prediction:
            ml_result = signal.ml_prediction
            print(f"  ML: {ml_result.probability_up:.0%} UP (confidence: {ml_result.confidence:.0%})")
        
        # Check if it meets our criteria
        if not signal:
            return None

        if signal.action == 'WAIT':
            print(f"  Signal: WAIT (edge: {signal.edge:+.1%}, confidence: {signal.confidence:.0%})")
            return None
        
        if signal.edge < self.min_edge or signal.confidence < self.min_confidence:
            print(f"  Signal: {signal.action} (below threshold)")
            print(f"         Edge: {signal.edge:+.1%} (need {self.min_edge:.0%})")
            print(f"         Confidence: {signal.confidence:.0%} (need {self.min_confidence:.0%})")
            return None
        
        # Valid opportunity!
        print(f"  Signal: {signal.action} [OPPORTUNITY!]")
        print(f"         Edge: {signal.edge:+.1%}")
        print(f"         Confidence: {signal.confidence:.0%}")
        
        # Calculate combined score for ranking
        score = signal.edge * signal.confidence
        
        return TradingOpportunity(
            signal=signal,
            snapshot=snapshot,
            physics=physics_result,
            ml=ml_result,
            score=score
        )
        
    def _store_for_validation(self, symbol: str, signal: TradingSignal, snapshot: MarketSnapshot):
        """Store signal and snapshot for next cycle validation"""
        self.previous_signals[symbol] = signal
        self.previous_snapshots[symbol] = snapshot
    
    def _display_opportunities(self, opportunities: List[TradingOpportunity]):
        """Display trading opportunities in ranked order"""
        
        # Sort by score (best first)
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        self.opportunities_found += len(opportunities)
        
        print(f"\n{'='*70}")
        print(f"FOUND {len(opportunities)} TRADING OPPORTUNIT{'Y' if len(opportunities) == 1 else 'IES'}")
        print(f"{'='*70}")
        
        for i, opp in enumerate(opportunities, 1):
            self._display_opportunity(i, opp)
        
        print(f"{'='*70}")
        print(f"Total Opportunities Found: {self.opportunities_found}")
        print(f"{'='*70}")
    
    def _display_opportunity(self, rank: int, opp: TradingOpportunity):
        """Display a single opportunity with rich details"""
        
        signal = opp.signal
        snapshot = opp.snapshot
        physics = opp.physics
        
        print(f"\n[#{rank}] {signal.symbol} - {signal.action}")
        print(f"{'─'*70}")
        
        # Display Reasoning
        if signal.reasoning:
            print("  ANALYSIS SUMMARY:")
            for reason in signal.reasoning:
                print(f"   • {reason}")
            print("")
        
        # Core metrics
        print(f"  Price: ${snapshot.price:,.2f}")
        print(f"  Edge: {signal.edge:+.1%} (vs market {signal.market_probability:.0%})")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Combined Score: {opp.score:.4f}")
        
        # Probabilities
        print(f"\n  Probabilities:")
        print(f"    Physics: {signal.physics_probability:.0%}")
        if signal.ml_probability:
            print(f"    ML: {signal.ml_probability:.0%}")
        print(f"    Combined: {signal.combined_probability:.0%}")
        print(f"    Market: {signal.market_probability:.0%}")
        
        # Position sizing
        print(f"\n  Position Sizing:")
        print(f"    Recommended: {signal.recommended_position_pct:.0%} of bankroll")
        print(f"    Expected ROI: {signal.expected_roi:+.1%}")
        
        # Risk assessment
        print(f"\n  Risk Assessment:")
        print(f"    Risk Score: {signal.risk_score:.0%} ({'Low' if signal.risk_score < 0.3 else 'Medium' if signal.risk_score < 0.6 else 'High'})")
        print(f"    Regime: {signal.regime.name if hasattr(signal.regime, 'name') else signal.regime}")
        
        # Market data
        print(f"\n  Market Data:")
        print(f"    Open Interest: ${snapshot.open_interest:,.0f}")
        print(f"    Funding Rate: {snapshot.funding_rate:.4f}%")
        print(f"    Long/Short: {snapshot.long_short_ratio:.2f}")
        print(f"    CVD: {snapshot.cvd:,.0f}")
        
        # Liquidation clusters
        if snapshot.liquidation_cluster_above:
            print(f"    Liq Cluster Above: ${snapshot.liquidation_cluster_above:,.2f} (${snapshot.liquidation_strength_above/1e6:.1f}M)")
        if snapshot.liquidation_cluster_below:
            print(f"    Liq Cluster Below: ${snapshot.liquidation_cluster_below:,.2f} (${snapshot.liquidation_strength_below/1e6:.1f}M)")
        
        # Physics signals
        if physics.signals:
            print(f"\n  Physics Signals:")
            for sig in physics.signals[:3]:  # Top 3
                print(f"    {sig}")
        
        # Warnings
        if physics.warnings:
            print(f"\n  Warnings:")
            for warn in physics.warnings:
                print(f"    {warn}")
    
    def _execute_best_opportunity(self, opportunities: List[TradingOpportunity]):
        """Execute trade for the best opportunity"""
        
        if not opportunities:
            return
        
        best = opportunities[0]
        signal = best.signal
        
        print(f"\n{'='*70}")
        print(f"AUTO-TRADE: Executing {signal.action} on {signal.symbol}")
        print(f"{'='*70}")
        print(f"Position Size: {signal.recommended_position_pct:.0%}")
        print(f"Expected ROI: {signal.expected_roi:+.1%}")
        
        # TODO: Implement actual Polymarket API integration
        print("\n[TODO] Polymarket API integration not yet implemented")
        print("       This would place the trade on Polymarket")
        print("       For now, this is a simulation/display only")
        
        # Log the signal
        self.signal_history.append({
            'timestamp': datetime.now(),
            'symbol': signal.symbol,
            'action': signal.action,
            'edge': signal.edge,
            'confidence': signal.confidence,
            'position_pct': signal.recommended_position_pct
        })
    
    def _run_data_collection(self):
        """Run training data collection in background"""
        try:
            save_counter = 0
            while True:
                # Collect snapshots from all symbols
                for symbol in self.symbols:
                    snapshot = self.aggregator.get_snapshot(symbol)
                    if snapshot:
                        self.data_collector.snapshots[symbol].append(snapshot)
                
                # Auto-save every 5 minutes (10 cycles at 30s each)
                save_counter += 1
                if save_counter >= 10:
                    self.data_collector.label_data()
                    self.data_collector.save_all_data()
                    save_counter = 0
                
                time.sleep(self.update_interval)
        except Exception as e:
            print(f"\n[ERROR] Data collection thread failed: {e}")
    
    def _shutdown(self):
        """Gracefully shutdown the bot"""
        print("\n\n" + "="*70)
        print("SHUTTING DOWN QUANTUM TRADING BOT")
        print("="*70)
        
        # Save collected training data
        if self.enable_data_collection:
            print("\nSaving collected training data...")
            try:
                self.data_collector.label_data()
                self.data_collector.save_all_data()
                self.data_collector.get_collection_stats()
            except Exception as e:
                print(f"[ERROR] Could not save training data: {e}")
        
        # Stop data aggregator
        self.aggregator.stop()
        
        # Display statistics
        print(f"\nSession Statistics:")
        print(f"  Cycles Completed: {self.cycles_completed}")
        print(f"  Opportunities Found: {self.opportunities_found}")
        print(f"  Signals Generated: {len(self.signal_history)}")
        
        if self.signal_history:
            print(f"\n  Recent Signals:")
            for sig in self.signal_history[-5:]:  # Last 5
                print(f"    {sig['timestamp'].strftime('%H:%M:%S')} - {sig['action']} {sig['symbol']} "
                      f"(edge: {sig['edge']:+.1%}, conf: {sig['confidence']:.0%})")
        
        print("\n" + "="*70)
        print("Bot stopped successfully")
        print("="*70)
        sys.exit(0)

# Main execution
if __name__ == "__main__":
    print("""
    ======================================================================
                                                                  
         POLYMARKET QUANTUM PREDICTOR - TRADING BOT              
                                                                  
      Physics-Based + Machine Learning + Liquidation Analysis        
                                                                  
    ======================================================================
    """)
    
    # Configuration
    SYMBOLS = ['BTC', 'ETH', 'SOL', 'XRP']
    UPDATE_INTERVAL = 30  # seconds
    MIN_EDGE = 0.12  # 12% minimum edge
    MIN_CONFIDENCE = 0.65  # 65% minimum confidence
    AUTO_TRADE = False  # Set to True to enable automatic trading
    ENABLE_DATA_COLLECTION = True  # Set to False to disable training data collection
    
    # Create and start bot
    bot = QuantumTradingBot(
        symbols=SYMBOLS,
        update_interval=UPDATE_INTERVAL,
        min_edge=MIN_EDGE,
        min_confidence=MIN_CONFIDENCE,
        auto_trade=AUTO_TRADE,
        enable_data_collection=ENABLE_DATA_COLLECTION
    )
    
    bot.start()
