"""
Genetic Programming Service
Manages genetic programming strategy evolution and operations
"""

import logging
import asyncio
import pandas as pd
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from strategies.genetic_programming_strategy import GeneticProgrammingStrategy
from core.config import get_settings
from data.market_data_manager import MarketDataManager

logger = logging.getLogger(__name__)

class GeneticProgrammingService:
    """
    Service for managing genetic programming strategy evolution
    
    Features:
    - Evolve trading strategies on historical data
    - Manage evolved strategy lifecycle
    - Real-time strategy execution
    - Results tracking and persistence
    """
    
    def __init__(self, market_data_manager: Optional[MarketDataManager] = None):
        self.settings = get_settings()
        self.market_data_manager = market_data_manager
        
        # GP Strategy instance
        self.gp_strategy: Optional[GeneticProgrammingStrategy] = None
        
        # Evolution tracking
        self.evolution_history = []
        self.is_evolving = False
        self.current_evolution_task = None
        
        # Data management
        self.training_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        
        logger.info("🧬 Genetic Programming Service initialized")
    
    async def start(self):
        """Start the genetic programming service"""
        try:
            logger.info("🚀 Starting Genetic Programming Service...")
            
            # Initialize strategy with default config
            config = {
                'population_size': getattr(self.settings, 'GA_POPULATION_SIZE', 300),
                'generations': getattr(self.settings, 'GA_GENERATIONS', 40),
                'crossover_prob': getattr(self.settings, 'GA_CROSSOVER_RATE', 0.5),
                'mutation_prob': getattr(self.settings, 'GA_MUTATION_RATE', 0.2),
                'save_results': True,
                'save_models': True
            }
            
            self.gp_strategy = GeneticProgrammingStrategy(config)
            
            # Load existing evolved strategy if available
            await self.load_best_evolved_strategy()
            
            logger.info("✅ Genetic Programming Service started")
            
        except Exception as e:
            logger.error(f"❌ Error starting GP service: {e}")
    
    async def stop(self):
        """Stop the genetic programming service"""
        try:
            if self.current_evolution_task:
                self.current_evolution_task.cancel()
                logger.info("⏹️ Cancelled ongoing evolution")
            
            logger.info("🛑 Genetic Programming Service stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping GP service: {e}")
    
    async def prepare_training_data(self, symbol: str, timeframe: str = "1h", 
                                   days_back: int = 365) -> bool:
        """Prepare training data for evolution"""
        try:
            if not self.market_data_manager:
                logger.error("❌ Market data manager not available")
                return False
            
            logger.info(f"📊 Preparing training data for {symbol} ({timeframe}, {days_back} days)")
            
            # Get historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            data = await self.market_data_manager.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if data is None or len(data) < 100:
                logger.error(f"❌ Insufficient data for {symbol}")
                return False
            
            # Split data for training and validation
            split_idx = int(len(data) * 0.8)  # 80% training, 20% validation
            self.training_data = data.iloc[:split_idx].copy()
            self.validation_data = data.iloc[split_idx:].copy()
            
            logger.info(f"✅ Training data prepared: {len(self.training_data)} training samples, {len(self.validation_data)} validation samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {e}")
            return False
    
    async def evolve_strategy_async(self, symbol: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evolve a trading strategy asynchronously"""
        if self.is_evolving:
            return {
                'success': False,
                'message': 'Evolution already in progress'
            }
        
        try:
            self.is_evolving = True
            evolution_start = datetime.now()
            
            logger.info(f"🧬 Starting evolution for {symbol}")
            
            # Prepare data if not already done
            if self.training_data is None:
                success = await self.prepare_training_data(symbol)
                if not success:
                    return {
                        'success': False,
                        'message': 'Failed to prepare training data'
                    }
            
            # Update strategy config if provided
            if config and self.gp_strategy:
                for key, value in config.items():
                    if hasattr(self.gp_strategy.config, key):
                        setattr(self.gp_strategy.config, key, value)
            
            # Run evolution in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def run_evolution():
                return self.gp_strategy.evolve_strategy(self.training_data)
            
            # Execute evolution
            success = await loop.run_in_executor(None, run_evolution)
            
            evolution_time = datetime.now() - evolution_start
            
            # Record evolution results
            evolution_result = {
                'symbol': symbol,
                'success': success,
                'evolution_time': evolution_time.total_seconds(),
                'timestamp': evolution_start,
                'config': {
                    'population_size': self.gp_strategy.config.population_size,
                    'generations': self.gp_strategy.config.generations,
                    'crossover_prob': self.gp_strategy.config.crossover_prob,
                    'mutation_prob': self.gp_strategy.config.mutation_prob
                }
            }
            
            if success and self.gp_strategy.evolved_function:
                evolution_result.update({
                    'best_fitness': self.gp_strategy.evolved_function.fitness,
                    'strategy_expression': self.gp_strategy.evolved_function.individual_str,
                    'evolution_stats': self.gp_strategy.evolution_stats
                })
                
                # Validate on validation data if available
                if self.validation_data is not None:
                    validation_fitness = self.gp_strategy.simulate_trading(
                        self.gp_strategy.evolved_function.gp_function,
                        self.validation_data
                    )
                    evolution_result['validation_fitness'] = validation_fitness
                    logger.info(f"📊 Validation fitness: {validation_fitness:.4f}%")
            
            self.evolution_history.append(evolution_result)
            
            logger.info(f"{'✅' if success else '❌'} Evolution completed in {evolution_time.total_seconds():.1f}s")
            
            return {
                'success': success,
                'message': 'Evolution completed successfully' if success else 'Evolution failed',
                'result': evolution_result
            }
            
        except Exception as e:
            logger.error(f"❌ Error during evolution: {e}")
            return {
                'success': False,
                'message': f'Evolution error: {str(e)}'
            }
        finally:
            self.is_evolving = False
    
    def start_evolution(self, symbol: str, config: Optional[Dict[str, Any]] = None) -> asyncio.Task:
        """Start evolution as a background task"""
        if self.current_evolution_task and not self.current_evolution_task.done():
            logger.warning("⚠️ Evolution already running")
            return self.current_evolution_task
        
        self.current_evolution_task = asyncio.create_task(
            self.evolve_strategy_async(symbol, config)
        )
        return self.current_evolution_task
    
    async def load_evolved_strategy(self, model_path: str) -> bool:
        """Load a previously evolved strategy"""
        try:
            if not self.gp_strategy:
                self.gp_strategy = GeneticProgrammingStrategy()
            
            success = self.gp_strategy.load_evolved_strategy(model_path)
            if success:
                logger.info(f"✅ Loaded evolved strategy from {model_path}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error loading evolved strategy: {e}")
            return False
    
    async def load_best_evolved_strategy(self) -> bool:
        """Load the best available evolved strategy"""
        try:
            if not self.gp_strategy:
                return False
            
            results_dir = Path(self.gp_strategy.config.results_dir)
            if not results_dir.exists():
                logger.info("📁 No results directory found")
                return False
            
            # Find the most recent model file
            model_files = list(results_dir.glob("gp_model_*.pkl"))
            if not model_files:
                logger.info("🤖 No evolved models found")
                return False
            
            # Load the most recent model
            latest_model = max(model_files, key=os.path.getctime)
            return await self.load_evolved_strategy(str(latest_model))
            
        except Exception as e:
            logger.error(f"❌ Error loading best evolved strategy: {e}")
            return False
    
    def get_strategy_signal(self, data: pd.DataFrame):
        """Get trading signal from evolved strategy"""
        if not self.gp_strategy or not self.gp_strategy.is_evolved:
            return None
        
        return self.gp_strategy.generate_signal(data)
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        status = {
            'is_evolving': self.is_evolving,
            'has_evolved_strategy': self.gp_strategy.is_evolved if self.gp_strategy else False,
            'evolution_history_count': len(self.evolution_history),
            'training_data_available': self.training_data is not None,
            'validation_data_available': self.validation_data is not None
        }
        
        if self.gp_strategy and self.gp_strategy.is_evolved:
            status.update({
                'strategy_info': self.gp_strategy.get_strategy_info(),
                'evolved_fitness': self.gp_strategy.evolved_function.fitness,
                'evolution_time': self.gp_strategy.evolved_function.created_at.isoformat()
            })
        
        if self.current_evolution_task:
            status['current_task_done'] = self.current_evolution_task.done()
        
        return status
    
    def get_evolution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get evolution history"""
        return self.evolution_history[-limit:] if limit else self.evolution_history
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available evolved models"""
        try:
            if not self.gp_strategy:
                return []
            
            results_dir = Path(self.gp_strategy.config.results_dir)
            if not results_dir.exists():
                return []
            
            models = []
            model_files = list(results_dir.glob("gp_model_*.pkl"))
            
            for model_file in model_files:
                stat = model_file.stat()
                models.append({
                    'filename': model_file.name,
                    'path': str(model_file),
                    'size_bytes': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            # Sort by creation time, newest first
            models.sort(key=lambda x: x['created_at'], reverse=True)
            return models
            
        except Exception as e:
            logger.error(f"❌ Error getting available models: {e}")
            return []
    
    def clear_training_data(self):
        """Clear training and validation data"""
        self.training_data = None
        self.validation_data = None
        logger.info("🧹 Training data cleared")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            'service_name': 'Genetic Programming Service',
            'status': {
                'is_running': True,
                'is_evolving': self.is_evolving,
                'has_strategy': self.gp_strategy is not None,
                'has_evolved_strategy': self.gp_strategy.is_evolved if self.gp_strategy else False
            },
            'config': {
                'population_size': self.gp_strategy.config.population_size if self.gp_strategy else None,
                'generations': self.gp_strategy.config.generations if self.gp_strategy else None,
                'results_dir': self.gp_strategy.config.results_dir if self.gp_strategy else None
            },
            'data': {
                'training_samples': len(self.training_data) if self.training_data is not None else 0,
                'validation_samples': len(self.validation_data) if self.validation_data is not None else 0
            },
            'evolution_history_count': len(self.evolution_history),
            'available_models_count': len(self.get_available_models())
        } 