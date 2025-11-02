"""
🚀 Autonomous Trading System - Main API Gateway
Built with FastAPI for high-performance async operations
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import asdict
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import trading, strategies, data, portfolio, backtesting, analytics
from api.routes import advanced_analytics
from api.routes import moongpt, historical_data, yahoo_data, indicator_listing, ml_evaluation, indicator_prediction, feature_importance
from api.routes import tiktok_sentiment
from api.routes import ai_models
from api.routes import interactive_brokers
from api.routes import hyperliquid
from api.routes import whale_tracking
from api.routes import solana_tracker as solana_tracker_router
from autonomous_trading_system.backend.api.routes import liquidation_hunter
from core.config import get_settings
from core.database import init_db, close_db
from core.logging_config import setup_logging
from data.market_data_manager import MarketDataManager
from strategies.breakout_strategy import BreakoutStrategy
from strategies.liquidation_strategy import LiquidationStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.rrs_strategy import RRSStrategy
from strategies.indicator_optimizer import IndicatorOptimizerStrategy
from strategies.regime_detection import RegimeDetectionStrategy
from strategies.sentiment_strategy import SentimentStrategy
from strategies.genetic_optimizer import GeneticOptimizerStrategy
from strategies.quick_execution_strategy import QuickExecutionStrategy
from strategies.ma_reversal_strategy import MAReversalStrategy
from strategies.supply_demand_zone_strategy import SupplyDemandZoneStrategy
from strategies.market_making_strategy import MarketMakingStrategy
from services.binance_streams import liqs_stream, big_liqs_stream, funding_stream, huge_trades_stream, recent_trades_stream
from services.social_sentiment_service import SocialSentimentService
from services.advanced_dashboard_service import AdvancedDashboardService
from services.multi_exchange_service import MultiExchangeService
from services.phemex_service import PhemexService
from services.phemex_risk_service import PhemexRiskService
from services.hyperliquid_bot import HyperliquidBotService
from services.hyperliquid_risk_service import HyperliquidRiskService
from services.sma_trading_service import SMATradingService
from services.rsi_trading_service import RsiTradingService
from services.vwap_trading_service import VwapTradingService
from services.websocket_manager import WebSocketManager, BinanceLiquidationStream, SolanaTokenScanner, PriceStreamManager
from services.order_manager import OrderManager, PositionManager
from services.backtesting_engine import BacktestingEngine
from services.token_security_analyzer import TokenSecurityAnalyzer
from services.wallet_tracker import WalletTracker
from services.strategy_engine import StrategyEngine
from services.portfolio_manager import PortfolioManager
from services.risk_manager import RiskManager
from services.indicator_service import IndicatorService
from services.bot1_trading_service import Bot1TradingService
from services.bollinger_trading_service import BollingerTradingService
from services.breakout_trading_service import BreakoutTradingService
from services.supply_demand_trading_service import SupplyDemandTradingService
from services.engulfing_trading_service import EngulfingTradingService
from services.vwap_probability_trading_service import VwapProbabilityTradingService
from services.stochrsi_trading_service import StochRSITradingService
from services.gap_trading_service import EnhancedEmaTradingService
from services.hyperliquid_breakout_service import HyperliquidBreakoutService
from services.liquidation_data_processor import start_liquidation_data_processor, stop_liquidation_data_processor, get_processor_status, process_liquidation_file
from services.liquidation_backtesting_service import (
    get_liquidation_backtest_status,
    KALMAN_AVAILABLE,
    LiquidationBacktestingService
)
from services.trading_hours_filter_service import (
    TradingHoursDataFilterService,
    SeasonalFilterConfig,
    filter_seasonal_single_file,
)
from services.manual_trading_service import (
    ManualTradingService,
    TradeRequest,
    TradeAction,
    execute_manual_buy,
    execute_manual_sell
)
from services.solana_trading_utils_service import SolanaTradingUtilsService
from services.solana_token_scanner_service import SolanaTokenScannerService
from services.twitter_sentiment_collector import TwitterSentimentCollector
from services.hyperliquid_rrs_analysis_service import hyperliquid_rrs_service
from services.hyperliquid_rrs_bot_service import hyperliquid_rrs_bot_service
from services.hyperliquid_utils_service import hyperliquid_utils_service
from services.early_buyer_tracker_service import early_buyer_tracker_service
from services.tiktok_sentiment_service import TikTokSentimentService
from services.whale_tracking_service import WhaleTrackingService
from services.hyperliquid_service import HyperliquidService
from services.liquidation_hunter_service import LiquidationHunterService
from services.hyperliquid_whale_service import HyperliquidWhaleService
from services.solana_token_tracker_service import SolanaTokenTrackerService
from services.solana_jupiter_service import SolanaJupiterService
from services.ez_bot_service import EZBotService
from services.quick_buy_sell_service import QuickBuySellService
from models.model_factory import get_model_factory

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

# Global instances
strategy_engine: Optional[StrategyEngine] = None
portfolio_manager: Optional[PortfolioManager] = None
risk_manager: Optional[RiskManager] = None
market_data_manager: Optional[MarketDataManager] = None
websocket_manager = None
order_manager = None
position_manager = None
backtesting_engine = None
token_security_analyzer = None
wallet_tracker = None
tiktok_sentiment_service = None
model_factory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global strategy_engine, portfolio_manager, risk_manager, market_data_manager, websocket_manager, order_manager, position_manager, backtesting_engine, token_security_analyzer, wallet_tracker, tiktok_sentiment_service, model_factory
    
    logger.info("🚀 Starting Autonomous Trading System...")
    
    try:
        # Initialize database
        await init_db()
        
        # Initialize core services
        settings = get_settings()
        
        # Initialize AI Model Factory
        logger.info("🤖 Initializing AI Model Factory...")
        model_factory = get_model_factory()
        app.state.model_factory = model_factory
        logger.info(f"🤖 AI Model Factory initialized. Available models: {model_factory.available_models}")
        
        # Initialize managers
        risk_manager = RiskManager(settings)
        portfolio_manager = PortfolioManager(settings, risk_manager)
        market_data_manager = MarketDataManager(settings)
        
        # Initialize strategy engine
        strategy_engine = StrategyEngine(
            market_data_manager=market_data_manager,
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            config=settings
        )
        
        # Initialize new services
        websocket_manager = WebSocketManager()
        order_manager = OrderManager(strategy_engine)
        position_manager = PositionManager(order_manager)
        backtesting_engine = BacktestingEngine()
        token_security_analyzer = TokenSecurityAnalyzer(settings)
        wallet_tracker = WalletTracker(settings)
        
        # Start WebSocket manager
        await websocket_manager.start()
        
        # Initialize liquidation stream if enabled
        if settings.LIQUIDATION_STRATEGY_ENABLED:
            liquidation_stream = BinanceLiquidationStream(websocket_manager)
            asyncio.create_task(liquidation_stream.start_stream())
            logger.info("📊 Liquidation stream started")
        
        # Initialize token scanner if enabled
        if settings.TOKEN_SCANNER_ENABLED and settings.HELIUS_API_KEY:
            # Use enhanced token scanner if enabled, otherwise use basic scanner
            if getattr(settings, 'ENABLE_ENHANCED_TOKEN_SCANNER', True):
                enhanced_token_scanner = SolanaTokenScannerService(websocket_manager, settings)
                asyncio.create_task(enhanced_token_scanner.start())
                app.state.enhanced_token_scanner = enhanced_token_scanner
                logger.info("🔍 Enhanced Solana Token Scanner started")
            else:
                token_scanner = SolanaTokenScanner(websocket_manager)
                asyncio.create_task(token_scanner.start_scanner())
                logger.info("🔍 Basic token scanner started")
        
        # Initialize Twitter sentiment collector if enabled
        if getattr(settings, 'ENABLE_TWITTER_SENTIMENT_COLLECTOR', True):
            twitter_collector = TwitterSentimentCollector(settings)
            if twitter_collector.enabled:
                asyncio.create_task(twitter_collector.start())
                app.state.twitter_collector = twitter_collector
                logger.info("🐦 Enhanced Twitter Sentiment Collector started")
            else:
                logger.info("🐦 Twitter Sentiment Collector disabled (missing credentials)")
        
        # Initialize TikTok sentiment service if enabled
        if getattr(settings, 'TIKTOK_SENTIMENT_ENABLED', True):
            tiktok_sentiment_service = TikTokSentimentService(settings)
            if tiktok_sentiment_service.enabled:
                asyncio.create_task(tiktok_sentiment_service.start())
                app.state.tiktok_sentiment_service = tiktok_sentiment_service
                logger.info("📹 TikTok Sentiment Service started")
            else:
                logger.info("📹 TikTok Sentiment Service disabled (missing dependencies)")
        
        # Initialize Solana Token Tracker service if enabled
        if getattr(settings, 'ENABLE_SOLANA_TOKEN_TRACKER', False):
            solana_token_tracker = SolanaTokenTrackerService()
            asyncio.create_task(solana_token_tracker.run_as_background_task())
            app.state.solana_token_tracker = solana_token_tracker
            logger.info("🌙 Solana Token Tracker Service started")
        
        # Initialize Hyperliquid whale scraper service if enabled
        if getattr(settings, 'ENABLE_WHALE_SCRAPER', False):
            whale_scraper_service = HyperliquidWhaleService(settings.model_dump())
            asyncio.create_task(whale_scraper_service.start())
            app.state.whale_scraper_service = whale_scraper_service
            logger.info("🐳 Hyperliquid Whale Scraper Service started")
        
        # Initialize Day 48 Services
        if getattr(settings, 'ENABLE_EZ_BOT_SERVICE', False):
            logger.info("Initializing Day 48 services: SolanaJupiterService and EZBotService...")
            
            # Initialize Jupiter Service (prerequisite for EZBot)
            jupiter_service = SolanaJupiterService(settings=settings)
            app.state.jupiter_service = jupiter_service
            logger.info("🤖 Solana Jupiter Service initialized.")

            # Initialize EZBot Service
            ez_bot_service = EZBotService(settings=settings, jupiter_service=jupiter_service)
            asyncio.create_task(ez_bot_service.start())
            app.state.ez_bot_service = ez_bot_service
            logger.info("🤖 EZ Bot Service started.")
        else:
            logger.info("🤖 EZ Bot Service is disabled in settings.")
        
        # Initialize Day 51 Quick Buy/Sell Service
        if getattr(settings, 'ENABLE_QUICK_BUY_SELL_BOT', False):
            logger.info("Initializing Day 51 Quick Buy/Sell Service...")
            qbs_service = QuickBuySellService(settings=settings)
            asyncio.create_task(qbs_service.start_monitoring())
            app.state.quick_buy_sell_service = qbs_service
            logger.info("⚡️ Quick Buy/Sell Service started.")
        else:
            logger.info("⚡️ Quick Buy/Sell Service is disabled in settings.")
        
        # Initialize price streams
        price_manager = PriceStreamManager(websocket_manager)

        # Initialize Day 2 Binance streams
        if settings.LIQS_STREAM_ENABLED:
            asyncio.create_task(liqs_stream())
            logger.info("📉 General liquidation stream started")
        if settings.BIG_LIQS_STREAM_ENABLED:
            asyncio.create_task(big_liqs_stream())
            logger.info("📈 Big liquidation stream started")
        if settings.FUNDING_STREAM_ENABLED:
            asyncio.create_task(funding_stream())
            logger.info("💰 Funding rate stream started")
        if settings.HUGE_TRADES_STREAM_ENABLED:
            asyncio.create_task(huge_trades_stream())
            logger.info("💹 Huge trades stream started")
        if settings.RECENT_TRADES_STREAM_ENABLED:
            asyncio.create_task(recent_trades_stream())
            logger.info("🕒 Recent trades stream started")
        
        # Initialize Indicator Service if enabled
        if settings.ENABLE_INDICATOR_SERVICE:
            indicator_service = IndicatorService(market_data_manager)
            asyncio.create_task(indicator_service.start())
            logger.info("🚀 Indicator service started")
            app.state.indicator_service = indicator_service
        
        # Initialize Phemex trading bot if enabled
        if settings.ENABLE_PHEMEX_BOT:
            phemex_service = PhemexService(settings)
            asyncio.create_task(phemex_service.start())
            logger.info("🤖 Phemex bot service started")
        
        # Initialize Hyperliquid trading bot if enabled
        if settings.ENABLE_HYPERLIQUID_BOT:
            hyperliquid_bot = HyperliquidBotService()
            asyncio.create_task(hyperliquid_bot.start())
            logger.info("🤖 Hyperliquid bot service started")
        
        # Initialize Hyperliquid breakout bot if enabled
        if getattr(settings, 'ENABLE_HYPERLIQUID_BREAKOUT_BOT', False):
            breakout_service = HyperliquidBreakoutService()
            asyncio.create_task(breakout_service.start())
            logger.info("🤖 Hyperliquid breakout bot started")
        
        # Initialize services for Liquidation Hunter
        logger.info("Initializing Hyperliquid and Whale Tracking services for the Hunter...")
        hyperliquid_service = HyperliquidService(settings)
        app.state.hyperliquid_service = hyperliquid_service
        
        whale_tracking_service = WhaleTrackingService(settings)
        app.state.whale_tracking_service = whale_tracking_service
        logger.info("Hyperliquid and Whale Tracking services initialized.")

        # Initialize Liquidation Hunter if enabled
        if settings.ENABLE_LIQUIDATION_HUNTER:
            liquidation_hunter_service = LiquidationHunterService(
                config=settings.dict(), 
                hyperliquid_service=hyperliquid_service,
                whale_tracking_service=whale_tracking_service
            )
            app.state.liquidation_hunter_service = liquidation_hunter_service
            asyncio.create_task(liquidation_hunter_service.start())
            logger.info("🤖 Liquidation Hunter service started with advanced analysis.")
        
        # Initialize Phemex risk manager if enabled
        if settings.ENABLE_PHEMEX_RISK:
            phemex_risk = PhemexRiskService(settings)
            asyncio.create_task(phemex_risk.start())
            logger.info("🔒 Phemex risk service started")
        # Initialize Hyperliquid risk manager if enabled
        if settings.ENABLE_HYPERLIQUID_RISK:
            hl_risk = HyperliquidRiskService(settings)
            asyncio.create_task(hl_risk.start())
            logger.info("🔒 Hyperliquid risk service started")
        # Initialize native SMA trading service if enabled
        if settings.ENABLE_SMA_BOT:
            sma_service = SMATradingService()
            asyncio.create_task(sma_service.start())
            logger.info("🚀 SMA trading service started")
        # Initialize native RSI trading service if enabled
        if settings.ENABLE_RSI_BOT:
            rsi_service = RsiTradingService()
            asyncio.create_task(rsi_service.start())
            logger.info("🚀 RSI trading service started")
        # Initialize native VWAP trading service if enabled
        if settings.ENABLE_VWAP_BOT:
            vwap_service = VwapTradingService()
            asyncio.create_task(vwap_service.start())
            logger.info("🚀 VWAP trading service started")
            # Initialize Day 10 Bot1 trading service if enabled
            if settings.ENABLE_BOT1:
                bot1_service = Bot1TradingService()
                asyncio.create_task(bot1_service.start())
                logger.info("🤖 Bot1 trading service started")
                app.state.bot1_service = bot1_service
            # Initialize Day 10 Bollinger trading service if enabled
            if settings.ENABLE_BOLLINGER_BOT:
                boll_service = BollingerTradingService()
                asyncio.create_task(boll_service.start())
                logger.info("🤖 Bollinger trading service started")
                app.state.bollinger_service = boll_service
            # Initialize Day 11 Breakout trading service if enabled
            if settings.ENABLE_BREAKOUT_BOT:
                breakout_service = BreakoutTradingService()
                asyncio.create_task(breakout_service.start())
                logger.info("🤖 Breakout trading service started")
                app.state.breakout_service = breakout_service
            # Initialize Day 11 Supply/Demand trading service if enabled
            if settings.ENABLE_SDZ_BOT:
                sdz_service = SupplyDemandTradingService()
                asyncio.create_task(sdz_service.start())
                logger.info("🤖 Supply/Demand trading service started")
                app.state.sdz_service = sdz_service
            # Initialize Day 12 Engulfing Candle trading service if enabled
            if settings.ENABLE_ENGULFING_BOT:
                eng_service = EngulfingTradingService()
                asyncio.create_task(eng_service.start())
                logger.info("🤖 Engulfing Candle trading service started")
                app.state.engulfing_service = eng_service
            # Initialize Day 12 VWAP Probability trading service if enabled
            if settings.ENABLE_VWAP_PROBABILITY_BOT:
                vwap_prob_service = VwapProbabilityTradingService()
                asyncio.create_task(vwap_prob_service.start())
                logger.info("🤖 VWAP Probability trading service started")
                app.state.vwap_probability_service = vwap_prob_service

            # Initialize Day 16 StochRSI trading service if enabled
            if settings.ENABLE_STOCHRSI_BOT:
                stochrsi_service = StochRSITradingService()
                asyncio.create_task(stochrsi_service.start())
                logger.info("🤖 StochRSI trading service started")
                app.state.stochrsi_service = stochrsi_service
            # Initialize Day 17 Enhanced EMA (Gap) trading service if enabled
            if settings.ENABLE_ENHANCED_EMA_BOT:
                enhanced_ema_service = EnhancedEmaTradingService()
                asyncio.create_task(enhanced_ema_service.start())
                logger.info("🤖 Enhanced EMA trading service started")
                app.state.enhanced_ema_service = enhanced_ema_service
        
        # Initialize Day 21 Liquidation Data Processor if enabled
        if settings.ENABLE_LIQUIDATION_DATA_PROCESSOR:
            asyncio.create_task(start_liquidation_data_processor())
            logger.info("📊 Liquidation data processor started")
        
        # Initialize liquidation backtesting service
        liquidation_backtesting_service = LiquidationBacktestingService()
        app.state.liquidation_backtesting_service = liquidation_backtesting_service
        logger.info("📊 Liquidation backtesting service initialized")
        
        # Initialize trading hours filter service
        if settings.ENABLE_TRADING_HOURS_FILTER:
            trading_hours_filter_service = TradingHoursDataFilterService(settings.TRADING_HOURS_FILTER_DATA_FOLDER)
            app.state.trading_hours_filter_service = trading_hours_filter_service
            logger.info("🕒 Trading hours data filter service initialized")
        
        # Initialize manual trading service
        if settings.ENABLE_MANUAL_TRADING:
            manual_trading_config = {
                'max_usd_order_size': settings.MANUAL_TRADING_MAX_USD_ORDER_SIZE,
                'orders_per_burst': settings.MANUAL_TRADING_ORDERS_PER_BURST,
                'tx_sleep_seconds': settings.MANUAL_TRADING_TX_SLEEP_SECONDS,
                'slippage_bps': settings.MANUAL_TRADING_SLIPPAGE_BPS,
                'position_close_threshold_usd': settings.MANUAL_TRADING_POSITION_CLOSE_THRESHOLD_USD,
                'target_reach_tolerance': settings.MANUAL_TRADING_TARGET_REACH_TOLERANCE,
                'max_retries': settings.MANUAL_TRADING_MAX_RETRIES,
                'retry_delay_multiplier': settings.MANUAL_TRADING_RETRY_DELAY_MULTIPLIER
            }
            manual_trading_service = ManualTradingService(
                market_data_manager=market_data_manager,
                order_manager=order_manager,
                config=manual_trading_config
            )
            asyncio.create_task(manual_trading_service.start())
            app.state.manual_trading_service = manual_trading_service
            logger.info("📋 Manual trading service initialized")
        
        # Initialize Solana Trading Utilities Service
        if settings.ENABLE_SOLANA_TRADING_UTILS:
            solana_config = {
                'birdeye_api_key': settings.SOLANA_BIRDEYE_API_KEY,
                'openai_api_key': settings.SOLANA_OPENAI_API_KEY,
                'solana_rpc_endpoint': settings.SOLANA_RPC_ENDPOINT,
                'wallet_secret_key': settings.SOLANA_WALLET_SECRET_KEY,
                'wallet_address': settings.SOLANA_WALLET_ADDRESS,
                'minimum_trades_in_last_hour': settings.SOLANA_MINIMUM_TRADES_IN_LAST_HOUR,
                'stop_loss_percentage': settings.SOLANA_STOP_LOSS_PERCENTAGE,
                'sell_at_multiple': settings.SOLANA_SELL_AT_MULTIPLE,
                'priority_fee': settings.SOLANA_PRIORITY_FEE,
                'do_not_trade_list': settings.SOLANA_DO_NOT_TRADE_LIST,
                'max_position_size_usd': settings.SOLANA_MAX_POSITION_SIZE_USD,
                'max_daily_trades': settings.SOLANA_MAX_DAILY_TRADES,
                'enable_ai_decisions': settings.SOLANA_ENABLE_AI_DECISIONS,
                'enable_meme_scoring': settings.SOLANA_ENABLE_MEME_SCORING,
                'cache_ttl': settings.SOLANA_PRICE_CACHE_TTL_SECONDS
            }
            solana_trading_utils_service = SolanaTradingUtilsService(solana_config)
            asyncio.create_task(solana_trading_utils_service.start())
            app.state.solana_trading_utils_service = solana_trading_utils_service
            logger.info("🔧 Solana Trading Utilities Service initialized")
        
        # Register strategies
        await _register_strategies()
        
        # Set up WebSocket event handlers
        await _setup_websocket_handlers()
        
        # Start strategy engine
        await strategy_engine.start()
        
        # Store in app state
        app.state.strategy_engine = strategy_engine
        app.state.portfolio_manager = portfolio_manager
        app.state.risk_manager = risk_manager
        app.state.market_data_manager = market_data_manager
        app.state.websocket_manager = websocket_manager
        app.state.order_manager = order_manager
        app.state.position_manager = position_manager
        app.state.backtesting_engine = backtesting_engine
        app.state.token_security_analyzer = token_security_analyzer
        app.state.wallet_tracker = wallet_tracker
        
        # Initialize new services
        app.state.social_sentiment_service = SocialSentimentService(settings)
        app.state.multi_exchange_service = MultiExchangeService(settings)
        
        # Initialize liquidation backtesting service
        app.state.liquidation_backtesting_service = LiquidationBacktestingService()
        logger.info("📊 Liquidation backtesting service initialized")
        
        # Initialize trading hours filter service
        if settings.ENABLE_TRADING_HOURS_FILTER:
            trading_hours_filter_service = TradingHoursDataFilterService(settings.TRADING_HOURS_FILTER_DATA_FOLDER)
            app.state.trading_hours_filter_service = trading_hours_filter_service
            logger.info("🕒 Trading hours data filter service initialized")
        
        # Initialize advanced dashboard (needs strategy engine)
        app.state.advanced_dashboard_service = AdvancedDashboardService(
            settings,
            strategy_engine,
            portfolio_manager,
            risk_manager,
            market_data_manager
        )
        
        # Start services
        await start_services(app.state)
        
        logger.info("✅ All services started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🛑 Shutting down services...")
        
        # Stop liquidation data processor if enabled
        if settings.ENABLE_LIQUIDATION_DATA_PROCESSOR:
            await stop_liquidation_data_processor()
            logger.info("📊 Liquidation data processor stopped")
        
        if strategy_engine:
            await strategy_engine.stop()
        if portfolio_manager:
            await portfolio_manager.stop()
        
        if websocket_manager:
            await websocket_manager.stop()
        
        if tiktok_sentiment_service:
            await tiktok_sentiment_service.stop()
        
        await close_db()
        logger.info("✅ Shutdown complete")


async def _register_strategies():
    """Register all available strategies"""
    try:
        # Register existing strategies
        await strategy_engine.register_strategy("mean_reversion", MeanReversionStrategy)
        await strategy_engine.register_strategy("rrs", RRSStrategy)
        await strategy_engine.register_strategy("indicator_optimizer", IndicatorOptimizerStrategy)
        await strategy_engine.register_strategy("regime_detection", RegimeDetectionStrategy)
        await strategy_engine.register_strategy("sentiment", SentimentStrategy)
        await strategy_engine.register_strategy("genetic_optimizer", GeneticOptimizerStrategy)
        
        # Register new strategies
        await strategy_engine.register_strategy("breakout", BreakoutStrategy)
        await strategy_engine.register_strategy("liquidation", LiquidationStrategy)
        await strategy_engine.register_strategy("quick_execution", QuickExecutionStrategy)
        await strategy_engine.register_strategy("ma_reversal", MAReversalStrategy)
        await strategy_engine.register_strategy("supply_demand_zone", SupplyDemandZoneStrategy)
        
        # Register Day 49 MA Reversal Strategy V2
        if settings.ENABLE_MA_REVERSAL_V2_STRATEGY:
            from strategies.ma_reversal_strategy_v2 import MAReversalStrategyV2
            await strategy_engine.register_strategy("ma_reversal_v2", MAReversalStrategyV2)
        
        # Register new Market Making strategy
        if settings.MARKET_MAKING_ENABLED:
            await strategy_engine.register_strategy(
                "market_making",
                MarketMakingStrategy,
                symbols=settings.MARKET_MAKING_SYMBOLS,
                config=settings
            )
        
        logger.info("📋 All strategies registered")
        
    except Exception as e:
        logger.error(f"❌ Error registering strategies: {e}")


async def _setup_websocket_handlers():
    """Set up WebSocket event handlers"""
    try:
        # Liquidation event handler
        async def handle_liquidation(data):
            # Process liquidation for all liquidation strategies
            for strategy_name, strategy in strategy_engine.strategies.items():
                if isinstance(strategy, LiquidationStrategy):
                    await strategy.process_liquidation_event(data)
        
        # New token event handler
        async def handle_new_token(data):
            logger.info(f"🪙 New token discovered: {data['address']}")
            # Add logic to evaluate new tokens
        
        # Price update handler
        async def handle_price_update(data):
            # Update strategy engine with new price data
            await strategy_engine.update_price(data['symbol'], data['price'])
        
        # Register handlers
        websocket_manager.subscribe('liquidation', handle_liquidation)
        websocket_manager.subscribe('new_token', handle_new_token)
        websocket_manager.subscribe('price_update', handle_price_update)
        
        logger.info("🔗 WebSocket handlers configured")
        
    except Exception as e:
        logger.error(f"❌ Error setting up WebSocket handlers: {e}")


# Create FastAPI app
app = FastAPI(
    title="Autonomous Trading System API",
    description="A comprehensive trading system with multiple strategies and exchange integrations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency functions
async def get_strategy_engine() -> StrategyEngine:
    if not strategy_engine:
        raise HTTPException(status_code=503, detail="Strategy engine not initialized")
    return strategy_engine


async def get_portfolio_manager() -> PortfolioManager:
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio manager not initialized")
    return portfolio_manager


async def get_risk_manager() -> RiskManager:
    if not risk_manager:
        raise HTTPException(status_code=503, detail="Risk manager not initialized")
    return risk_manager


# Override the dependency functions in route modules
strategies.get_strategy_engine = get_strategy_engine
trading.get_strategy_engine = get_strategy_engine
trading.get_portfolio_manager = get_portfolio_manager
trading.get_risk_manager = get_risk_manager
portfolio.get_portfolio_manager = get_portfolio_manager
portfolio.get_risk_manager = get_risk_manager


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "strategy_engine": strategy_engine.is_running if strategy_engine else False,
            "portfolio_manager": portfolio_manager.is_running if portfolio_manager else False,
            "risk_manager": risk_manager.is_active if risk_manager else False,
        }
    }


# System status endpoint
@app.get("/status")
async def system_status():
    """Get detailed system status"""
    if not all([strategy_engine, portfolio_manager, risk_manager]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    return {
        "system": {
            "uptime": strategy_engine.uptime,
            "active_strategies": len(strategy_engine.active_strategies),
            "total_strategies": len(strategy_engine.available_strategies),
        },
        "portfolio": {
            "total_value": await portfolio_manager.get_total_value(),
            "open_positions": len(await portfolio_manager.get_open_positions()),
            "daily_pnl": await portfolio_manager.get_daily_pnl(),
        },
        "risk": {
            "max_drawdown": await risk_manager.get_max_drawdown(),
            "current_exposure": await risk_manager.get_current_exposure(),
            "risk_score": await risk_manager.get_risk_score(),
        }
    }


# Emergency stop endpoint
@app.post("/emergency-stop")
async def emergency_stop():
    """Emergency stop all trading activities"""
    logger.warning("🚨 EMERGENCY STOP TRIGGERED")
    
    if strategy_engine:
        await strategy_engine.emergency_stop()
    if portfolio_manager:
        await portfolio_manager.close_all_positions()
    
    return {"status": "emergency_stop_executed", "timestamp": datetime.utcnow().isoformat()}


# Include API routes
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["strategies"])
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(backtesting.router, prefix="/api/v1/backtesting", tags=["backtesting"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(advanced_analytics.router)

# Include additional routers from Day 30-33 projects
app.include_router(moongpt.router, prefix="/api/v1/moongpt", tags=["moongpt"])
app.include_router(historical_data.router, prefix="/api/v1/historical-data", tags=["historical_data"])
app.include_router(yahoo_data.router, prefix="/api/v1/yahoo-data", tags=["yahoo_data"])
app.include_router(indicator_listing.router, prefix="/api/v1/indicators", tags=["indicators"])
app.include_router(ml_evaluation.router, prefix="/api/v1/ml-evaluation", tags=["ML Evaluation"])
app.include_router(indicator_prediction.router, prefix="/api/v1/indicator-prediction", tags=["Indicator Prediction"])
app.include_router(feature_importance.router, prefix="/api/v1/feature-importance", tags=["Feature Importance"])
app.include_router(tiktok_sentiment.router, prefix="/api/v1/tiktok", tags=["TikTok Sentiment"])
app.include_router(ai_models.router, prefix="/api/v1/ai-models", tags=["AI Models"])
app.include_router(interactive_brokers.router, prefix="/api/v1/ib", tags=["Interactive Brokers"])
app.include_router(hyperliquid.router, prefix="/api/v1/hyperliquid", tags=["Hyperliquid"])
app.include_router(whale_tracking.router, prefix="/api/v1/whales", tags=["Whale Tracking"])
app.include_router(liquidation_hunter.router, prefix="/api/v1/liquidation-hunter", tags=["Liquidation Hunter"])
app.include_router(solana_tracker_router.router, prefix=settings.API_PREFIX, tags=["Solana Token Tracker"])

# Include the new EZ Bot router
from api.routes import ez_bot as ez_bot_router
app.include_router(ez_bot_router.router, prefix="/api/v1/ez-bot", tags=["EZ Bot"])

# Include the new MA Reversal V2 router
from api.routes import ma_reversal_v2 as ma_reversal_v2_router
app.include_router(ma_reversal_v2_router.router, prefix="/api/v1", tags=["MA Reversal Strategy V2"])

# Include the new Quick Buy/Sell router
from api.routes import quick_buy_sell as qbs_router
app.include_router(qbs_router.router, prefix="/api/v1/quick-buy-sell", tags=["Quick Buy/Sell Bot"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )


# Add new API routes
@app.get("/api/liquidations/{symbol}")
async def get_liquidation_summary(symbol: str):
    """Get liquidation summary for a symbol"""
    try:
        for strategy_name, strategy in strategy_engine.strategies.items():
            if isinstance(strategy, LiquidationStrategy):
                summary = strategy.get_liquidation_summary(symbol)
                return {"symbol": symbol, "liquidation_summary": summary}
        
        return {"error": "No liquidation strategy found"}
        
    except Exception as e:
        logger.error(f"Error getting liquidation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest(request: dict):
    """Run a backtest"""
    try:
        strategy_name = request.get("strategy")
        parameters = request.get("parameters", {})
        start_date = request.get("start_date")
        end_date = request.get("end_date")
        
        # Get strategy class
        strategy_class = None
        if strategy_name == "mean_reversion":
            strategy_class = MeanReversionStrategy
        elif strategy_name == "breakout":
            strategy_class = BreakoutStrategy
        elif strategy_name == "liquidation":
            strategy_class = LiquidationStrategy
        # Add more strategies as needed
        
        if not strategy_class:
            raise HTTPException(status_code=400, detail="Invalid strategy name")
        
        # For demo purposes, create sample data
        # In production, this would fetch real historical data
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.random.rand(len(dates)) * 0.02),
            'low': prices * (1 - np.random.rand(len(dates)) * 0.02),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Run backtest
        result = await backtesting_engine.run_backtest(
            strategy_class, data, parameters
        )
        
        return {
            "strategy_name": result.strategy_name,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/orders")
async def create_order(request: dict):
    """Create a new order"""
    try:
        from services.order_manager import OrderSide, OrderType
        
        symbol = request.get("symbol")
        side = OrderSide(request.get("side"))
        order_type = OrderType(request.get("type"))
        quantity = float(request.get("quantity"))
        price = request.get("price")
        
        order = await order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        
        return {
            "order_id": order.id,
            "status": order.status.value,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": order.price
        }
        
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/orders")
async def get_active_orders(symbol: str = None):
    """Get active orders"""
    try:
        orders = await order_manager.get_active_orders(symbol)
        
        return {
            "orders": [
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "type": order.order_type.value,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status.value,
                    "created_at": order.created_at.isoformat()
                }
                for order in orders
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    try:
        success = await order_manager.cancel_order(order_id)
        
        return {"success": success, "order_id": order_id}
        
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions(symbol: str = None):
    """Get current positions"""
    try:
        if symbol:
            position = await position_manager.get_position(symbol)
            return {"position": position}
        else:
            # Return all positions
            positions = {}
            for sym in position_manager.positions.keys():
                positions[sym] = await position_manager.get_position(sym)
            return {"positions": positions}
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str, percentage: float = 100.0):
    """Close a position"""
    try:
        success = await position_manager.close_position(symbol, percentage)
        
        return {"success": success, "symbol": symbol, "percentage": percentage}
        
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get sentiment data for a symbol"""
    try:
        if not hasattr(app.state, 'social_sentiment_service'):
            raise HTTPException(status_code=503, detail="Social sentiment service not available")
        
        sentiment_data = await app.state.social_sentiment_service.get_symbol_sentiment(symbol.upper())
        
        if not sentiment_data:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol}")
        
        return sentiment_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/sentiment/trending")
async def get_trending_tokens():
    """Get trending tokens based on sentiment"""
    try:
        if not hasattr(app.state, 'social_sentiment_service'):
            raise HTTPException(status_code=503, detail="Social sentiment service not available")
        
        trending_tokens = await app.state.social_sentiment_service.get_trending_tokens()
        
        return {
            "trending_tokens": [
                {
                    "symbol": token.symbol,
                    "sentiment_score": token.sentiment_score,
                    "mention_count": token.mention_count,
                    "engagement_rate": token.engagement_rate,
                    "platform": token.platform,
                    "first_seen": token.first_seen.isoformat(),
                    "keywords": token.keywords
                }
                for token in trending_tokens
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting trending tokens: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/sentiment/signals")
async def get_sentiment_signals():
    """Get trading signals based on sentiment analysis"""
    try:
        if not hasattr(app.state, 'social_sentiment_service'):
            raise HTTPException(status_code=503, detail="Social sentiment service not available")
        
        signals = await app.state.social_sentiment_service.get_sentiment_signals()
        
        return {"signals": signals}
        
    except Exception as e:
        logger.error(f"❌ Error getting sentiment signals: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/dashboard/overview")
async def get_dashboard_overview():
    """Get complete dashboard overview"""
    try:
        if not hasattr(app.state, 'advanced_dashboard_service'):
            raise HTTPException(status_code=503, detail="Dashboard service not available")
        
        overview = await app.state.advanced_dashboard_service.get_dashboard_overview()
        
        return overview
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/dashboard/portfolio")
async def get_portfolio_summary():
    """Get portfolio summary"""
    try:
        if not hasattr(app.state, 'advanced_dashboard_service'):
            raise HTTPException(status_code=503, detail="Dashboard service not available")
        
        portfolio = await app.state.advanced_dashboard_service.get_portfolio_summary()
        
        return portfolio
        
    except Exception as e:
        logger.error(f"❌ Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/dashboard/strategies")
async def get_strategy_performance():
    """Get strategy performance summary"""
    try:
        if not hasattr(app.state, 'advanced_dashboard_service'):
            raise HTTPException(status_code=503, detail="Dashboard service not available")
        
        strategies = await app.state.advanced_dashboard_service.get_strategy_summary()
        
        return {"strategies": strategies}
        
    except Exception as e:
        logger.error(f"❌ Error getting strategy performance: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/dashboard/tokens")
async def get_token_overview():
    """Get token metrics overview"""
    try:
        if not hasattr(app.state, 'advanced_dashboard_service'):
            raise HTTPException(status_code=503, detail="Dashboard service not available")
        
        tokens = await app.state.advanced_dashboard_service.get_token_overview()
        
        return {"tokens": tokens}
        
    except Exception as e:
        logger.error(f"❌ Error getting token overview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/dashboard/alerts")
async def get_recent_alerts(limit: int = 20):
    """Get recent trading alerts"""
    try:
        if not hasattr(app.state, 'advanced_dashboard_service'):
            raise HTTPException(status_code=503, detail="Dashboard service not available")
        
        alerts = await app.state.advanced_dashboard_service.get_recent_alerts(limit)
        
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"❌ Error getting recent alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/dashboard/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge a trading alert"""
    try:
        if not hasattr(app.state, 'advanced_dashboard_service'):
            raise HTTPException(status_code=503, detail="Dashboard service not available")
        
        success = await app.state.advanced_dashboard_service.acknowledge_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/exchanges/prices")
async def get_aggregated_prices():
    """Get aggregated prices from all exchanges"""
    try:
        if not hasattr(app.state, 'multi_exchange_service'):
            raise HTTPException(status_code=503, detail="Multi-exchange service not available")
        
        prices = await app.state.multi_exchange_service.get_aggregated_prices()
        
        return {"prices": prices}
        
    except Exception as e:
        logger.error(f"❌ Error getting aggregated prices: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/exchanges/arbitrage")
async def get_arbitrage_opportunities():
    """Get current arbitrage opportunities"""
    try:
        if not hasattr(app.state, 'multi_exchange_service'):
            raise HTTPException(status_code=503, detail="Multi-exchange service not available")
        
        opportunities = await app.state.multi_exchange_service.get_arbitrage_opportunities()
        
        return {
            "opportunities": [
                {
                    "symbol": opp.symbol,
                    "buy_exchange": opp.buy_exchange,
                    "sell_exchange": opp.sell_exchange,
                    "buy_price": opp.buy_price,
                    "sell_price": opp.sell_price,
                    "profit_pct": opp.profit_pct,
                    "max_quantity": opp.max_quantity,
                    "estimated_profit": opp.estimated_profit,
                    "timestamp": opp.timestamp.isoformat()
                }
                for opp in opportunities
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting arbitrage opportunities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/exchanges/balances")
async def get_exchange_balances():
    """Get balances across all exchanges"""
    try:
        if not hasattr(app.state, 'multi_exchange_service'):
            raise HTTPException(status_code=503, detail="Multi-exchange service not available")
        
        balances = await app.state.multi_exchange_service.get_exchange_balances()
        
        # Convert balance objects to dictionaries
        formatted_balances = {}
        for exchange, exchange_balances in balances.items():
            formatted_balances[exchange] = {}
            for symbol, balance in exchange_balances.items():
                formatted_balances[exchange][symbol] = {
                    "free": balance.free,
                    "locked": balance.locked,
                    "total": balance.total,
                    "timestamp": balance.timestamp.isoformat()
                }
        
        return {"balances": formatted_balances}
        
    except Exception as e:
        logger.error(f"❌ Error getting exchange balances: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/exchanges/best-price/{symbol}")
async def get_best_price(symbol: str, side: str):
    """Get best price for a symbol across all exchanges"""
    try:
        if not hasattr(app.state, 'multi_exchange_service'):
            raise HTTPException(status_code=503, detail="Multi-exchange service not available")
        
        if side not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
        
        best_price = await app.state.multi_exchange_service.get_best_price(symbol.upper(), side)
        
        if not best_price:
            raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")
        
        return best_price
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting best price: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/exchanges/stats")
async def get_exchange_stats():
    """Get multi-exchange service statistics"""
    try:
        if not hasattr(app.state, 'multi_exchange_service'):
            raise HTTPException(status_code=503, detail="Multi-exchange service not available")
        
        stats = await app.state.multi_exchange_service.get_service_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting exchange stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/data-processor/status")
async def get_data_processor_status():
    """Get liquidation data processor status"""
    try:
        status = await get_processor_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting data processor status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data-processor/process")
async def process_liquidation_data(file_path: str = None):
    """Manually process a liquidation data file"""
    try:
        result = await process_liquidation_file(file_path)
        
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result['error'])
    except Exception as e:
        logger.error(f"Error processing liquidation data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/liquidation-backtest/status")
async def get_liquidation_backtest_status_endpoint():
    """Get liquidation backtesting service status"""
    try:
        status = await get_liquidation_backtest_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting liquidation backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/liquidation-backtest/run")
async def run_liquidation_backtest_endpoint(
    symbol: str,
    parameters: Dict[str, Any] = None,
    run_optimization: bool = False,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """Run liquidation backtesting for a specific symbol"""
    try:
        if run_optimization:
            result = await app.state.liquidation_backtesting_service.run_optimization(
                symbol=symbol,
                initial_cash=initial_cash,
                commission=commission
            )
        else:
            result = await app.state.liquidation_backtesting_service.run_single_backtest(
                symbol=symbol,
                parameters=parameters,
                initial_cash=initial_cash,
                commission=commission
            )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor,
            "optimization_results": result.optimization_results
        }
        
        return JSONResponse(content=result_dict)
        
    except Exception as e:
        logger.error(f"Error running liquidation backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/liquidation-backtest/multi-symbol")
async def run_multi_symbol_liquidation_backtest(
    symbols: List[str] = None,
    parameters: Dict[str, Any] = None,
    run_optimization: bool = False
):
    """Run liquidation backtesting for multiple symbols"""
    try:
        if symbols is None:
            symbols = settings.LIQUIDATION_DATA_SYMBOLS
        
        results = await app.state.liquidation_backtesting_service.run_multi_symbol_backtest(
            symbols=symbols,
            parameters=parameters,
            run_optimization=run_optimization
        )
        
        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
                "final_equity": result.final_equity
            })
        
        return JSONResponse(content={"results": results_dict})
        
    except Exception as e:
        logger.error(f"Error running multi-symbol liquidation backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/liquidation-backtest/results")
async def get_liquidation_backtest_results(limit: int = 10):
    """Get recent liquidation backtest results"""
    try:
        results = app.state.liquidation_backtesting_service.get_recent_results(limit)
        return JSONResponse(content={"results": results})
    except Exception as e:
        logger.error(f"Error getting liquidation backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/longonlliq-backtest/run")
async def run_longonlliq_backtest_endpoint(
    symbol: str,
    parameters: Dict[str, Any] = None,
    run_optimization: bool = False,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run Long on L LIQ backtesting for a specific symbol
    
    Args:
        symbol: Trading symbol
        parameters: Strategy parameters override
        run_optimization: Whether to run optimization
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Default parameters for LongOnLLiq strategy if not provided
        if parameters is None and not run_optimization:
            parameters = {
                'l_liq_entry_thresh': 100000,
                'entry_time_window_mins': 5,
                's_liq_closure_thresh': 50000,
                'exit_time_window_mins': 5,
                'take_profit': 0.02,
                'stop_loss': 0.01
            }
        
        if run_optimization:
            result = await app.state.liquidation_backtesting_service.run_optimization(
                symbol=symbol,
                initial_cash=initial_cash,
                commission=commission,
                strategy_type="longonlliq"
            )
        else:
            result = await app.state.liquidation_backtesting_service.run_single_backtest(
                symbol=symbol,
                parameters=parameters,
                initial_cash=initial_cash,
                commission=commission,
                strategy_type="longonlliq"
            )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor,
            "optimization_results": result.optimization_results
        }
        
        return JSONResponse(content=result_dict)
        
    except Exception as e:
        logger.error(f"Error running Long on L LIQ backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/longonlliq-backtest/multi-symbol")
async def run_multi_symbol_longonlliq_backtest(
    symbols: List[str] = None,
    parameters: Dict[str, Any] = None,
    run_optimization: bool = False
):
    """
    Run Long on L LIQ backtesting for multiple symbols
    
    Args:
        symbols: List of symbols to test
        parameters: Strategy parameters
        run_optimization: Whether to run optimization
    """
    try:
        if symbols is None:
            symbols = settings.LIQUIDATION_DATA_SYMBOLS
        
        # Default parameters for LongOnLLiq strategy if not provided
        if parameters is None and not run_optimization:
            parameters = {
                'l_liq_entry_thresh': 100000,
                'entry_time_window_mins': 5,
                's_liq_closure_thresh': 50000,
                'exit_time_window_mins': 5,
                'take_profit': 0.02,
                'stop_loss': 0.01
            }
        
        results = await app.state.liquidation_backtesting_service.run_multi_symbol_backtest(
            symbols=symbols,
            parameters=parameters,
            run_optimization=run_optimization,
            strategy_type="longonlliq"
        )
        
        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
                "final_equity": result.final_equity
            })
        
        return JSONResponse(content={"results": results_dict})
        
    except Exception as e:
        logger.error(f"Error running multi-symbol Long on L LIQ backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/longonlliq-backtest/optimize/{symbol}")
async def optimize_longonlliq_strategy(
    symbol: str,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run optimization for Long on L LIQ strategy with comprehensive parameter ranges
    
    Args:
        symbol: Trading symbol
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Comprehensive optimization ranges matching the original script
        optimization_ranges = {
            'l_liq_entry_thresh': range(10000, 500000, 10000),      # L LIQ threshold for entry
            'entry_time_window_mins': range(1, 11, 1),              # Entry lookback window
            's_liq_closure_thresh': range(10000, 500000, 10000),    # S LIQ threshold for exit
            'exit_time_window_mins': range(1, 11, 1),               # Exit lookback window
            'take_profit': [i / 1000 for i in range(5, 31, 5)],     # TP: 0.5% to 3.0% in 0.5% steps
            'stop_loss': [i / 1000 for i in range(5, 31, 5)]        # SL: 0.5% to 3.0% in 0.5% steps
        }
        
        result = await app.state.liquidation_backtesting_service.run_optimization(
            symbol=symbol,
            optimization_ranges=optimization_ranges,
            initial_cash=initial_cash,
            commission=commission,
            maximize='Equity Final [$]',
            strategy_type="longonlliq"
        )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor,
            "optimization_results": result.optimization_results
        }
        
        return JSONResponse(content=result_dict)
        
    except Exception as e:
        logger.error(f"Error optimizing Long on L LIQ strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === LONG ON S LIQ STRATEGY ENDPOINTS (ETH Strategy) ===

@app.post("/api/v1/longonsliq-backtest/run")
async def run_longonsliq_backtest_endpoint(
    symbol: str,
    parameters: Dict[str, Any] = None,
    run_optimization: bool = False,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run Long on S LIQ backtesting for a specific symbol (from liq_bt_eth.py)
    
    Args:
        symbol: Trading symbol
        parameters: Strategy parameters override
        run_optimization: Whether to run optimization
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Default parameters for LongOnSLiq strategy if not provided
        if parameters is None and not run_optimization:
            parameters = {
                's_liq_entry_thresh': 100000,
                'entry_time_window_mins': 20,
                'take_profit': 0.02,
                'stop_loss': 0.01
            }
        
        if run_optimization:
            result = await app.state.liquidation_backtesting_service.run_optimization(
                symbol=symbol,
                initial_cash=initial_cash,
                commission=commission,
                strategy_type="longonsliq"
            )
        else:
            result = await app.state.liquidation_backtesting_service.run_single_backtest(
                symbol=symbol,
                parameters=parameters,
                initial_cash=initial_cash,
                commission=commission,
                strategy_type="longonsliq"
            )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor,
            "optimization_results": result.optimization_results
        }
        
        return JSONResponse(content=result_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running Long on S LIQ backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/longonsliq-backtest/multi-symbol")
async def run_multi_symbol_longonsliq_backtest(
    symbols: List[str] = None,
    parameters: Dict[str, Any] = None,
    run_optimization: bool = False
):
    """
    Run Long on S LIQ backtesting for multiple symbols
    
    Args:
        symbols: List of symbols to test
        parameters: Strategy parameters
        run_optimization: Whether to run optimization
    """
    try:
        if symbols is None:
            symbols = settings.LIQUIDATION_DATA_SYMBOLS
        
        # Default parameters for LongOnSLiq strategy if not provided
        if parameters is None and not run_optimization:
            parameters = {
                's_liq_entry_thresh': 100000,
                'entry_time_window_mins': 20,
                'take_profit': 0.02,
                'stop_loss': 0.01
            }
        
        results = await app.state.liquidation_backtesting_service.run_multi_symbol_backtest(
            symbols=symbols,
            parameters=parameters,
            run_optimization=run_optimization,
            strategy_type="longonsliq"
        )
        
        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
                "final_equity": result.final_equity
            })
        
        return JSONResponse(content={"results": results_dict})
        
    except Exception as e:
        logger.error(f"Error running multi-symbol Long on S LIQ backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/longonsliq-backtest/optimize/{symbol}")
async def optimize_longonsliq_strategy(
    symbol: str,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run optimization for Long on S LIQ strategy with parameter ranges matching the original ETH script
    
    Args:
        symbol: Trading symbol
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Optimization ranges matching the original liq_bt_eth.py script
        optimization_ranges = {
            's_liq_entry_thresh': range(10000, 500000, 10000),       # S LIQ threshold for entry
            'entry_time_window_mins': range(5, 60, 5),               # Entry lookback window
            'take_profit': [i / 1000 for i in range(5, 31, 5)],      # TP: 0.5% to 3.0% in 0.5% steps
            'stop_loss': [i / 1000 for i in range(5, 31, 5)]         # SL: 0.5% to 3.0% in 0.5% steps
        }
        
        result = await app.state.liquidation_backtesting_service.run_optimization(
            symbol=symbol,
            optimization_ranges=optimization_ranges,
            initial_cash=initial_cash,
            commission=commission,
            maximize='Equity Final [$]',
            strategy_type="longonsliq"
        )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor,
            "optimization_results": result.optimization_results
        }
        
        return JSONResponse(content=result_dict)
        
    except Exception as e:
        logger.error(f"Error optimizing Long on S LIQ strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === WIF STRATEGY ENDPOINTS (Manual Grid Search) ===

@app.post("/api/v1/wif-backtest/manual-optimization")
async def run_wif_manual_optimization_endpoint(
    symbol: str = "WIF",
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run WIF manual grid search optimization (from liq_bt_wif.py)
    
    This endpoint implements the exact manual optimization logic from the original script
    with comprehensive parameter testing and progress tracking.
    
    Args:
        symbol: Trading symbol (defaults to WIF)
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        result = await app.state.liquidation_backtesting_service.run_wif_manual_optimization(
            symbol=symbol,
            initial_cash=initial_cash,
            commission=commission
        )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor,
            "optimization_results": result.optimization_results
        }
        
        return JSONResponse(content=result_dict)
        
    except Exception as e:
        logger.error(f"Error running WIF manual optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/wif-backtest/run")
async def run_wif_backtest_endpoint(
    symbol: str = "WIF",
    parameters: Dict[str, Any] = None,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run WIF strategy backtest with custom parameters
    
    Args:
        symbol: Trading symbol (defaults to WIF)
        parameters: Strategy parameters override
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Default parameters for WIF strategy if not provided
        if parameters is None:
            parameters = {
                's_liq_entry_thresh': 100000,
                'entry_time_window_mins': 20,
                'take_profit': 0.02,
                'stop_loss': 0.01
            }
        
        result = await app.state.liquidation_backtesting_service.run_single_backtest(
            symbol=symbol,
            parameters=parameters,
            initial_cash=initial_cash,
            commission=commission,
            strategy_type="wif_longonsliq"
        )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor
        }
        
        return JSONResponse(content=result_dict)
        
    except Exception as e:
        logger.error(f"Error running WIF backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/wif-backtest/status")
async def get_wif_backtest_status():
    """Get WIF backtesting service status"""
    try:
        status = await app.state.liquidation_backtesting_service.get_backtest_status()
        status["wif_strategy_enabled"] = True
        status["wif_manual_optimization_available"] = True
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting WIF backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === SHORT LIQUIDATION STRATEGY ENDPOINTS (Alpha Decay Testing) ===

@app.post("/api/v1/short-liquidation-backtest/run")
async def run_short_liquidation_backtest_endpoint(
    symbol: str = "SOL",
    parameters: Dict[str, Any] = None,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run Short Liquidation strategy backtest (from liq_bt_short_alphadecay.py)
    
    This strategy goes SHORT on S LIQ volume spikes and exits on L LIQ volume spikes.
    
    Args:
        symbol: Trading symbol (defaults to SOL)
        parameters: Strategy parameters override
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Default parameters for Short Liquidation strategy if not provided
        if parameters is None:
            parameters = {
                'short_liquidation_thresh': 100000,
                'entry_time_window_mins': 5,
                'long_liquidation_closure_thresh': 50000,
                'exit_time_window_mins': 5,
                'take_profit_pct': 0.02,
                'stop_loss_pct': 0.01
            }
        
        result = await app.state.liquidation_backtesting_service.run_single_backtest(
            symbol=symbol,
            parameters=parameters,
            initial_cash=initial_cash,
            commission=commission,
            strategy_type="short_liquidation"
        )
        
        # Convert result to dict for JSON response
        result_dict = {
            "symbol": result.symbol,
            "strategy_name": result.strategy_name,
            "parameters": result.parameters,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "avg_trade": result.avg_trade,
            "profit_factor": result.profit_factor
        }
        
        return JSONResponse(content=result_dict)
        
    except Exception as e:
        logger.error(f"Error running Short Liquidation backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/alpha-decay-test/run")
async def run_alpha_decay_test_endpoint(
    symbol: str = "SOL",
    delays: List[int] = None,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run Alpha Decay Test for DelayedLiquidationStrategy (from liq_bt_short_alphadecay.py)
    
    Tests how strategy performance degrades with increasing entry delays.
    This is a unique test that measures the time-sensitivity of liquidation signals.
    
    Args:
        symbol: Trading symbol (defaults to SOL as in original script)
        delays: List of delay values in minutes (defaults to [0,1,2,5,10,15,30,60])
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Use default delays from original script if not provided
        if delays is None:
            delays = [0, 1, 2, 5, 10, 15, 30, 60]
        
        result = await app.state.liquidation_backtesting_service.run_alpha_decay_test(
            symbol=symbol,
            delays=delays,
            initial_cash=initial_cash,
            commission=commission
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error running Alpha Decay Test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/alpha-decay-test/custom-delays")
async def run_alpha_decay_test_custom_delays_endpoint(
    symbol: str = "SOL",
    min_delay: int = 0,
    max_delay: int = 60,
    step_size: int = 5,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """
    Run Alpha Decay Test with custom delay range
    
    Args:
        symbol: Trading symbol
        min_delay: Minimum delay in minutes
        max_delay: Maximum delay in minutes
        step_size: Step size for delays
        initial_cash: Initial capital
        commission: Trading commission
    """
    try:
        # Generate custom delay range
        delays = list(range(min_delay, max_delay + 1, step_size))
        
        result = await app.state.liquidation_backtesting_service.run_alpha_decay_test(
            symbol=symbol,
            delays=delays,
            initial_cash=initial_cash,
            commission=commission
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error running custom Alpha Decay Test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alpha-decay-test/status")
async def get_alpha_decay_test_status():
    """Get Alpha Decay Test service status"""
    try:
        status = await app.state.liquidation_backtesting_service.get_backtest_status()
        status["alpha_decay_test_enabled"] = True
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting Alpha Decay Test status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/short-monte-backtest/run")
async def run_short_monte_backtest_endpoint(
    symbol: str = "SOL",
    parameters: dict = None,
    initial_cash: float = 100000,
    commission: float = 0.002
):
    """Run Short Monte Carlo backtest with custom parameters"""
    try:
        if not app.state.liquidation_backtesting_service:
            raise HTTPException(status_code=503, detail="Liquidation backtesting service not available")
        
        # Run single backtest using short_monte strategy type
        result = await app.state.liquidation_backtesting_service.run_single_backtest(
            symbol=symbol,
            parameters=parameters,
            initial_cash=initial_cash,
            commission=commission,
            strategy_type="short_monte"
        )
        
        return JSONResponse(content={
            "success": True,
            "result": {
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "final_equity": result.final_equity,
                "start_date": result.start_date,
                "end_date": result.end_date
            }
        })
        
    except Exception as e:
        logger.error(f"Error running Short Monte backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/short-monte-backtest/status")
async def get_short_monte_backtest_status():
    """Get Short Monte backtest service status"""
    try:
        status = await app.state.liquidation_backtesting_service.get_backtest_status()
        status["short_monte_strategy_enabled"] = True
        status["plotting_enabled"] = True
        status["default_symbol"] = "SOL"
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting Short Monte backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/kalman-reversal-backtest/run")
async def run_kalman_reversal_backtest_endpoint(
    symbol: str = "BTC",
    parameters: dict = None,
    initial_cash: float = 100000,
    commission: float = 0.002,
    test_data_folder: str = "test_data"
):
    """Run Kalman Filter Reversal backtest with custom parameters"""
    try:
        if not app.state.liquidation_backtesting_service:
            raise HTTPException(status_code=503, detail="Liquidation backtesting service not available")
        
        if not KALMAN_AVAILABLE:
            raise HTTPException(status_code=400, detail="pykalman is required for Kalman Filter strategy. Please install: pip install pykalman")
        
        # Set default parameters if none provided
        if parameters is None:
            parameters = {
                'window': 50,
                'take_profit': 0.05,
                'stop_loss': 0.03,
                'observation_covariance': 1.0,
                'transition_covariance': 0.01
            }
        
        # Run single backtest using kalman_reversal strategy type
        result = await app.state.liquidation_backtesting_service.run_single_backtest(
            symbol=symbol,
            parameters=parameters,
            initial_cash=initial_cash,
            commission=commission,
            strategy_type="kalman_reversal"
        )
        
        return JSONResponse(content={
            "success": True,
            "result": {
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "final_equity": result.final_equity,
                "start_date": result.start_date,
                "end_date": result.end_date,
                "best_trade": result.best_trade,
                "worst_trade": result.worst_trade,
                "avg_trade": result.avg_trade,
                "profit_factor": result.profit_factor
            }
        })
        
    except Exception as e:
        logger.error(f"Error running Kalman Reversal backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/kalman-reversal-backtest/optimize")
async def run_kalman_reversal_optimization_endpoint(
    symbol: str = "BTC",
    optimization_ranges: dict = None,
    initial_cash: float = 100000,
    commission: float = 0.002,
    maximize: str = "Equity Final [$]"
):
    """Run Kalman Filter Reversal optimization"""
    try:
        if not app.state.liquidation_backtesting_service:
            raise HTTPException(status_code=503, detail="Liquidation backtesting service not available")
        
        if not KALMAN_AVAILABLE:
            raise HTTPException(status_code=400, detail="pykalman is required for Kalman Filter strategy. Please install: pip install pykalman")
        
        # Run optimization using kalman_reversal strategy type
        result = await app.state.liquidation_backtesting_service.run_optimization(
            symbol=symbol,
            optimization_ranges=optimization_ranges,
            initial_cash=initial_cash,
            commission=commission,
            maximize=maximize,
            strategy_type="kalman_reversal"
        )
        
        return JSONResponse(content={
            "success": True,
            "result": {
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "final_equity": result.final_equity,
                "start_date": result.start_date,
                "end_date": result.end_date,
                "optimization_results": result.optimization_results
            }
        })
        
    except Exception as e:
        logger.error(f"Error running Kalman Reversal optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/kalman-reversal-backtest/status")
async def get_kalman_reversal_backtest_status():
    """Get Kalman Filter Reversal backtest service status"""
    try:
        status = await app.state.liquidation_backtesting_service.get_backtest_status()
        status["kalman_reversal_enabled"] = KALMAN_AVAILABLE
        status["pykalman_available"] = KALMAN_AVAILABLE
        if KALMAN_AVAILABLE:
            status["supported_features"] = [
                "Single backtest runs",
                "Parameter optimization", 
                "Multi-symbol testing",
                "OHLCV data loading",
                "Scikit-optimize integration"
            ]
        else:
            status["missing_dependencies"] = ["pykalman", "scikit-optimize"]
            status["install_command"] = "pip install pykalman scikit-optimize"
        
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting Kalman Reversal backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === TRADING HOURS DATA FILTER ENDPOINTS ===

@app.get("/api/v1/trading-hours-filter/status")
async def get_trading_hours_filter_status_endpoint():
    """Get trading hours data filter service status"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            return JSONResponse(content={
                "service_enabled": False,
                "message": "Trading hours filter service not enabled"
            })
        
        status = await app.state.trading_hours_filter_service.get_service_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting trading hours filter status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/trading-hours-filter/filter-file")
async def filter_single_file_endpoint(
    filename: str,
    data_folder: str = None,
    market_open_utc: str = "13:30",
    one_hour_after_open_utc: str = "14:30",
    one_hour_before_close_utc: str = "19:00",
    market_close_utc: str = "20:00",
    exclude_weekends: bool = True,
    output_suffix: str = "-1stlasthr"
):
    """Filter a single file to trading hours (first/last hour)"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        from services.trading_hours_filter_service import TradingHoursConfig
        
        # Create custom config
        config = TradingHoursConfig(
            market_open_utc=market_open_utc,
            one_hour_after_open_utc=one_hour_after_open_utc,
            one_hour_before_close_utc=one_hour_before_close_utc,
            market_close_utc=market_close_utc,
            exclude_weekends=exclude_weekends,
            output_suffix=output_suffix
        )
        
        # Filter the file
        result = await app.state.trading_hours_filter_service.filter_file(
            filename, config, data_folder
        )
        
        return JSONResponse(content={
            "success": result.success,
            "input_file": result.input_file,
            "output_file": result.output_file,
            "original_records": result.original_records,
            "filtered_records": result.filtered_records,
            "filter_config": result.filter_config,
            "processing_time_seconds": result.processing_time_seconds,
            "error_message": result.error_message
        })
        
    except Exception as e:
        logger.error(f"Error filtering file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/trading-hours-filter/filter-multiple")
async def filter_multiple_files_endpoint(
    filenames: List[str],
    data_folder: str = None,
    market_open_utc: str = "13:30",
    one_hour_after_open_utc: str = "14:30",
    one_hour_before_close_utc: str = "19:00",
    market_close_utc: str = "20:00",
    exclude_weekends: bool = True,
    output_suffix: str = "-1stlasthr"
):
    """Filter multiple files to trading hours"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        from services.trading_hours_filter_service import TradingHoursConfig
        
        # Create custom config
        config = TradingHoursConfig(
            market_open_utc=market_open_utc,
            one_hour_after_open_utc=one_hour_after_open_utc,
            one_hour_before_close_utc=one_hour_before_close_utc,
            market_close_utc=market_close_utc,
            exclude_weekends=exclude_weekends,
            output_suffix=output_suffix
        )
        
        # Filter the files
        results = await app.state.trading_hours_filter_service.filter_multiple_files(
            filenames, config, data_folder
        )
        
        return JSONResponse(content={
            "total_files": len(results),
            "successful_files": sum(1 for r in results if r.success),
            "failed_files": sum(1 for r in results if not r.success),
            "results": [
                {
                    "success": r.success,
                    "input_file": r.input_file,
                    "output_file": r.output_file,
                    "original_records": r.original_records,
                    "filtered_records": r.filtered_records,
                    "processing_time_seconds": r.processing_time_seconds,
                    "error_message": r.error_message
                }
                for r in results
            ]
        })
        
    except Exception as e:
        logger.error(f"Error filtering multiple files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/trading-hours-filter/discover-and-filter")
async def discover_and_filter_all_endpoint(
    data_folder: str = None,
    file_pattern: str = "*.csv",
    market_open_utc: str = "13:30",
    one_hour_after_open_utc: str = "14:30",
    one_hour_before_close_utc: str = "19:00",
    market_close_utc: str = "20:00",
    exclude_weekends: bool = True,
    output_suffix: str = "-1stlasthr"
):
    """Discover all CSV files in data folder and filter them to trading hours"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        from services.trading_hours_filter_service import TradingHoursConfig
        
        # Create custom config
        config = TradingHoursConfig(
            market_open_utc=market_open_utc,
            one_hour_after_open_utc=one_hour_after_open_utc,
            one_hour_before_close_utc=one_hour_before_close_utc,
            market_close_utc=market_close_utc,
            exclude_weekends=exclude_weekends,
            output_suffix=output_suffix
        )
        
        # Discover and filter all files
        results = await app.state.trading_hours_filter_service.discover_and_filter_all(
            config, data_folder, file_pattern
        )
        
        return JSONResponse(content={
            "total_files_discovered": len(results),
            "successful_files": sum(1 for r in results if r.success),
            "failed_files": sum(1 for r in results if not r.success),
            "file_pattern": file_pattern,
            "data_folder": data_folder or "default",
            "results": [
                {
                    "success": r.success,
                    "input_file": r.input_file,
                    "output_file": r.output_file,
                    "original_records": r.original_records,
                    "filtered_records": r.filtered_records,
                    "processing_time_seconds": r.processing_time_seconds,
                    "error_message": r.error_message
                }
                for r in results
            ]
        })
        
    except Exception as e:
        logger.error(f"Error discovering and filtering files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trading-hours-filter/results")
async def get_trading_hours_filter_results(limit: int = 10):
    """Get recent trading hours filter results"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        results = app.state.trading_hours_filter_service.get_recent_results(limit)
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        logger.error(f"Error getting trading hours filter results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === MARKET/NON-MARKET HOURS DATA FILTER ENDPOINTS ===

@app.post("/api/v1/market-non-market-filter/filter-file")
async def filter_market_non_market_single_file_endpoint(
    filename: str,
    data_folder: str = None,
    market_open_utc: str = "13:30",
    market_close_utc: str = "20:00",
    exclude_weekends: bool = True,
    market_hours_suffix: str = "-mkt-open",
    non_market_hours_suffix: str = "-mkt-closed"
):
    """Filter a single file into market hours and non-market hours"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        from services.trading_hours_filter_service import MarketHoursConfig
        
        # Create custom config
        config = MarketHoursConfig(
            market_open_utc=market_open_utc,
            market_close_utc=market_close_utc,
            exclude_weekends=exclude_weekends,
            market_hours_suffix=market_hours_suffix,
            non_market_hours_suffix=non_market_hours_suffix
        )
        
        # Filter the file
        result = await app.state.trading_hours_filter_service.filter_market_non_market_hours(
            filename, config, data_folder
        )
        
        return JSONResponse(content={
            "success": result.success,
            "input_file": result.input_file,
            "market_hours_file": result.output_file,
            "non_market_hours_file": result.output_file_secondary,
            "original_records": result.original_records,
            "market_hours_records": result.filtered_records,
            "non_market_hours_records": result.filtered_records_secondary,
            "filter_config": result.filter_config,
            "processing_time_seconds": result.processing_time_seconds,
            "error_message": result.error_message
        })
        
    except Exception as e:
        logger.error(f"Error filtering file into market/non-market hours: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/market-non-market-filter/filter-multiple")
async def filter_market_non_market_multiple_files_endpoint(
    filenames: List[str],
    data_folder: str = None,
    market_open_utc: str = "13:30",
    market_close_utc: str = "20:00",
    exclude_weekends: bool = True,
    market_hours_suffix: str = "-mkt-open",
    non_market_hours_suffix: str = "-mkt-closed"
):
    """Filter multiple files into market hours and non-market hours"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        from services.trading_hours_filter_service import MarketHoursConfig
        
        # Create custom config
        config = MarketHoursConfig(
            market_open_utc=market_open_utc,
            market_close_utc=market_close_utc,
            exclude_weekends=exclude_weekends,
            market_hours_suffix=market_hours_suffix,
            non_market_hours_suffix=non_market_hours_suffix
        )
        
        # Filter the files
        results = await app.state.trading_hours_filter_service.filter_multiple_market_non_market(
            filenames, config, data_folder
        )
        
        return JSONResponse(content={
            "total_files": len(results),
            "successful_files": sum(1 for r in results if r.success),
            "failed_files": sum(1 for r in results if not r.success),
            "results": [
                {
                    "success": r.success,
                    "input_file": r.input_file,
                    "market_hours_file": r.output_file,
                    "non_market_hours_file": r.output_file_secondary,
                    "original_records": r.original_records,
                    "market_hours_records": r.filtered_records,
                    "non_market_hours_records": r.filtered_records_secondary,
                    "processing_time_seconds": r.processing_time_seconds,
                    "error_message": r.error_message
                }
                for r in results
            ]
        })
        
    except Exception as e:
        logger.error(f"Error filtering multiple files into market/non-market hours: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/market-non-market-filter/discover-and-filter")
async def discover_and_filter_market_non_market_all_endpoint(
    data_folder: str = None,
    file_pattern: str = "*.csv",
    market_open_utc: str = "13:30",
    market_close_utc: str = "20:00",
    exclude_weekends: bool = True,
    market_hours_suffix: str = "-mkt-open",
    non_market_hours_suffix: str = "-mkt-closed"
):
    """Discover all CSV files in data folder and filter them into market/non-market hours"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        from services.trading_hours_filter_service import MarketHoursConfig
        
        # Create custom config
        config = MarketHoursConfig(
            market_open_utc=market_open_utc,
            market_close_utc=market_close_utc,
            exclude_weekends=exclude_weekends,
            market_hours_suffix=market_hours_suffix,
            non_market_hours_suffix=non_market_hours_suffix
        )
        
        # Discover and filter all files
        results = await app.state.trading_hours_filter_service.discover_and_filter_market_non_market_all(
            config, data_folder, file_pattern
        )
        
        return JSONResponse(content={
            "total_files_discovered": len(results),
            "successful_files": sum(1 for r in results if r.success),
            "failed_files": sum(1 for r in results if not r.success),
            "file_pattern": file_pattern,
            "data_folder": data_folder or "default",
            "results": [
                {
                    "success": r.success,
                    "input_file": r.input_file,
                    "market_hours_file": r.output_file,
                    "non_market_hours_file": r.output_file_secondary,
                    "original_records": r.original_records,
                    "market_hours_records": r.filtered_records,
                    "non_market_hours_records": r.filtered_records_secondary,
                    "processing_time_seconds": r.processing_time_seconds,
                    "error_message": r.error_message
                }
                for r in results
            ]
        })
        
    except Exception as e:
        logger.error(f"Error discovering and filtering files into market/non-market hours: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-non-market-filter/status")
async def get_market_non_market_filter_status_endpoint():
    """Get market/non-market hours filter service status"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            return JSONResponse(content={
                "service_enabled": False,
                "message": "Market/non-market hours filter service not enabled"
            })
        
        status = await app.state.trading_hours_filter_service.get_service_status()
        
        # Add market/non-market specific status info
        status["market_non_market_features"] = {
            "market_hours_filtering": True,
            "non_market_hours_filtering": True,
            "concurrent_processing": True,
            "auto_discovery": True,
            "dual_output_files": True
        }
        
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting market/non-market filter status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-non-market-filter/results")
async def get_market_non_market_filter_results(limit: int = 10):
    """Get recent market/non-market hours filter results"""
    try:
        if not hasattr(app.state, 'trading_hours_filter_service'):
            raise HTTPException(status_code=503, detail="Trading hours filter service not available")
        
        results = app.state.trading_hours_filter_service.get_recent_results(limit)
        
        # Filter for market/non-market results (those with secondary output files)
        market_non_market_results = [
            r for r in results 
            if r.get('output_file_secondary') is not None
        ]
        
        return JSONResponse(content={"results": market_non_market_results})
        
    except Exception as e:
        logger.error(f"Error getting market/non-market filter results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def start_services(app_state):
    """Start all services"""
    try:
        logger.info("🔄 Starting services...")
        
        # Start core services
        await app_state.market_data_manager.start()
        await app_state.portfolio_manager.initialize()
        await app_state.risk_manager.initialize()
        
        # Start new services
        if hasattr(app_state, 'social_sentiment_service'):
            asyncio.create_task(app_state.social_sentiment_service.start())
            logger.info("✅ Social Sentiment Service started")
        
        if hasattr(app_state, 'multi_exchange_service'):
            asyncio.create_task(app_state.multi_exchange_service.start())
            logger.info("✅ Multi-Exchange Service started")
        
        if hasattr(app_state, 'advanced_dashboard_service'):
            asyncio.create_task(app_state.advanced_dashboard_service.start())
            logger.info("✅ Advanced Dashboard Service started")
        
        # Start strategy engine
        await app_state.strategy_engine.start()
        
        logger.info("✅ All services started successfully")
        
    except Exception as e:
        logger.error(f"❌ Error starting services: {e}")
        raise

async def shutdown_services(app_state):
    """Shutdown all services"""
    try:
        # Shutdown services in reverse order
        if hasattr(app_state, 'strategy_engine'):
            await app_state.strategy_engine.stop()
        
        if hasattr(app_state, 'advanced_dashboard_service'):
            await app_state.advanced_dashboard_service.cleanup()
        
        if hasattr(app_state, 'multi_exchange_service'):
            await app_state.multi_exchange_service.cleanup()
        
        if hasattr(app_state, 'social_sentiment_service'):
            await app_state.social_sentiment_service.cleanup()
        
        if hasattr(app_state, 'order_execution_service'):
            await app_state.order_execution_service.cleanup()
        
        if hasattr(app_state, 'market_data_manager'):
            await app_state.market_data_manager.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# ================== SEASONAL FILTERING ENDPOINTS ==================

@app.post("/api/seasonal-filter/single-file")
async def seasonal_filter_single_file(
    filename: str = Query(..., description="Name of the CSV file to filter"),
    data_folder: str = Query(default=settings.TRADING_HOURS_DATA_FOLDER, description="Data folder path"),
    exclude_start_month: int = Query(default=settings.SEASONAL_EXCLUDE_START_MONTH, description="Start month to exclude (1-12)"),
    exclude_end_month: int = Query(default=settings.SEASONAL_EXCLUDE_END_MONTH, description="End month to exclude (1-12)"),
    output_suffix: str = Query(default=settings.SEASONAL_OUTPUT_SUFFIX, description="Output file suffix")
):
    """Filter a single CSV file to exclude seasonal months"""
    try:
        result = await filter_seasonal_single_file(
            filename=filename,
            data_folder=data_folder,
            exclude_start_month=exclude_start_month,
            exclude_end_month=exclude_end_month,
            output_suffix=output_suffix
        )
        
        return {
            "success": result.success,
            "message": f"Seasonal filtering completed for {filename}",
            "data": {
                "input_file": result.input_file,
                "output_file": result.output_file,
                "original_records": result.original_records,
                "filtered_records": result.filtered_records,
                "processing_time_seconds": result.processing_time_seconds,
                "filter_config": result.filter_config,
                "error_message": result.error_message
            }
        }
        
    except Exception as e:
        logger.error(f"Error in seasonal filter endpoint: {str(e)}")
        return {
            "success": False,
            "message": f"Error filtering {filename}",
            "error": str(e)
        }

@app.post("/api/seasonal-filter/multiple-files")
async def seasonal_filter_multiple_files(
    filenames: List[str] = Body(..., description="List of CSV filenames to filter"),
    data_folder: str = Body(default=settings.TRADING_HOURS_DATA_FOLDER, description="Data folder path"),
    exclude_start_month: int = Body(default=settings.SEASONAL_EXCLUDE_START_MONTH, description="Start month to exclude (1-12)"),
    exclude_end_month: int = Body(default=settings.SEASONAL_EXCLUDE_END_MONTH, description="End month to exclude (1-12)"),
    output_suffix: str = Body(default=settings.SEASONAL_OUTPUT_SUFFIX, description="Output file suffix")
):
    """Filter multiple CSV files to exclude seasonal months"""
    try:
        service = app.state.trading_hours_filter_service
        config = SeasonalFilterConfig(
            exclude_start_month=exclude_start_month,
            exclude_end_month=exclude_end_month,
            output_suffix=output_suffix
        )
        
        results = await service.filter_multiple_seasonal(filenames, config, data_folder)
        
        successful_count = sum(1 for r in results if r.success)
        
        return {
            "success": True,
            "message": f"Batch seasonal filtering completed: {successful_count}/{len(filenames)} files processed successfully",
            "data": {
                "total_files": len(filenames),
                "successful_files": successful_count,
                "failed_files": len(filenames) - successful_count,
                "results": [
                    {
                        "input_file": r.input_file,
                        "output_file": r.output_file,
                        "original_records": r.original_records,
                        "filtered_records": r.filtered_records,
                        "processing_time_seconds": r.processing_time_seconds,
                        "success": r.success,
                        "error_message": r.error_message
                    }
                    for r in results
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in batch seasonal filter endpoint: {str(e)}")
        return {
            "success": False,
            "message": f"Error filtering {len(filenames)} files",
            "error": str(e)
        }

@app.post("/api/seasonal-filter/discover-and-filter-all")
async def seasonal_filter_discover_and_filter_all(
    data_folder: str = Body(default=settings.TRADING_HOURS_DATA_FOLDER, description="Data folder path"),
    file_pattern: str = Body(default=settings.TRADING_HOURS_FILE_PATTERN, description="File pattern to match"),
    exclude_start_month: int = Body(default=settings.SEASONAL_EXCLUDE_START_MONTH, description="Start month to exclude (1-12)"),
    exclude_end_month: int = Body(default=settings.SEASONAL_EXCLUDE_END_MONTH, description="End month to exclude (1-12)"),
    output_suffix: str = Body(default=settings.SEASONAL_OUTPUT_SUFFIX, description="Output file suffix")
):
    """Discover all CSV files in folder and filter them to exclude seasonal months"""
    try:
        service = app.state.trading_hours_filter_service
        config = SeasonalFilterConfig(
            exclude_start_month=exclude_start_month,
            exclude_end_month=exclude_end_month,
            output_suffix=output_suffix
        )
        
        results = await service.discover_and_filter_seasonal_all(config, data_folder, file_pattern)
        
        if not results:
            return {
                "success": False,
                "message": f"No CSV files found matching pattern '{file_pattern}' in {data_folder}",
                "data": {"discovered_files": 0, "processed_files": 0}
            }
        
        successful_count = sum(1 for r in results if r.success)
        
        return {
            "success": True,
            "message": f"Auto-discovery and seasonal filtering completed: {successful_count}/{len(results)} files processed successfully",
            "data": {
                "discovered_files": len(results),
                "successful_files": successful_count,
                "failed_files": len(results) - successful_count,
                "results": [
                    {
                        "input_file": r.input_file,
                        "output_file": r.output_file,
                        "original_records": r.original_records,
                        "filtered_records": r.filtered_records,
                        "processing_time_seconds": r.processing_time_seconds,
                        "success": r.success,
                        "error_message": r.error_message
                    }
                    for r in results
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in auto-discovery seasonal filter endpoint: {str(e)}")
        return {
            "success": False,
            "message": f"Error in auto-discovery seasonal filtering",
            "error": str(e)
        }

@app.get("/api/seasonal-filter/config")
async def get_seasonal_filter_config():
    """Get current seasonal filter configuration"""
    try:
        service = app.state.trading_hours_filter_service
        config = service.default_seasonal_config
        
        return {
            "success": True,
            "message": "Seasonal filter configuration retrieved",
            "data": {
                "exclude_start_month": config.exclude_start_month,
                "exclude_end_month": config.exclude_end_month,
                "output_suffix": config.output_suffix,
                "excluded_months_names": [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ][config.exclude_start_month-1:config.exclude_end_month]
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving seasonal filter config: {str(e)}")
        return {
            "success": False,
            "message": "Error retrieving seasonal filter configuration",
            "error": str(e)
        }

@app.put("/api/seasonal-filter/config")
async def update_seasonal_filter_config(
    exclude_start_month: int = Body(..., description="Start month to exclude (1-12)", ge=1, le=12),
    exclude_end_month: int = Body(..., description="End month to exclude (1-12)", ge=1, le=12),
    output_suffix: str = Body(default="-nosummers", description="Output file suffix")
):
    """Update seasonal filter configuration"""
    try:
        if exclude_start_month > exclude_end_month:
            return {
                "success": False,
                "message": "Start month cannot be greater than end month",
                "error": "Invalid month range"
            }
        
        service = app.state.trading_hours_filter_service
        service.default_seasonal_config = SeasonalFilterConfig(
            exclude_start_month=exclude_start_month,
            exclude_end_month=exclude_end_month,
            output_suffix=output_suffix
        )
        
        return {
            "success": True,
            "message": "Seasonal filter configuration updated successfully",
            "data": {
                "exclude_start_month": exclude_start_month,
                "exclude_end_month": exclude_end_month,
                "output_suffix": output_suffix,
                "excluded_months_names": [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ][exclude_start_month-1:exclude_end_month]
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating seasonal filter config: {str(e)}")
        return {
            "success": False,
            "message": "Error updating seasonal filter configuration",
            "error": str(e)
        }

@app.get("/api/seasonal-filter/results")
async def get_seasonal_filter_results(limit: int = Query(default=10, description="Number of recent results to return")):
    """Get recent seasonal filtering results"""
    try:
        service = app.state.trading_hours_filter_service
        results = service.get_recent_results(limit)
        
        # Filter for seasonal filtering results only
        seasonal_results = [
            r for r in results 
            if r.get('filter_config', {}).get('output_suffix', '').endswith('nosummers') or
               r.get('filter_config', {}).get('exclude_start_month') is not None
        ]
        
        return {
            "success": True,
            "message": f"Retrieved {len(seasonal_results)} recent seasonal filtering results",
            "data": {
                "total_results": len(seasonal_results),
                "results": seasonal_results
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving seasonal filter results: {str(e)}")
        return {
            "success": False,
            "message": "Error retrieving seasonal filter results",
            "error": str(e)
        }

# ================== MANUAL TRADING ENDPOINTS ==================

@app.get("/api/manual-trading/status")
async def get_manual_trading_status():
    """Get manual trading service status"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            return {
                "success": False,
                "message": "Manual trading service not available",
                "service_enabled": False
            }
        
        status = await app.state.manual_trading_service.get_service_status()
        return {
            "success": True,
            "message": "Manual trading service status retrieved",
            "data": status
        }
        
    except Exception as e:
        logger.error(f"Error getting manual trading status: {str(e)}")
        return {
            "success": False,
            "message": "Error retrieving manual trading status",
            "error": str(e)
        }

@app.get("/api/manual-trading/position/{symbol}")
async def get_position_details(symbol: str):
    """Get current position details for a symbol"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            raise HTTPException(status_code=503, detail="Manual trading service not available")
        
        position = await app.state.manual_trading_service.get_position_details(symbol.upper())
        
        return {
            "success": True,
            "message": f"Position details retrieved for {symbol}",
            "data": {
                "symbol": position.symbol,
                "tokens": position.tokens,
                "price": position.price,
                "usd_value": position.usd_value,
                "timestamp": position.timestamp.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting position details for {symbol}: {str(e)}")
        return {
            "success": False,
            "message": f"Error retrieving position details for {symbol}",
            "error": str(e)
        }

@app.post("/api/manual-trading/open-position")
async def open_position_to_target(
    symbol: str = Body(..., description="Trading symbol"),
    target_usd_size: float = Body(..., description="Target position size in USD"),
    max_chunk_usd: float = Body(default=None, description="Maximum chunk size in USD"),
    orders_per_burst: int = Body(default=1, description="Number of orders per burst"),
    slippage_bps: int = Body(default=50, description="Slippage tolerance in basis points"),
    tx_delay_seconds: float = Body(default=2.0, description="Delay between transactions in seconds"),
    user_id: str = Body(default=None, description="User ID for tracking"),
    reason: str = Body(default=None, description="Reason for the trade")
):
    """Open position to target USD size with chunked buying"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            raise HTTPException(status_code=503, detail="Manual trading service not available")
        
        # Use defaults from settings if not provided
        if max_chunk_usd is None:
            max_chunk_usd = settings.MANUAL_TRADING_MAX_USD_ORDER_SIZE
        
        # Create trade request
        request = TradeRequest(
            action=TradeAction.OPEN_POSITION,
            symbol=symbol.upper(),
            target_usd_size=target_usd_size,
            max_chunk_usd=max_chunk_usd,
            orders_per_burst=orders_per_burst,
            slippage_bps=slippage_bps,
            tx_delay_seconds=tx_delay_seconds,
            user_id=user_id,
            reason=reason
        )
        
        # Execute trade
        result = await app.state.manual_trading_service.execute_trade(request)
        
        return {
            "success": result.success,
            "message": f"Position opening {'completed' if result.success else 'failed'} for {symbol}",
            "data": {
                "request_id": result.request_id,
                "action": result.action.value,
                "symbol": result.symbol,
                "initial_position": {
                    "tokens": result.initial_position.tokens,
                    "price": result.initial_position.price,
                    "usd_value": result.initial_position.usd_value
                },
                "final_position": {
                    "tokens": result.final_position.tokens,
                    "price": result.final_position.price,
                    "usd_value": result.final_position.usd_value
                },
                "orders_executed": result.orders_executed,
                "total_volume_usd": result.total_volume_usd,
                "execution_time_seconds": result.execution_time_seconds,
                "error_message": result.error_message
            }
        }
        
    except Exception as e:
        logger.error(f"Error opening position for {symbol}: {str(e)}")
        return {
            "success": False,
            "message": f"Error opening position for {symbol}",
            "error": str(e)
        }

@app.post("/api/manual-trading/close-position")
async def close_position_fully(
    symbol: str = Body(..., description="Trading symbol"),
    max_chunk_usd: float = Body(default=None, description="Maximum chunk size in USD"),
    slippage_bps: int = Body(default=50, description="Slippage tolerance in basis points"),
    tx_delay_seconds: float = Body(default=2.0, description="Delay between transactions in seconds"),
    user_id: str = Body(default=None, description="User ID for tracking"),
    reason: str = Body(default=None, description="Reason for the trade")
):
    """Close entire position with chunked selling"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            raise HTTPException(status_code=503, detail="Manual trading service not available")
        
        # Use defaults from settings if not provided
        if max_chunk_usd is None:
            max_chunk_usd = settings.MANUAL_TRADING_MAX_USD_ORDER_SIZE
        
        # Create trade request
        request = TradeRequest(
            action=TradeAction.CLOSE_POSITION,
            symbol=symbol.upper(),
            max_chunk_usd=max_chunk_usd,
            slippage_bps=slippage_bps,
            tx_delay_seconds=tx_delay_seconds,
            user_id=user_id,
            reason=reason
        )
        
        # Execute trade
        result = await app.state.manual_trading_service.execute_trade(request)
        
        return {
            "success": result.success,
            "message": f"Position closing {'completed' if result.success else 'failed'} for {symbol}",
            "data": {
                "request_id": result.request_id,
                "action": result.action.value,
                "symbol": result.symbol,
                "initial_position": {
                    "tokens": result.initial_position.tokens,
                    "price": result.initial_position.price,
                    "usd_value": result.initial_position.usd_value
                },
                "final_position": {
                    "tokens": result.final_position.tokens,
                    "price": result.final_position.price,
                    "usd_value": result.final_position.usd_value
                },
                "orders_executed": result.orders_executed,
                "total_volume_usd": result.total_volume_usd,
                "execution_time_seconds": result.execution_time_seconds,
                "error_message": result.error_message
            }
        }
        
    except Exception as e:
        logger.error(f"Error closing position for {symbol}: {str(e)}")
        return {
            "success": False,
            "message": f"Error closing position for {symbol}",
            "error": str(e)
        }

@app.post("/api/manual-trading/quick-buy")
async def quick_buy_position(
    symbol: str = Body(default=None, description="Trading symbol (uses default if not provided)"),
    target_usd_size: float = Body(default=None, description="Target size in USD (uses default if not provided)")
):
    """Quick buy using default configuration"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            raise HTTPException(status_code=503, detail="Manual trading service not available")
        
        # Use defaults from settings
        if symbol is None:
            symbol = settings.MANUAL_TRADING_PRIMARY_SYMBOL
        if target_usd_size is None:
            target_usd_size = settings.MANUAL_TRADING_DEFAULT_USD_SIZE
        
        # Execute using convenience function
        result = await execute_manual_buy(
            symbol=symbol.upper(),
            target_usd_size=target_usd_size,
            service=app.state.manual_trading_service
        )
        
        return {
            "success": result.success,
            "message": f"Quick buy {'completed' if result.success else 'failed'} for {symbol}",
            "data": {
                "request_id": result.request_id,
                "symbol": result.symbol,
                "target_usd_size": target_usd_size,
                "final_usd_value": result.final_position.usd_value,
                "orders_executed": result.orders_executed,
                "execution_time_seconds": result.execution_time_seconds,
                "error_message": result.error_message
            }
        }
        
    except Exception as e:
        logger.error(f"Error in quick buy: {str(e)}")
        return {
            "success": False,
            "message": "Error in quick buy operation",
            "error": str(e)
        }

@app.post("/api/manual-trading/quick-sell")
async def quick_sell_position(
    symbol: str = Body(default=None, description="Trading symbol (uses default if not provided)")
):
    """Quick sell using default configuration"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            raise HTTPException(status_code=503, detail="Manual trading service not available")
        
        # Use default from settings
        if symbol is None:
            symbol = settings.MANUAL_TRADING_PRIMARY_SYMBOL
        
        # Execute using convenience function
        result = await execute_manual_sell(
            symbol=symbol.upper(),
            service=app.state.manual_trading_service
        )
        
        return {
            "success": result.success,
            "message": f"Quick sell {'completed' if result.success else 'failed'} for {symbol}",
            "data": {
                "request_id": result.request_id,
                "symbol": result.symbol,
                "initial_usd_value": result.initial_position.usd_value,
                "final_usd_value": result.final_position.usd_value,
                "orders_executed": result.orders_executed,
                "execution_time_seconds": result.execution_time_seconds,
                "error_message": result.error_message
            }
        }
        
    except Exception as e:
        logger.error(f"Error in quick sell: {str(e)}")
        return {
            "success": False,
            "message": "Error in quick sell operation",
            "error": str(e)
        }

@app.get("/api/manual-trading/config")
async def get_manual_trading_config():
    """Get current manual trading configuration"""
    try:
        return {
            "success": True,
            "message": "Manual trading configuration retrieved",
            "data": {
                "service_enabled": settings.ENABLE_MANUAL_TRADING,
                "primary_symbol": settings.MANUAL_TRADING_PRIMARY_SYMBOL,
                "default_usd_size": settings.MANUAL_TRADING_DEFAULT_USD_SIZE,
                "max_usd_order_size": settings.MANUAL_TRADING_MAX_USD_ORDER_SIZE,
                "orders_per_burst": settings.MANUAL_TRADING_ORDERS_PER_BURST,
                "tx_sleep_seconds": settings.MANUAL_TRADING_TX_SLEEP_SECONDS,
                "slippage_bps": settings.MANUAL_TRADING_SLIPPAGE_BPS,
                "position_close_threshold_usd": settings.MANUAL_TRADING_POSITION_CLOSE_THRESHOLD_USD,
                "target_reach_tolerance": settings.MANUAL_TRADING_TARGET_REACH_TOLERANCE,
                "max_retries": settings.MANUAL_TRADING_MAX_RETRIES,
                "retry_delay_multiplier": settings.MANUAL_TRADING_RETRY_DELAY_MULTIPLIER,
                "risk_controls_enabled": settings.MANUAL_TRADING_ENABLE_RISK_CONTROLS,
                "max_position_size_usd": settings.MANUAL_TRADING_MAX_POSITION_SIZE_USD,
                "daily_trade_limit": settings.MANUAL_TRADING_DAILY_TRADE_LIMIT
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving manual trading config: {str(e)}")
        return {
            "success": False,
            "message": "Error retrieving manual trading configuration",
            "error": str(e)
        }

@app.get("/api/manual-trading/trades")
async def get_manual_trading_history(limit: int = Query(default=10, description="Number of recent trades to return")):
    """Get recent manual trading history"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            raise HTTPException(status_code=503, detail="Manual trading service not available")
        
        trades = app.state.manual_trading_service.get_recent_trades(limit)
        
        return {
            "success": True,
            "message": f"Retrieved {len(trades)} recent manual trades",
            "data": {
                "total_trades": len(trades),
                "trades": trades
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving manual trading history: {str(e)}")
        return {
            "success": False,
            "message": "Error retrieving manual trading history",
            "error": str(e)
        }

@app.get("/api/manual-trading/active-trades")
async def get_active_manual_trades():
    """Get all active manual trades"""
    try:
        if not hasattr(app.state, 'manual_trading_service'):
            raise HTTPException(status_code=503, detail="Manual trading service not available")
        
        active_trades = await app.state.manual_trading_service.get_active_trades()
        
        return {
            "success": True,
            "data": {
                "active_trades": active_trades,
                "count": len(active_trades)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active manual trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== SOLANA TRADING UTILITIES API ENDPOINTS ==============

@app.get("/api/solana-utils/status")
async def get_solana_utils_status():
    """Get Solana trading utilities service status"""
    if not hasattr(app.state, 'solana_trading_utils_service'):
        raise HTTPException(status_code=404, detail="Solana trading utilities service not initialized")
    
    service = app.state.solana_trading_utils_service
    return {
        "status": "active" if service.is_running else "inactive",
        "service_name": "Solana Trading Utilities Service",
        "uptime_seconds": service.get_uptime(),
        "cache_stats": service.get_cache_stats(),
        "config": {
            "enable_ai_decisions": service.config.get('enable_ai_decisions', False),
            "enable_meme_scoring": service.config.get('enable_meme_scoring', False),
            "ai_confidence_threshold": service.config.get('ai_confidence_threshold', 0.7)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ============== ENHANCED SOLANA TOKEN SCANNER API ENDPOINTS ==============

@app.get("/api/token-scanner/status")
async def get_token_scanner_status():
    """Get enhanced token scanner status"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    return {
        "status": "active" if scanner.is_running else "inactive",
        "service_name": "Enhanced Solana Token Scanner",
        "uptime_seconds": scanner.get_uptime(),
        "stats": scanner.get_stats(),
        "config": {
            "csv_file": scanner.csv_file,
            "export_json": scanner.export_json,
            "heartbeat_interval": scanner.heartbeat_interval,
            "max_seen_signatures": scanner.max_seen_signatures,
            "websocket_url": scanner.websocket_url
        }
    }

@app.post("/api/token-scanner/start")
async def start_token_scanner():
    """Start the enhanced token scanner"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    if scanner.is_running:
        return {"message": "Token scanner is already running", "status": "active"}
    
    try:
        asyncio.create_task(scanner.start())
        return {"message": "Token scanner started successfully", "status": "active"}
    except Exception as e:
        logger.error(f"Failed to start token scanner: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start token scanner: {str(e)}")

@app.post("/api/token-scanner/stop")
async def stop_token_scanner():
    """Stop the enhanced token scanner"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    if not scanner.is_running:
        return {"message": "Token scanner is already stopped", "status": "inactive"}
    
    try:
        await scanner.stop()
        return {"message": "Token scanner stopped successfully", "status": "inactive"}
    except Exception as e:
        logger.error(f"Failed to stop token scanner: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop token scanner: {str(e)}")

@app.get("/api/token-scanner/discovered-tokens")
async def get_discovered_tokens(limit: int = Query(default=50, description="Number of recent tokens to return", ge=1, le=500)):
    """Get recently discovered tokens"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    tokens = scanner.get_recent_tokens(limit)
    
    return {
        "tokens": [asdict(token) for token in tokens],
        "count": len(tokens),
        "total_discovered": scanner.get_stats().get('total_tokens_discovered', 0)
    }

@app.get("/api/token-scanner/export-csv")
async def export_tokens_csv():
    """Export discovered tokens to CSV"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    try:
        file_path = await scanner.export_to_csv()
        return {
            "message": "CSV export completed successfully",
            "file_path": file_path,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export CSV: {str(e)}")

@app.get("/api/token-scanner/export-json")
async def export_tokens_json():
    """Export discovered tokens to JSON"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    try:
        file_path = await scanner.export_to_json()
        return {
            "message": "JSON export completed successfully",
            "file_path": file_path,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export JSON: {str(e)}")

@app.get("/api/token-scanner/stats")
async def get_token_scanner_stats():
    """Get detailed token scanner statistics"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    stats = scanner.get_stats()
    
    return {
        "scanner_stats": stats,
        "uptime_seconds": scanner.get_uptime(),
        "is_running": scanner.is_running,
        "last_heartbeat": scanner.last_heartbeat.isoformat() if scanner.last_heartbeat else None,
        "websocket_status": "connected" if scanner.websocket else "disconnected"
    }

@app.post("/api/token-scanner/clear-data")
async def clear_scanner_data():
    """Clear all discovered token data"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    try:
        cleared_count = scanner.clear_data()
        return {
            "message": f"Cleared {cleared_count} tokens from scanner data",
            "cleared_count": cleared_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear scanner data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear scanner data: {str(e)}")

@app.get("/api/token-scanner/config")
async def get_token_scanner_config():
    """Get token scanner configuration"""
    if not hasattr(app.state, 'enhanced_token_scanner'):
        raise HTTPException(status_code=404, detail="Enhanced token scanner not initialized")
    
    scanner = app.state.enhanced_token_scanner
    return {
        "config": {
            "csv_file": scanner.csv_file,
            "export_json": scanner.export_json,
            "json_export_interval": scanner.json_export_interval,
            "heartbeat_interval": scanner.heartbeat_interval,
            "max_seen_signatures": scanner.max_seen_signatures,
            "websocket_url": scanner.websocket_url,
            "helius_api_key_configured": bool(scanner.helius_api_key),
            "auto_export_enabled": scanner.auto_export_enabled,
            "notification_enabled": scanner.notification_enabled
        },
        "runtime_info": {
            "is_running": scanner.is_running,
            "uptime_seconds": scanner.get_uptime(),
            "last_export": scanner.last_export.isoformat() if scanner.last_export else None
        }
    }

@app.get("/api/ez2/status")
async def get_ez2_status():
    """Get EZ2 trading service status"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            # Initialize EZ2 service if not already done
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
            await app.state.ez2_service.start()
        
        status = await app.state.ez2_service.get_service_status()
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        logger.error(f"Error getting EZ2 status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting EZ2 status: {str(e)}")

@app.post("/api/ez2/execute")
async def execute_ez2_mode(
    mode: int = Body(..., description="Trading mode (0, 1, 2, 4, 5)"),
    symbol: str = Body(default=None, description="Token mint address"),
    target_usd_size: float = Body(default=None, description="Target position size in USD"),
    duration_minutes: int = Body(default=None, description="Duration in minutes for market maker mode")
):
    """Execute a specific EZ2 trading mode"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
            await app.state.ez2_service.start()
        
        # Validate mode
        if mode not in [0, 1, 2, 4, 5]:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Valid modes are 0, 1, 2, 4, 5")
        
        # Execute the mode
        kwargs = {}
        if target_usd_size:
            kwargs['target_usd_size'] = target_usd_size
        if duration_minutes:
            kwargs['duration_minutes'] = duration_minutes
        
        result = await app.state.ez2_service.execute_mode(
            mode=mode,
            symbol=symbol,
            **kwargs
        )
        
        return {
            "success": result.success,
            "data": {
                "mode": result.mode,
                "symbol": result.symbol,
                "message": result.message,
                "error": result.error,
                "final_position_usd": result.final_position_usd,
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing EZ2 mode {mode}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing EZ2 mode {mode}: {str(e)}")

@app.post("/api/ez2/close-position")
async def close_position_ez2(
    symbol: str = Body(..., description="Token mint address to close position for")
):
    """Close position for specified symbol (Mode 0)"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
            await app.state.ez2_service.start()
        
        result = await app.state.ez2_service.execute_mode(mode=0, symbol=symbol)
        
        return {
            "success": result.success,
            "data": {
                "mode": result.mode,
                "symbol": result.symbol,
                "message": result.message,
                "error": result.error,
                "final_position_usd": result.final_position_usd,
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=f"Error closing position: {str(e)}")

@app.post("/api/ez2/buy-position")
async def buy_position_ez2(
    symbol: str = Body(..., description="Token mint address to buy"),
    target_usd_size: float = Body(default=None, description="Target position size in USD")
):
    """Open buying position for specified symbol (Mode 1)"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
            await app.state.ez2_service.start()
        
        result = await app.state.ez2_service.execute_mode(
            mode=1, 
            symbol=symbol, 
            target_usd_size=target_usd_size
        )
        
        return {
            "success": result.success,
            "data": {
                "mode": result.mode,
                "symbol": result.symbol,
                "message": result.message,
                "error": result.error,
                "final_position_usd": result.final_position_usd,
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buying position: {e}")
        raise HTTPException(status_code=500, detail=f"Error buying position: {str(e)}")

@app.post("/api/ez2/pnl-close")
async def pnl_close_ez2():
    """Close positions based on PnL thresholds (Mode 4)"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
            await app.state.ez2_service.start()
        
        result = await app.state.ez2_service.execute_mode(mode=4)
        
        return {
            "success": result.success,
            "data": {
                "mode": result.mode,
                "symbol": result.symbol,
                "message": result.message,
                "error": result.error,
                "final_position_usd": result.final_position_usd,
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing PnL close: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing PnL close: {str(e)}")

@app.post("/api/ez2/market-maker/start")
async def start_market_maker_ez2(
    symbol: str = Body(..., description="Token mint address for market making"),
    duration_minutes: int = Body(default=None, description="Duration in minutes (optional)")
):
    """Start market maker mode for specified symbol (Mode 5)"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
            await app.state.ez2_service.start()
        
        if duration_minutes:
            # Run for specific duration
            result = await app.state.ez2_service.execute_mode(
                mode=5, 
                symbol=symbol, 
                duration_minutes=duration_minutes
            )
            return {
                "success": result.success,
                "data": {
                    "message": result.message,
                    "error": result.error,
                    "duration_minutes": duration_minutes
                }
            }
        else:
            # Start continuous market maker
            message = await app.state.ez2_service.start_continuous_market_maker(symbol)
            return {
                "success": True,
                "data": {
                    "message": message,
                    "symbol": symbol,
                    "continuous": True
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting market maker: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting market maker: {str(e)}")

@app.post("/api/ez2/market-maker/stop")
async def stop_market_maker_ez2():
    """Stop the active market maker"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            raise HTTPException(status_code=503, detail="EZ2 service not available")
        
        message = await app.state.ez2_service.stop_market_maker()
        return {
            "success": True,
            "data": {
                "message": message
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping market maker: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping market maker: {str(e)}")

@app.get("/api/ez2/modes")
async def get_available_modes_ez2():
    """Get list of available trading modes"""
    return {
        "success": True,
        "data": {
            "modes": {
                "0": {
                    "name": "Close Position",
                    "description": "Close position for specified symbol in chunks",
                    "parameters": ["symbol"]
                },
                "1": {
                    "name": "Buy Position", 
                    "description": "Open buying position up to target size",
                    "parameters": ["symbol", "target_usd_size (optional)"]
                },
                "2": {
                    "name": "ETH SMA Strategy",
                    "description": "Execute trades based on ETH SMA signals",
                    "parameters": [],
                    "status": "Not implemented - requires ETH data source"
                },
                "4": {
                    "name": "PnL Close",
                    "description": "Close positions based on portfolio PnL thresholds",
                    "parameters": []
                },
                "5": {
                    "name": "Market Maker",
                    "description": "Simple market making with buy under/sell over logic",
                    "parameters": ["symbol", "duration_minutes (optional)"]
                }
            }
        }
    }

@app.get("/api/ez2/config")
async def get_ez2_config():
    """Get current EZ2 trading configuration"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
            await app.state.ez2_service.start()
        
        status = await app.state.ez2_service.get_service_status()
        return {
            "success": True,
            "data": {
                "configuration": status.get("configuration", {}),
                "primary_symbol": status.get("configuration", {}).get("primary_symbol"),
                "trading_parameters": {
                    "usd_size": status.get("configuration", {}).get("usd_size"),
                    "max_usd_order_size": status.get("configuration", {}).get("max_usd_order_size"),
                    "slippage": status.get("configuration", {}).get("slippage"),
                    "orders_per_open": status.get("configuration", {}).get("orders_per_open")
                },
                "thresholds": {
                    "lowest_balance": status.get("configuration", {}).get("lowest_balance"),
                    "target_balance": status.get("configuration", {}).get("target_balance"),
                    "buy_under": status.get("configuration", {}).get("buy_under"),
                    "sell_over": status.get("configuration", {}).get("sell_over")
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

@app.post("/api/ez2/start")
async def start_ez2_service():
    """Start the EZ2 trading service"""
    try:
        if not hasattr(app.state, 'ez2_service'):
            from services.ez2_service import EZ2SolanaTradingService
            app.state.ez2_service = EZ2SolanaTradingService()
        
        await app.state.ez2_service.start()
        
        return {
            "success": True,
            "data": {
                "message": "EZ2 trading service started",
                "status": "active" if app.state.ez2_service.is_running else "inactive"
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting EZ2 service: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting EZ2 service: {str(e)}")

@app.post("/api/ez2/stop")
async def stop_ez2_service():
    """Stop the EZ2 trading service"""
    try:
        if hasattr(app.state, 'ez2_service'):
            await app.state.ez2_service.stop()
            delattr(app.state, 'ez2_service')
        
        return {
            "success": True,
            "data": {
                "message": "EZ2 trading service stopped"
            }
        }
        
    except Exception as e:
        logger.error(f"Error stopping EZ2 service: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping EZ2 service: {str(e)}")

@app.get("/api/solana-utils/status")
async def get_solana_utils_status():
    """Get Solana trading utilities service status"""
    try:
        if not hasattr(app.state, 'solana_trading_utils_service'):
            raise HTTPException(status_code=503, detail="Solana trading utilities service not available")
        
        status = await app.state.solana_trading_utils_service.get_service_status()
        
        return {
            "success": True,
            "data": status
        }
        
    except Exception as e:
        logger.error(f"Error getting Solana utils status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== TWITTER SOLANA BOT API ENDPOINTS ==============

@app.get("/api/twitter-solana-bot/status")
async def get_twitter_solana_bot_status():
    """Get Twitter Solana Bot service status and dependency check"""
    try:
        from .services.twitter_solana_bot_service import twitter_solana_bot_service
        status = twitter_solana_bot_service.get_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"Error getting Twitter Solana Bot status: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/twitter-solana-bot/test-contract-detection")
async def test_contract_detection(request: dict):
    """Test contract address detection on sample text"""
    try:
        from .services.twitter_solana_bot_service import twitter_solana_bot_service
        text = request.get("text", "")
        if not text:
            return {"status": "error", "message": "Text parameter required"}
        
        result = twitter_solana_bot_service.simulate_tweet_detection(text)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error testing contract detection: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/twitter-solana-bot/config")
async def get_twitter_solana_bot_config():
    """Get current Twitter Solana Bot configuration"""
    try:
        from .services.twitter_solana_bot_service import twitter_solana_bot_service
        config = twitter_solana_bot_service.get_configuration()
        return {"status": "success", "data": config}
    except Exception as e:
        logger.error(f"Error getting Twitter Solana Bot config: {e}")
        return {"status": "error", "message": str(e)}

# ============== RRS ANALYSIS API ENDPOINTS ==============

@app.get("/api/rrs-analysis/status")
async def get_rrs_analysis_status():
    """Get RRS Analysis service status and configuration"""
    try:
        from .services.rrs_analysis_service import rrs_service
        status = rrs_service.get_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"Error getting RRS Analysis status: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/rrs-analysis/run-full-analysis")
async def run_full_rrs_analysis():
    """Run complete RRS analysis across all configured timeframes"""
    try:
        from .services.rrs_analysis_service import rrs_service
        
        if rrs_service.is_running:
            return {"status": "error", "message": "RRS analysis is already running"}
        
        result = await rrs_service.run_full_analysis()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error running RRS analysis: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/rrs-analysis/rankings")
async def get_rrs_rankings(timeframe: str = None):
    """Get latest RRS rankings for a specific timeframe or consolidated results"""
    try:
        from .services.rrs_analysis_service import rrs_service
        result = await rrs_service.get_latest_rrs_rankings(timeframe)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting RRS rankings: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/rrs-analysis/process-timeframe")
async def process_rrs_timeframe(request: dict):
    """Process RRS analysis for a specific timeframe"""
    try:
        from .services.rrs_analysis_service import rrs_service
        
        timeframe = request.get("timeframe", settings.RRS_DEFAULT_TIMEFRAME)
        lookback_days = request.get("lookback_days", settings.RRS_LOOKBACK_DAYS)
        benchmark_symbol = request.get("benchmark_symbol", settings.RRS_BENCHMARK_SYMBOL)
        
        if rrs_service.is_running:
            return {"status": "error", "message": "RRS analysis is already running"}
        
        rrs_service.is_running = True
        try:
            await rrs_service.process_timeframe(timeframe, lookback_days, benchmark_symbol)
            return {
                "status": "success", 
                "message": f"RRS analysis completed for {timeframe}",
                "data": {
                    "timeframe": timeframe,
                    "lookback_days": lookback_days,
                    "benchmark_symbol": benchmark_symbol
                }
            }
        finally:
            rrs_service.is_running = False
            
    except Exception as e:
        logger.error(f"Error processing RRS timeframe: {e}")
        rrs_service.is_running = False
        return {"status": "error", "message": str(e)}

@app.post("/api/rrs-analysis/interpret-score")
async def interpret_rrs_score(request: dict):
    """Interpret an RRS score based on configured thresholds"""
    try:
        from .services.rrs_analysis_service import rrs_service
        
        score = request.get("score")
        if score is None:
            return {"status": "error", "message": "Score parameter required"}
        
        try:
            score = float(score)
        except (ValueError, TypeError):
            return {"status": "error", "message": "Score must be a valid number"}
        
        interpretation = rrs_service.interpret_rrs_score(score)
        return {
            "status": "success", 
            "data": {
                "score": score,
                "interpretation": interpretation,
                "thresholds": {
                    "exceptional": settings.RRS_EXCEPTIONAL_THRESHOLD,
                    "strong": settings.RRS_STRONG_THRESHOLD,
                    "moderate": settings.RRS_MODERATE_THRESHOLD,
                    "weak": settings.RRS_WEAK_THRESHOLD
                }
            }
        }
    except Exception as e:
        logger.error(f"Error interpreting RRS score: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/rrs-analysis/symbols")
async def get_rrs_symbols():
    """Get configured RRS symbols and their addresses"""
    try:
        return {
            "status": "success", 
            "data": {
                "symbols": settings.RRS_SYMBOLS,
                "benchmark_symbol": settings.RRS_BENCHMARK_SYMBOL,
                "total_count": len(settings.RRS_SYMBOLS)
            }
        }
    except Exception as e:
        logger.error(f"Error getting RRS symbols: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/rrs-analysis/config")
async def get_rrs_analysis_config():
    """Get RRS analysis configuration"""
    try:
        from services.rrs_analysis_service import rrs_service
        return rrs_service.get_config()
    except Exception as e:
        logger.error(f"Error getting RRS config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== HYPERLIQUID RRS ANALYSIS ENDPOINTS ==============

@app.get("/api/hyperliquid-rrs-analysis/status")
async def get_hyperliquid_rrs_analysis_status():
    """Get Hyperliquid RRS analysis service status"""
    try:
        return hyperliquid_rrs_service.get_status()
    except Exception as e:
        logger.error(f"Error getting Hyperliquid RRS status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-rrs-analysis/run-full-analysis")
async def run_full_hyperliquid_rrs_analysis():
    """Run complete Hyperliquid RRS analysis across all timeframes"""
    try:
        await hyperliquid_rrs_service.run_full_analysis()
        return {"status": "success", "message": "Hyperliquid RRS analysis completed"}
    except Exception as e:
        logger.error(f"Error running Hyperliquid RRS analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-rrs-analysis/rankings")
async def get_hyperliquid_rrs_rankings(timeframe: str = None):
    """Get Hyperliquid RRS rankings for specific timeframe or consolidated rankings"""
    try:
        return hyperliquid_rrs_service.get_rankings(timeframe)
    except Exception as e:
        logger.error(f"Error getting Hyperliquid RRS rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-rrs-analysis/process-timeframe")
async def process_hyperliquid_rrs_timeframe(request: dict):
    """Process Hyperliquid RRS analysis for specific timeframe"""
    try:
        timeframe = request.get('timeframe')
        lookback_days = request.get('lookback_days', settings.HYPERLIQUID_RRS_LOOKBACK_DAYS)
        benchmark_symbol = request.get('benchmark_symbol', settings.HYPERLIQUID_RRS_BENCHMARK_SYMBOL)
        
        if not timeframe:
            raise HTTPException(status_code=400, detail="Timeframe is required")
        
        if timeframe not in settings.HYPERLIQUID_RRS_TIMEFRAMES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timeframe. Must be one of: {settings.HYPERLIQUID_RRS_TIMEFRAMES}"
            )
        
        await hyperliquid_rrs_service.process_timeframe(timeframe, lookback_days, benchmark_symbol)
        
        return {
            "status": "success",
            "message": f"Hyperliquid RRS analysis completed for timeframe {timeframe}",
            "timeframe": timeframe,
            "lookback_days": lookback_days,
            "benchmark_symbol": benchmark_symbol
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Hyperliquid RRS timeframe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-rrs-analysis/interpret-score")
async def interpret_hyperliquid_rrs_score(request: dict):
    """Interpret Hyperliquid RRS score using configured thresholds"""
    try:
        rrs_score = request.get('rrs_score')
        
        if rrs_score is None:
            raise HTTPException(status_code=400, detail="rrs_score is required")
        
        try:
            rrs_score = float(rrs_score)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="rrs_score must be a valid number")
        
        interpretation = hyperliquid_rrs_service.interpret_score(rrs_score)
        
        return {
            "status": "success",
            "interpretation": interpretation
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error interpreting Hyperliquid RRS score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-rrs-analysis/symbols")
async def get_hyperliquid_rrs_symbols():
    """Get configured Hyperliquid RRS symbols"""
    try:
        return hyperliquid_rrs_service.get_symbols()
    except Exception as e:
        logger.error(f"Error getting Hyperliquid RRS symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-rrs-analysis/config")
async def get_hyperliquid_rrs_analysis_config():
    """Get Hyperliquid RRS analysis configuration"""
    try:
        return hyperliquid_rrs_service.get_config()
    except Exception as e:
        logger.error(f"Error getting Hyperliquid RRS config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== HYPERLIQUID RRS BOT ENDPOINTS ==============

@app.get("/api/hyperliquid-rrs-bot/status")
async def get_hyperliquid_rrs_bot_status():
    """Get Hyperliquid RRS bot status"""
    try:
        return hyperliquid_rrs_bot_service.get_status()
    except Exception as e:
        logger.error(f"Error getting Hyperliquid RRS bot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-rrs-bot/start")
async def start_hyperliquid_rrs_bot():
    """Start the Hyperliquid RRS trading bot"""
    try:
        await hyperliquid_rrs_bot_service.start()
        return {"status": "success", "message": "Hyperliquid RRS bot started"}
    except Exception as e:
        logger.error(f"Error starting Hyperliquid RRS bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-rrs-bot/stop")
async def stop_hyperliquid_rrs_bot():
    """Stop the Hyperliquid RRS trading bot"""
    try:
        await hyperliquid_rrs_bot_service.stop()
        return {"status": "success", "message": "Hyperliquid RRS bot stopped"}
    except Exception as e:
        logger.error(f"Error stopping Hyperliquid RRS bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-rrs-bot/positions")
async def get_hyperliquid_rrs_bot_positions():
    """Get current bot positions"""
    try:
        return {
            "status": "success",
            "positions": hyperliquid_rrs_bot_service.get_positions()
        }
    except Exception as e:
        logger.error(f"Error getting bot positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-rrs-bot/trading-state")
async def get_hyperliquid_rrs_bot_trading_state():
    """Get complete trading state"""
    try:
        return {
            "status": "success",
            "trading_state": hyperliquid_rrs_bot_service.get_trading_state()
        }
    except Exception as e:
        logger.error(f"Error getting trading state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-rrs-bot/force-rrs-update")
async def force_hyperliquid_rrs_update():
    """Force update of RRS data"""
    try:
        rrs_data = await hyperliquid_rrs_bot_service.force_rrs_update()
        return {
            "status": "success",
            "message": "RRS data updated",
            "rrs_data": rrs_data
        }
    except Exception as e:
        logger.error(f"Error forcing RRS update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-rrs-bot/close-all-positions")
async def close_all_hyperliquid_rrs_bot_positions():
    """Close all open positions"""
    try:
        closed_count = await hyperliquid_rrs_bot_service.close_all_positions("Manual close via API")
        return {
            "status": "success",
            "message": f"Closed {closed_count} positions",
            "closed_positions": closed_count
        }
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== HYPERLIQUID UTILS ENDPOINTS ==============

from services.hyperliquid_utils_service import hyperliquid_utils_service

@app.get("/api/hyperliquid-utils/status")
async def get_hyperliquid_utils_status():
    """Get Hyperliquid Utils service status"""
    try:
        return {
            "status": "success",
            "service": "hyperliquid_utils",
            "account_configured": hyperliquid_utils_service.account is not None,
            "exchange_configured": hyperliquid_utils_service.exchange is not None,
            "info_configured": hyperliquid_utils_service.info is not None
        }
    except Exception as e:
        logger.error(f"Error getting Hyperliquid Utils status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/ask-bid/{symbol}")
async def get_ask_bid(symbol: str):
    """Get ask/bid prices and L2 order book data for a symbol"""
    try:
        async with hyperliquid_utils_service as service:
            ask, bid, levels = await service.ask_bid(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "ask": ask,
                "bid": bid,
                "spread": ask - bid,
                "spread_pct": ((ask - bid) / bid) * 100,
                "l2_levels": levels
            }
    except Exception as e:
        logger.error(f"Error getting ask/bid for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/spot-info/{symbol}")
async def get_spot_info(symbol: str):
    """Get spot price and symbol information"""
    try:
        async with hyperliquid_utils_service as service:
            mid_px, hoe_ass_symbol, sz_decimals, px_decimals = await service.spot_price_and_symbol_info(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "mid_price": mid_px,
                "trading_symbol": hoe_ass_symbol,
                "size_decimals": sz_decimals,
                "price_decimals": px_decimals
            }
    except Exception as e:
        logger.error(f"Error getting spot info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/decimals/{symbol}")
async def get_symbol_decimals(symbol: str):
    """Get size and price decimals for a symbol"""
    try:
        async with hyperliquid_utils_service as service:
            sz_decimals, px_decimals = await service.get_sz_px_decimals(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "size_decimals": sz_decimals,
                "price_decimals": px_decimals
            }
    except Exception as e:
        logger.error(f"Error getting decimals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/spot-limit-order")
async def place_spot_limit_order(request: dict):
    """Place a spot limit order"""
    try:
        coin = request.get('coin')
        is_buy = request.get('is_buy')
        size = request.get('size')
        limit_price = request.get('limit_price')
        sz_decimals = request.get('sz_decimals')
        px_decimals = request.get('px_decimals')
        
        if not all([coin, is_buy is not None, size, limit_price, sz_decimals is not None, px_decimals is not None]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        async with hyperliquid_utils_service as service:
            result = await service.spot_limit_order(coin, is_buy, size, limit_price, sz_decimals, px_decimals)
            return {
                "status": "success",
                "order_result": result
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing spot limit order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/limit-order")
async def place_limit_order(request: dict):
    """Place a futures limit order"""
    try:
        coin = request.get('coin')
        is_buy = request.get('is_buy')
        size = request.get('size')
        limit_price = request.get('limit_price')
        reduce_only = request.get('reduce_only', False)
        
        if not all([coin, is_buy is not None, size, limit_price]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        async with hyperliquid_utils_service as service:
            result = await service.limit_order(coin, is_buy, size, limit_price, reduce_only)
            return {
                "status": "success",
                "order_result": result
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing limit order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/adjust-leverage")
async def adjust_leverage(request: dict):
    """Adjust leverage for a symbol"""
    try:
        symbol = request.get('symbol')
        leverage = request.get('leverage')
        
        if not all([symbol, leverage]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        async with hyperliquid_utils_service as service:
            result = await service.adjust_leverage(symbol, leverage)
            return {
                "status": "success",
                "symbol": symbol,
                "leverage": leverage,
                "result": result
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adjusting leverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/calculate-size-95pct")
async def calculate_size_95_percent(request: dict):
    """Calculate position size based on 95% of account balance"""
    try:
        symbol = request.get('symbol')
        leverage = request.get('leverage')
        
        if not all([symbol, leverage]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        async with hyperliquid_utils_service as service:
            leverage_result, size = await service.adjust_leverage_size_signal(symbol, leverage)
            return {
                "status": "success",
                "symbol": symbol,
                "leverage": leverage_result,
                "calculated_size": size,
                "method": "95% of account balance"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating size: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/calculate-size-usd")
async def calculate_size_usd(request: dict):
    """Calculate position size based on specific USD amount"""
    try:
        symbol = request.get('symbol')
        usd_size = request.get('usd_size')
        leverage = request.get('leverage')
        
        if not all([symbol, usd_size, leverage]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        async with hyperliquid_utils_service as service:
            leverage_result, size = await service.adjust_leverage_usd_size(symbol, usd_size, leverage)
            return {
                "status": "success",
                "symbol": symbol,
                "leverage": leverage_result,
                "calculated_size": size,
                "usd_amount": usd_size,
                "method": "specific USD amount"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating USD size: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/ohlcv/{symbol}")
async def get_ohlcv_data(symbol: str, interval: str = "1h", lookback_days: int = 30):
    """Fetch OHLCV candlestick data"""
    try:
        async with hyperliquid_utils_service as service:
            snapshot_data = await service.get_ohlcv_data(symbol, interval, lookback_days)
            if not snapshot_data:
                raise HTTPException(status_code=404, detail="No data found")
            
            df = service.process_data_to_df(snapshot_data)
            return {
                "status": "success",
                "symbol": symbol,
                "interval": interval,
                "lookback_days": lookback_days,
                "data_points": len(snapshot_data),
                "raw_data": snapshot_data,
                "processed_data": df.to_dict('records') if not df.empty else []
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OHLCV data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/vwap/{symbol}")
async def calculate_vwap(symbol: str):
    """Calculate VWAP for a symbol"""
    try:
        async with hyperliquid_utils_service as service:
            df, latest_vwap = await service.calculate_vwap(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "latest_vwap": latest_vwap,
                "data_points": len(df),
                "vwap_data": df['VWAP'].tail(20).to_dict() if 'VWAP' in df.columns else {}
            }
    except Exception as e:
        logger.error(f"Error calculating VWAP for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/position/{symbol}")
async def get_position(symbol: str):
    """Get current position information"""
    try:
        async with hyperliquid_utils_service as service:
            positions, in_pos, size, pos_sym, entry_px, pnl_perc, long = await service.get_position(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "in_position": in_pos,
                "size": size,
                "position_symbol": pos_sym,
                "entry_price": entry_px,
                "pnl_percentage": pnl_perc,
                "is_long": long,
                "positions": positions
            }
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/spot-position/{symbol}")
async def get_spot_position(symbol: str):
    """Get current spot position information"""
    try:
        async with hyperliquid_utils_service as service:
            positions, in_pos, size, pos_sym, entry_px, pnl_perc, long = await service.get_spot_position(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "in_position": in_pos,
                "size": size,
                "position_symbol": pos_sym,
                "entry_price": entry_px,
                "pnl_percentage": pnl_perc,
                "is_long": long,
                "positions": positions
            }
    except Exception as e:
        logger.error(f"Error getting spot position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/cancel-all-orders")
async def cancel_all_orders():
    """Cancel all open orders"""
    try:
        async with hyperliquid_utils_service as service:
            await service.cancel_all_orders()
            return {
                "status": "success",
                "message": "All orders cancelled"
            }
    except Exception as e:
        logger.error(f"Error cancelling all orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/cancel-symbol-orders/{symbol}")
async def cancel_symbol_orders(symbol: str):
    """Cancel all open orders for a specific symbol"""
    try:
        async with hyperliquid_utils_service as service:
            await service.cancel_symbol_orders(symbol)
            return {
                "status": "success",
                "message": f"All orders cancelled for {symbol}"
            }
    except Exception as e:
        logger.error(f"Error cancelling orders for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/kill-switch/{symbol}")
async def kill_switch(symbol: str):
    """Emergency position close"""
    try:
        async with hyperliquid_utils_service as service:
            await service.kill_switch(symbol)
            return {
                "status": "success",
                "message": f"Kill switch executed for {symbol}"
            }
    except Exception as e:
        logger.error(f"Error executing kill switch for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/pnl-close")
async def pnl_close(request: dict):
    """Close position based on PnL thresholds"""
    try:
        symbol = request.get('symbol')
        target = request.get('target')
        max_loss = request.get('max_loss')
        
        if not all([symbol, target is not None, max_loss is not None]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        async with hyperliquid_utils_service as service:
            await service.pnl_close(symbol, target, max_loss)
            return {
                "status": "success",
                "message": f"PnL close check completed for {symbol}",
                "target": target,
                "max_loss": max_loss
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in PnL close: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/atr/{symbol}")
async def calculate_atr(symbol: str, window: int = 14, lookback_days: int = 30):
    """Calculate Average True Range"""
    try:
        async with hyperliquid_utils_service as service:
            df, last_atr = await service.calculate_atr(symbol, window, lookback_days)
            return {
                "status": "success",
                "symbol": symbol,
                "atr": last_atr,
                "window": window,
                "lookback_days": lookback_days,
                "data_points": len(df)
            }
    except Exception as e:
        logger.error(f"Error calculating ATR for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/bollinger-bands/{symbol}")
async def calculate_bollinger_bands(symbol: str, length: int = 20, std_dev: int = 2, lookback_days: int = 30):
    """Calculate Bollinger Bands and classify tight/wide"""
    try:
        async with hyperliquid_utils_service as service:
            df, tight, wide = await service.calculate_bollinger_bands(symbol, length, std_dev, lookback_days)
            return {
                "status": "success",
                "symbol": symbol,
                "tight": tight,
                "wide": wide,
                "length": length,
                "std_dev": std_dev,
                "lookback_days": lookback_days,
                "current_band_width": df['BandWidth'].iloc[-1] if 'BandWidth' in df.columns else None,
                "data_points": len(df)
            }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/spot-symbols")
async def get_all_spot_symbols():
    """Get all available spot symbols"""
    try:
        async with hyperliquid_utils_service as service:
            symbols = await service.get_all_spot_symbols()
            return {
                "status": "success",
                "symbols": symbols,
                "count": len(symbols)
            }
    except Exception as e:
        logger.error(f"Error getting spot symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hyperliquid-utils/order-book-analysis/{symbol}")
async def analyze_order_book(symbol: str):
    """Get order book data from multiple exchanges"""
    try:
        async with hyperliquid_utils_service as service:
            (combined_df_binance, combined_df_bybit, combined_df_coinbase, 
             max_bid_price, max_ask_price, bid_before_biggest, ask_before_biggest) = await service.ob_data(symbol)
            
            return {
                "status": "success",
                "symbol": symbol,
                "max_bid_price": max_bid_price,
                "max_ask_price": max_ask_price,
                "bid_before_biggest": bid_before_biggest,
                "ask_before_biggest": ask_before_biggest,
                "binance_data": combined_df_binance.head(10).to_dict('records'),
                "bybit_data": combined_df_bybit.head(10).to_dict('records'),
                "coinbase_data": combined_df_coinbase.head(10).to_dict('records')
            }
    except Exception as e:
        logger.error(f"Error analyzing order book for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/open-order-deluxe")
async def open_order_deluxe(request: dict):
    """Place limit order with stop loss and take profit"""
    try:
        symbol = request.get('symbol')
        entry_price = request.get('entry_price')
        stop_loss = request.get('stop_loss')
        take_profit = request.get('take_profit')
        size = request.get('size')
        
        if not all([symbol, entry_price, stop_loss, take_profit, size]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        async with hyperliquid_utils_service as service:
            result = await service.open_order_deluxe(symbol, entry_price, stop_loss, take_profit, size)
            return {
                "status": "success",
                "symbol": symbol,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "size": size,
                "order_results": result
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing deluxe order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hyperliquid-utils/create-symbol-info")
async def create_symbol_info(request: dict):
    """Create symbol info for Hyperliquid trading"""
    try:
        symbol = request.get("symbol")
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")
        
        result = await hyperliquid_utils_service.create_symbol_info(symbol)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error creating symbol info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== EARLY BUYER TRACKER ENDPOINTS ====================

@app.get("/api/early-buyer-tracker/status")
async def get_early_buyer_tracker_status():
    """Get early buyer tracker service status"""
    try:
        config = early_buyer_tracker_service.get_config()
        return {
            "status": "success",
            "service": "Early Buyer Tracker",
            "version": "1.0.0",
            "config": config
        }
    except Exception as e:
        logger.error(f"Error getting early buyer tracker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/early-buyer-tracker/track")
async def track_early_buyers(request: dict):
    """
    Track early buyers for a token
    
    Body:
    {
        "token_address": "string",
        "start_date": "MM-DD-YYYY",
        "end_date": "MM-DD-YYYY",
        "min_trade_size": 3000.0,
        "max_trade_size": 100000000.0,
        "sort_type": "asc",
        "save_to_file": true
    }
    """
    try:
        token_address = request.get("token_address")
        if not token_address:
            raise HTTPException(status_code=400, detail="Token address is required")
        
        start_date = request.get("start_date", "01-01-2020")
        end_date = request.get("end_date", "12-31-2030")
        min_trade_size = request.get("min_trade_size")
        max_trade_size = request.get("max_trade_size")
        sort_type = request.get("sort_type")
        save_to_file = request.get("save_to_file", True)
        
        result = await early_buyer_tracker_service.track_early_buyers(
            token_address=token_address,
            start_date=start_date,
            end_date=end_date,
            min_trade_size=min_trade_size,
            max_trade_size=max_trade_size,
            sort_type=sort_type,
            save_to_file=save_to_file
        )
        
        return result
    except Exception as e:
        logger.error(f"Error tracking early buyers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/early-buyer-tracker/analyses")
async def get_saved_analyses(limit: int = Query(default=20, description="Number of analyses to return", ge=1, le=100)):
    """Get list of saved early buyer analyses"""
    try:
        analyses = await early_buyer_tracker_service.get_saved_analyses(limit)
        return {
            "status": "success",
            "analyses": analyses,
            "count": len(analyses)
        }
    except Exception as e:
        logger.error(f"Error getting saved analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/early-buyer-tracker/analysis/{filename}")
async def load_analysis(filename: str):
    """Load a specific analysis from CSV file"""
    try:
        df = await early_buyer_tracker_service.load_analysis(filename)
        if df is None:
            raise HTTPException(status_code=404, detail="Analysis file not found")
        
        # Convert to dict for JSON response
        data = df.to_dict('records')
        
        # Calculate summary stats
        summary = {
            "total_trades": len(df),
            "unique_buyers": df['Owner'].nunique() if not df.empty else 0,
            "total_volume_usd": df['USD Value'].sum() if not df.empty else 0,
            "avg_trade_size_usd": df['USD Value'].mean() if not df.empty else 0,
            "earliest_trade": df['Timestamp'].min() if not df.empty else None,
            "latest_trade": df['Timestamp'].max() if not df.empty else None
        }
        
        return {
            "status": "success",
            "filename": filename,
            "summary": summary,
            "data": data[:1000] if len(data) > 1000 else data,  # Limit for performance
            "total_records": len(data)
        }
    except Exception as e:
        logger.error(f"Error loading analysis {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/early-buyer-tracker/quick-track/{token_address}")
async def quick_track_early_buyers(
    token_address: str,
    min_trade_size: float = Query(default=3000.0, description="Minimum trade size in USD"),
    days_back: int = Query(default=30, description="Number of days to look back", ge=1, le=365)
):
    """Quick track early buyers for the last N days"""
    try:
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_date_str = start_date.strftime("%m-%d-%Y")
        end_date_str = end_date.strftime("%m-%d-%Y")
        
        result = await early_buyer_tracker_service.track_early_buyers(
            token_address=token_address,
            start_date=start_date_str,
            end_date=end_date_str,
            min_trade_size=min_trade_size,
            save_to_file=True
        )
        
        return result
    except Exception as e:
        logger.error(f"Error quick tracking early buyers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/early-buyer-tracker/config")
async def get_early_buyer_tracker_config():
    """Get early buyer tracker configuration"""
    try:
        config = early_buyer_tracker_service.get_config()
        return {
            "status": "success",
            "config": config
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/early-buyer-tracker/batch-track")
async def batch_track_early_buyers(request: dict):
    """
    Track early buyers for multiple tokens
    
    Body:
    {
        "token_addresses": ["address1", "address2", ...],
        "start_date": "MM-DD-YYYY",
        "end_date": "MM-DD-YYYY",
        "min_trade_size": 3000.0,
        "max_trade_size": 100000000.0,
        "sort_type": "asc"
    }
    """
    try:
        token_addresses = request.get("token_addresses", [])
        if not token_addresses:
            raise HTTPException(status_code=400, detail="Token addresses list is required")
        
        start_date = request.get("start_date", "01-01-2020")
        end_date = request.get("end_date", "12-31-2030")
        min_trade_size = request.get("min_trade_size")
        max_trade_size = request.get("max_trade_size")
        sort_type = request.get("sort_type")
        
        results = []
        for token_address in token_addresses:
            try:
                result = await early_buyer_tracker_service.track_early_buyers(
                    token_address=token_address,
                    start_date=start_date,
                    end_date=end_date,
                    min_trade_size=min_trade_size,
                    max_trade_size=max_trade_size,
                    sort_type=sort_type,
                    save_to_file=True
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "token_address": token_address,
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "results": results,
            "total_tokens": len(token_addresses),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "error"])
        }
    except Exception as e:
        logger.error(f"Error batch tracking early buyers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Load settings
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    ) 