"""
Advanced Analytics API Routes - ML insights and arbitrage monitoring
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

router = APIRouter()

class MLInsights(BaseModel):
    strategy_name: str
    model_type: str
    prediction_accuracy: float
    feature_importance: Dict[str, float]
    last_prediction: Optional[Dict[str, Any]]
    training_status: str
    confidence_distribution: Dict[str, int]


class ArbitrageMetrics(BaseModel):
    total_opportunities: int
    successful_executions: int
    success_rate: float
    total_profit: float
    avg_profit_per_trade: float
    top_exchanges: List[Dict[str, Any]]
    recent_opportunities: List[Dict[str, Any]]


class PerformanceAnalytics(BaseModel):
    strategy_comparison: Dict[str, Dict[str, float]]
    risk_adjusted_returns: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    drawdown_analysis: Dict[str, Any]
    volatility_metrics: Dict[str, float]


@router.get("/ml-insights", response_model=List[MLInsights])
async def get_ml_insights():
    """Get machine learning strategy insights and performance"""
    try:
        # Mock ML insights data
        insights = [
            MLInsights(
                strategy_name="LSTM_ML",
                model_type="LSTM Neural Network",
                prediction_accuracy=0.73,
                feature_importance={
                    "close_price": 0.25,
                    "volume": 0.18,
                    "rsi": 0.15,
                    "macd": 0.12,
                    "bollinger_bands": 0.10,
                    "volatility": 0.08,
                    "time_features": 0.07,
                    "other": 0.05
                },
                last_prediction={
                    "symbol": "BTC",
                    "predicted_change": 2.3,
                    "confidence": 0.78,
                    "timestamp": datetime.utcnow().isoformat(),
                    "actual_outcome": None
                },
                training_status="completed",
                confidence_distribution={
                    "high_confidence": 45,
                    "medium_confidence": 35,
                    "low_confidence": 20
                }
            ),
            MLInsights(
                strategy_name="DQN_RL",
                model_type="Deep Q-Network",
                prediction_accuracy=0.68,
                feature_importance={
                    "state_representation": 0.30,
                    "reward_signal": 0.25,
                    "action_history": 0.20,
                    "market_conditions": 0.15,
                    "volatility": 0.10
                },
                last_prediction={
                    "symbol": "ETH",
                    "action": "BUY",
                    "q_value": 0.85,
                    "epsilon": 0.15,
                    "timestamp": datetime.utcnow().isoformat()
                },
                training_status="training",
                confidence_distribution={
                    "high_confidence": 38,
                    "medium_confidence": 42,
                    "low_confidence": 20
                }
            )
        ]
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ML insights: {str(e)}")


@router.get("/arbitrage-metrics", response_model=ArbitrageMetrics)
async def get_arbitrage_metrics():
    """Get arbitrage strategy performance metrics"""
    try:
        # Mock arbitrage metrics
        metrics = ArbitrageMetrics(
            total_opportunities=156,
            successful_executions=142,
            success_rate=0.91,
            total_profit=12450.75,
            avg_profit_per_trade=87.68,
            top_exchanges=[
                {"exchange": "binance", "opportunities": 45, "success_rate": 0.95},
                {"exchange": "coinbase", "opportunities": 38, "success_rate": 0.89},
                {"exchange": "kraken", "opportunities": 32, "success_rate": 0.87},
                {"exchange": "hyperliquid", "opportunities": 41, "success_rate": 0.93}
            ],
            recent_opportunities=[
                {
                    "symbol": "BTC",
                    "buy_exchange": "kraken",
                    "sell_exchange": "binance",
                    "profit_pct": 0.65,
                    "volume": 2500,
                    "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat()
                },
                {
                    "symbol": "ETH",
                    "buy_exchange": "coinbase",
                    "sell_exchange": "hyperliquid",
                    "profit_pct": 0.42,
                    "volume": 1800,
                    "timestamp": (datetime.utcnow() - timedelta(minutes=12)).isoformat()
                },
                {
                    "symbol": "SOL",
                    "buy_exchange": "binance",
                    "sell_exchange": "coinbase",
                    "profit_pct": 0.78,
                    "volume": 3200,
                    "timestamp": (datetime.utcnow() - timedelta(minutes=18)).isoformat()
                }
            ]
        )
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get arbitrage metrics: {str(e)}")


@router.get("/performance-analytics", response_model=PerformanceAnalytics)
async def get_performance_analytics():
    """Get comprehensive performance analytics across all strategies"""
    try:
        # Mock performance analytics
        analytics = PerformanceAnalytics(
            strategy_comparison={
                "mean_reversion": {
                    "total_return": 8.5,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -3.2,
                    "win_rate": 0.68,
                    "avg_trade_duration": 4.5
                },
                "bollinger_bands": {
                    "total_return": 12.3,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": -2.1,
                    "win_rate": 0.72,
                    "avg_trade_duration": 3.2
                },
                "lstm_ml": {
                    "total_return": 15.7,
                    "sharpe_ratio": 2.1,
                    "max_drawdown": -4.5,
                    "win_rate": 0.73,
                    "avg_trade_duration": 6.8
                },
                "dqn_rl": {
                    "total_return": 11.2,
                    "sharpe_ratio": 1.6,
                    "max_drawdown": -3.8,
                    "win_rate": 0.68,
                    "avg_trade_duration": 5.1
                },
                "arbitrage": {
                    "total_return": 9.8,
                    "sharpe_ratio": 3.2,
                    "max_drawdown": -0.8,
                    "win_rate": 0.91,
                    "avg_trade_duration": 0.1
                }
            },
            risk_adjusted_returns={
                "arbitrage": 3.2,
                "lstm_ml": 2.1,
                "bollinger_bands": 1.8,
                "dqn_rl": 1.6,
                "mean_reversion": 1.2
            },
            correlation_matrix={
                "mean_reversion": {"bollinger_bands": 0.65, "lstm_ml": 0.42, "dqn_rl": 0.38, "arbitrage": 0.12},
                "bollinger_bands": {"lstm_ml": 0.58, "dqn_rl": 0.45, "arbitrage": 0.08},
                "lstm_ml": {"dqn_rl": 0.72, "arbitrage": 0.15},
                "dqn_rl": {"arbitrage": 0.18},
                "arbitrage": {}
            },
            drawdown_analysis={
                "max_portfolio_drawdown": -5.2,
                "avg_drawdown_duration": 2.3,
                "recovery_time": 4.1,
                "drawdown_frequency": 0.15,
                "worst_month": "2024-01"
            },
            volatility_metrics={
                "portfolio_volatility": 12.5,
                "strategy_volatility_avg": 15.8,
                "volatility_reduction": 20.9,
                "var_95": -2.8,
                "cvar_95": -4.2
            }
        )
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance analytics: {str(e)}")


@router.get("/strategy-heatmap")
async def get_strategy_heatmap():
    """Get strategy performance heatmap data"""
    try:
        # Generate mock heatmap data
        strategies = ["mean_reversion", "bollinger_bands", "lstm_ml", "dqn_rl", "arbitrage"]
        symbols = ["BTC", "ETH", "SOL", "WIF", "POPCAT"]
        
        heatmap_data = []
        for i, strategy in enumerate(strategies):
            for j, symbol in enumerate(symbols):
                # Generate realistic performance data
                base_performance = np.random.normal(5, 3)  # 5% average return, 3% std
                
                # Add strategy-specific bias
                if strategy == "arbitrage":
                    base_performance = abs(base_performance) * 0.5  # Lower but consistent
                elif strategy == "lstm_ml":
                    base_performance *= 1.2  # Slightly better
                
                heatmap_data.append({
                    "strategy": strategy,
                    "symbol": symbol,
                    "performance": round(base_performance, 2),
                    "trades": np.random.randint(10, 50),
                    "win_rate": round(np.random.uniform(0.55, 0.85), 2)
                })
        
        return {"heatmap_data": heatmap_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategy heatmap: {str(e)}")


@router.get("/risk-analysis")
async def get_risk_analysis():
    """Get comprehensive risk analysis"""
    try:
        risk_analysis = {
            "portfolio_risk": {
                "total_exposure": 85000,
                "max_exposure_limit": 100000,
                "exposure_utilization": 0.85,
                "concentration_risk": 0.32,
                "leverage_ratio": 2.1
            },
            "strategy_risk": {
                "high_risk_strategies": ["lstm_ml", "dqn_rl"],
                "medium_risk_strategies": ["bollinger_bands", "mean_reversion"],
                "low_risk_strategies": ["arbitrage"],
                "risk_diversification": 0.78
            },
            "market_risk": {
                "beta": 1.15,
                "correlation_to_btc": 0.82,
                "sector_concentration": {
                    "crypto": 0.95,
                    "defi": 0.05
                },
                "volatility_regime": "medium"
            },
            "operational_risk": {
                "system_uptime": 0.998,
                "execution_latency": 45,  # milliseconds
                "slippage_avg": 0.08,  # percentage
                "failed_trades": 0.02
            },
            "alerts": [
                {
                    "type": "warning",
                    "message": "High correlation between ML strategies detected",
                    "severity": "medium",
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "type": "info",
                    "message": "Arbitrage opportunities increasing",
                    "severity": "low",
                    "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat()
                }
            ]
        }
        
        return risk_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get risk analysis: {str(e)}")


@router.get("/model-diagnostics")
async def get_model_diagnostics():
    """Get ML model diagnostic information"""
    try:
        diagnostics = {
            "lstm_model": {
                "training_loss": 0.0045,
                "validation_loss": 0.0052,
                "epochs_trained": 87,
                "early_stopping": True,
                "overfitting_score": 0.15,
                "feature_drift": 0.08,
                "prediction_stability": 0.92,
                "last_retrain": (datetime.utcnow() - timedelta(hours=6)).isoformat()
            },
            "dqn_model": {
                "avg_reward": 0.23,
                "exploration_rate": 0.15,
                "episodes_trained": 1250,
                "convergence_status": "stable",
                "action_distribution": {
                    "buy": 0.35,
                    "sell": 0.32,
                    "hold": 0.33
                },
                "q_value_stability": 0.88,
                "last_retrain": (datetime.utcnow() - timedelta(hours=12)).isoformat()
            },
            "data_quality": {
                "missing_data_pct": 0.02,
                "outlier_detection": 0.05,
                "data_freshness": 0.98,
                "feature_correlation_health": 0.85
            },
            "performance_drift": {
                "lstm_drift": 0.12,
                "dqn_drift": 0.18,
                "threshold": 0.25,
                "retrain_recommended": False
            }
        }
        
        return diagnostics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model diagnostics: {str(e)}")


@router.get("/real-time-signals")
async def get_real_time_signals():
    """Get real-time trading signals from all strategies"""
    try:
        signals = [
            {
                "strategy": "lstm_ml",
                "symbol": "BTC",
                "signal": "BUY",
                "confidence": 0.78,
                "price": 45250.50,
                "predicted_move": 2.3,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "strategy": "arbitrage",
                "symbol": "ETH",
                "signal": "ARBITRAGE",
                "confidence": 0.95,
                "buy_exchange": "kraken",
                "sell_exchange": "binance",
                "profit_pct": 0.65,
                "timestamp": (datetime.utcnow() - timedelta(seconds=30)).isoformat()
            },
            {
                "strategy": "bollinger_bands",
                "symbol": "SOL",
                "signal": "SELL",
                "confidence": 0.72,
                "price": 99.85,
                "band_compression": 0.018,
                "timestamp": (datetime.utcnow() - timedelta(minutes=2)).isoformat()
            }
        ]
        
        return {"signals": signals}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time signals: {str(e)}")


@router.get("/optimization-suggestions")
async def get_optimization_suggestions():
    """Get AI-powered optimization suggestions"""
    try:
        suggestions = [
            {
                "type": "parameter_optimization",
                "strategy": "lstm_ml",
                "suggestion": "Increase sequence length to 80 for better pattern recognition",
                "expected_improvement": "5-8% accuracy increase",
                "confidence": 0.82,
                "priority": "high"
            },
            {
                "type": "risk_management",
                "strategy": "portfolio",
                "suggestion": "Reduce correlation between ML strategies by adjusting position sizes",
                "expected_improvement": "15% risk reduction",
                "confidence": 0.75,
                "priority": "medium"
            },
            {
                "type": "execution_optimization",
                "strategy": "arbitrage",
                "suggestion": "Implement faster execution for opportunities > 0.8% profit",
                "expected_improvement": "12% more successful arbitrages",
                "confidence": 0.88,
                "priority": "high"
            },
            {
                "type": "feature_engineering",
                "strategy": "dqn_rl",
                "suggestion": "Add market sentiment features to state representation",
                "expected_improvement": "10-15% better decision making",
                "confidence": 0.70,
                "priority": "medium"
            }
        ]
        
        return {"suggestions": suggestions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization suggestions: {str(e)}") 