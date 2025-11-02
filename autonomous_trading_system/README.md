# 🚀 Autonomous Trading System

A comprehensive full-stack autonomous trading system built from proven algorithmic trading strategies and components.

## 🎯 Overview

This system consolidates multiple trading algorithms, data sources, and execution engines into a unified platform capable of:

- **Multi-Strategy Trading**: Mean reversion, momentum, technical indicators, supply/demand zones
- **Multi-Exchange Support**: HyperLiquid, Solana DEX, Interactive Brokers, Binance
- **Real-time Analysis**: Market data, whale positions, liquidations, social sentiment
- **Advanced Backtesting**: Strategy optimization, walk-forward analysis, genetic algorithms
- **Risk Management**: Position sizing, stop losses, portfolio management
- **Web Dashboard**: Real-time monitoring, strategy management, performance analytics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND DASHBOARD                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Strategy  │ │  Portfolio  │ │    Market Analysis      │ │
│  │ Management  │ │ Monitoring  │ │     & Signals          │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     API GATEWAY                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Trading   │ │    Data     │ │      Strategy           │ │
│  │  Endpoints  │ │  Endpoints  │ │     Endpoints           │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  STRATEGY ENGINE                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Mean Rev.   │ │ Bollinger   │ │    Supply/Demand        │ │
│  │ Strategies  │ │ Strategies  │ │      Strategies         │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │    VWAP     │ │   StochRSI  │ │    Liquidation          │ │
│  │ Strategies  │ │ Strategies  │ │     Strategies          │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA LAYER                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ HyperLiquid │ │   Solana    │ │    Market Data          │ │
│  │    Data     │ │    Data     │ │     Feeds               │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Whale     │ │ Historical  │ │     Social              │ │
│  │ Positions   │ │    Data     │ │    Sentiment            │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                EXECUTION LAYER                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ HyperLiquid │ │   Solana    │ │    Risk Management      │ │
│  │  Execution  │ │  Execution  │ │     & Position          │ │
│  │             │ │             │ │      Sizing             │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
autonomous_trading_system/
├── backend/
│   ├── api/                    # FastAPI application
│   ├── strategies/             # Trading strategy implementations
│   ├── data/                   # Data management and feeds
│   ├── execution/              # Order execution engines
│   ├── risk/                   # Risk management
│   └── utils/                  # Utility functions
├── frontend/                   # React dashboard
├── database/                   # Database schemas and migrations
├── config/                     # Configuration management
├── tests/                      # Test suites
└── docs/                       # Documentation
```

## 🤖 Available Strategies

### Technical Analysis Strategies
- **Mean Reversion**: SMA-based with configurable thresholds
- **Bollinger Bands**: Band compression and breakout detection
- **VWAP**: Volume-weighted average price strategies
- **StochRSI**: Stochastic RSI momentum strategies
- **Moving Average Crossovers**: Multiple timeframe analysis

### Advanced Strategies
- **Supply/Demand Zones**: Price level analysis and zone trading
- **Liquidation-based**: Trading around liquidation events
- **Kalman Filter**: Advanced signal processing for trend detection
- **Genetic Programming**: Evolved strategies using GP algorithms

### Market Structure Strategies
- **Whale Position Tracking**: Following large position movements
- **Social Sentiment**: Twitter and social media analysis
- **Token Screening**: New token launch analysis (Solana)

## 🔌 Exchange Integrations

- **HyperLiquid**: Perpetual futures trading
- **Solana DEX**: Decentralized exchange trading
- **Interactive Brokers**: Traditional markets
- **Binance**: Spot and futures trading

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+
pip install -r requirements.txt

# Node.js 16+ (for frontend)
npm install
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

### Running the System
```bash
# Start backend
cd backend
python main.py

# Start frontend (new terminal)
cd frontend
npm start

# Access dashboard at http://localhost:3000
```

## 📊 Features

### Real-time Monitoring
- Live portfolio tracking
- Strategy performance metrics
- Market data visualization
- Risk metrics dashboard

### Strategy Management
- Enable/disable strategies
- Parameter optimization
- Backtesting interface
- Performance analytics

### Risk Management
- Position sizing algorithms
- Stop loss management
- Portfolio diversification
- Drawdown protection

## 🔧 Configuration

All strategies and system parameters are configurable through:
- Environment variables
- Configuration files
- Web dashboard interface
- API endpoints

## 📈 Backtesting

Comprehensive backtesting framework supporting:
- Historical data analysis
- Parameter optimization
- Walk-forward testing
- Monte Carlo simulation
- Risk-adjusted metrics

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and never risk more than you can afford to lose.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ by the ATC Bootcamp Community** 