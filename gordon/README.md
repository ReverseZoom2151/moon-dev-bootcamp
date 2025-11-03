# 🤖 Gordon - The Complete Financial Research & Trading Agent

> **"Combining Warren Buffett's fundamental analysis with Renaissance Technologies' quantitative trading"**

Gordon is an advanced AI-powered financial assistant that seamlessly integrates fundamental research capabilities with sophisticated algorithmic trading strategies. Built on top of cutting-edge LLM technology and professional trading infrastructure.

## 🚀 Features

### 📊 Financial Research (Powered by Dexter)
- **Fundamental Analysis**: Income statements, balance sheets, cash flows
- **SEC Filings**: 10-K, 10-Q, 8-K analysis
- **Market Intelligence**: Real-time news, analyst estimates
- **Peer Comparison**: Industry and competitor analysis

### 📈 Advanced Trading System
- **10+ Trading Strategies**: SMA, RSI, VWAP, Bollinger Bands, Mean Reversion
- **Algorithmic Orders**: TWAP, VWAP, Iceberg orders
- **Multi-Exchange Support**: Binance, Bitfinex, Hyperliquid
- **Real-time Market Data**: Live prices, order books, trade streams

### 🛡️ Risk Management
- **Position Sizing**: Kelly Criterion-based sizing
- **Risk Limits**: Automated drawdown and loss limits
- **Portfolio Analytics**: VaR, Sharpe ratio, correlation analysis
- **Safety Features**: Dry-run mode, emergency stops

### 🔬 Backtesting & Optimization
- **Historical Testing**: Test strategies on years of data
- **Parameter Optimization**: Find optimal strategy parameters
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios
- **Walk-Forward Analysis**: Out-of-sample validation

### 💡 Hybrid Analysis (Gordon's Secret Sauce!)
- **Combined Insights**: Merges fundamental and technical analysis
- **Smart Recommendations**: AI-powered trade suggestions
- **Confidence Scoring**: Weighted analysis with confidence levels
- **Action Items**: Clear, actionable trading steps

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for AI capabilities)
- Exchange API keys (for trading)

### Quick Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd gordon
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY=your-key-here
# - BINANCE_API_KEY=your-key-here (optional)
# - BINANCE_API_SECRET=your-secret-here (optional)
```

4. **Run Gordon**:
```bash
python cli.py
```

## 🎮 Usage

### Interactive Mode
```bash
python cli.py
```

### Single Query
```bash
python cli.py "Analyze Apple's financial health"
```

### Example Commands

#### Financial Research
```
Gordon> Analyze Tesla's revenue growth
Gordon> Compare Microsoft and Google profit margins
Gordon> Show me Amazon's latest 10-K filing
```

#### Trading Strategies
```
Gordon> Run RSI strategy on BTC/USDT
Gordon> Execute SMA crossover on ETH/USDT
Gordon> Backtest mean reversion on SOL from 2024-01-01 to 2024-12-31
```

#### Hybrid Analysis (Gordon's Specialty!)
```
Gordon> hybrid AAPL
Gordon> analyze TSLA with trading signals
Gordon> Should I buy NVDA?
```

#### Risk Management
```
Gordon> Calculate position size for 2% risk
Gordon> Show my portfolio risk metrics
Gordon> Check if buying 1 BTC violates risk limits
```

## 🔧 Configuration

Gordon uses a flexible configuration system via `config.yaml`:

```yaml
trading:
  risk:
    max_position_size: 0.1  # 10% max per position
    max_drawdown: 0.2       # 20% max drawdown
    daily_loss_limit: 0.05  # 5% daily loss limit

strategies:
  sma:
    short_period: 10
    long_period: 30

safety:
  dry_run: true  # Set to false for live trading
```

Environment variables override config file settings:
- `GORDON_BASE_POSITION_SIZE`
- `GORDON_MAX_DRAWDOWN`
- `GORDON_DRY_RUN`

## 📊 Architecture

```
gordon/
├── agent.py              # Core LLM agent
├── hybrid_analyzer.py    # Fundamental + Technical fusion
├── config_manager.py     # Configuration management
├── tools/
│   ├── finance/         # Financial research tools
│   └── trading/         # Trading execution tools
├── exchange_orchestrator/  # Exchange connections
└── backtesting/         # Backtesting engine
```

## 🛡️ Safety & Best Practices

1. **Always start in dry-run mode** - Test strategies without real money
2. **Set appropriate risk limits** - Use position sizing and stop losses
3. **Backtest before live trading** - Validate strategies on historical data
4. **Monitor actively** - Gordon assists but doesn't replace human judgment
5. **Keep API keys secure** - Never commit credentials to version control

## 🎯 Gordon's Trading Philosophy

Gordon combines two powerful approaches:

1. **Fundamental Analysis** (The Buffett Way)
   - Company financials and health
   - Long-term value assessment
   - Quality metrics and moats

2. **Technical/Quantitative Trading** (The Simons Way)
   - Price patterns and indicators
   - Statistical arbitrage
   - Algorithmic execution

By merging these approaches, Gordon provides a unique perspective that considers both the intrinsic value of assets and their market dynamics.

## 🤝 Contributing

We welcome contributions! Areas of interest:
- New trading strategies
- Additional exchange integrations
- Enhanced risk models
- ML/AI improvements

## ⚠️ Disclaimer

**IMPORTANT**: Trading cryptocurrencies and stocks involves substantial risk of loss. Gordon is an educational and research tool. Always:
- Do your own research
- Never invest more than you can afford to lose
- Understand the risks before trading
- Consider consulting financial advisors

## 📜 License

[Your License Here]

## 🙏 Acknowledgments

- Built on the Dexter research framework
- Powered by OpenAI's GPT models
- Exchange integrations via CCXT
- Technical analysis with TA-Lib

---

**Gordon**: *"Where fundamental analysis meets algorithmic trading"* 🚀📈