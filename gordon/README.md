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

1. **Install Gordon** (recommended):

```bash
pip install -e .
# Or install with optional dependencies:
pip install -e ".[dev,ml]"  # Includes dev tools and ML libraries
```

**Alternative**: Install dependencies only:

```bash
pip install -r requirements.txt
```

1. **Configure environment**:

```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY=your-key-here
# - BINANCE_API_KEY=your-key-here (optional)
# - BINANCE_API_SECRET=your-secret-here (optional)
```

1. **Run Gordon**:

**Option A: After installing the package** (recommended):

```bash
# Install first:
pip install -e .
# Then run:
gordon
# Or: python -m gordon.entrypoints.cli
```

**Option B: Without installing** (quick start):

```bash
# From the project root (parent of gordon directory):
# Windows:
set PYTHONPATH=%CD% && python -m gordon.entrypoints.cli

# Linux/Mac:
export PYTHONPATH=$(pwd) && python -m gordon.entrypoints.cli

# Or use the launcher scripts:
# Windows:
.\gordon\run_gordon.bat --help
# Linux/Mac:
bash gordon/run_gordon.sh --help
```

**Option C: Direct execution** (from gordon/entrypoints directory):

```bash
cd gordon/entrypoints
python cli.py --help
```

## 🎮 Usage

### Interactive Mode

```bash
gordon
# Or: python -m gordon.entrypoints.cli
```

### Single Query

```bash
gordon "Analyze Apple's financial health"
# Or: python -m gordon.entrypoints.cli "Analyze Apple's financial health"
```

### Example Commands

#### Financial Research

```text
Gordon> Analyze Tesla's revenue growth
Gordon> Compare Microsoft and Google profit margins
Gordon> Show me Amazon's latest 10-K filing
```

#### Trading Strategies

```text
Gordon> Run RSI strategy on BTC/USDT
Gordon> Execute SMA crossover on ETH/USDT
Gordon> Backtest mean reversion on SOL from 2024-01-01 to 2024-12-31
```

#### Hybrid Analysis (Gordon's Specialty!)

```text
Gordon> hybrid AAPL
Gordon> analyze TSLA with trading signals
Gordon> Should I buy NVDA?
```

#### Risk Management

```text
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

```text
gordon/
├── agent/              # AI agent (research + trading)
│   ├── agent.py       # Core LLM agent
│   ├── hybrid_analyzer.py  # Fundamental + Technical fusion
│   ├── config_manager.py  # Agent configuration (YAML)
│   └── conversational_assistant.py  # Chat interface (Day 30)
├── core/              # Core trading components
│   ├── orchestrator.py     # Exchange orchestrator
│   ├── strategies/         # Trading strategies (40+)
│   ├── risk/              # Risk management modules
│   ├── algo_orders/        # Algorithmic order types
│   └── streams/           # Market data streaming
├── exchanges/         # Exchange adapters (Binance, Bitfinex, Hyperliquid)
├── backtesting/      # Backtesting engine
│   ├── runners/      # Backtest runners
│   ├── strategies/  # Strategy implementations
│   ├── data/        # Data providers (Binance, Bitfinex, Yahoo)
│   └── evolution/   # GP strategy evolution (Day 29)
├── ml/               # ML indicator evaluation (Day 33)
├── research/         # Research components
│   └── social/      # Twitter sentiment (Day 28, 36)
├── tools/            # Tool integrations
│   ├── finance/     # Financial research tools
│   └── trading/    # Trading execution tools
├── utilities/        # Shared utilities
├── entrypoints/      # Entry points
│   └── cli.py       # Unified CLI (agent + orchestrator modes)
└── config.yaml       # Agent configuration
```

### Configuration System

Gordon uses **two separate configuration systems**:

1. **Agent Config** (`gordon/config.yaml`)
   - Used by: Agent, Hybrid Analyzer, Conversational Assistant
   - Manager: `gordon.agent.config_manager.ConfigManager`
   - Features: Agent settings, trading risk, ML settings, conversation memory

2. **Orchestrator Config** (`gordon/config/orchestrator_config.json`)
   - Used by: Exchange Orchestrator, Strategy Manager
   - Manager: `gordon.config.config_manager.ConfigManager`
   - Features: Exchange credentials, orchestrator settings, risk limits

See `CONFIG_README.md` for detailed documentation.

## 📦 Installation Options

### Option 1: Install as Package (Recommended)

```bash
pip install -e .
# This installs Gordon and creates the 'gordon' command
```

### Option 2: Install Dependencies Only

```bash
pip install -r requirements.txt
# Then run with: python -m gordon.entrypoints.cli
```

### Option 3: Install with Optional Dependencies

```bash
pip install -e ".[dev,ml]"  # Development tools + ML libraries
pip install -e ".[database]"  # Database support
```

**After installation**, you can run Gordon with:

```bash
gordon  # Uses the installed command
# Or: python -m gordon.entrypoints.cli
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
