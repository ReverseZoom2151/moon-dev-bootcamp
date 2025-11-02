import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  Button,
  Tab,
  Tabs,
  LinearProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Stop,
  Refresh,
  AccountBalance,
  ShowChart,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import axios from 'axios';
import { toast, Toaster } from 'react-hot-toast';
import { motion } from 'framer-motion';

// Types
interface StrategyStatus {
  name: string;
  enabled: boolean;
  status: string;
  last_signal?: any;
  performance: {
    total_trades: number;
    winning_trades: number;
    total_pnl: number;
    win_rate: number;
    sharpe_ratio: number;
  };
}

interface PortfolioData {
  total_value: number;
  cash_balance: number;
  positions: any[];
  daily_pnl: number;
  total_pnl: number;
  open_positions: number;
}

interface SystemStatus {
  status: string;
  services: {
    strategy_engine: boolean;
    portfolio_manager: boolean;
    risk_manager: boolean;
  };
  system: {
    uptime: string;
    active_strategies: number;
    total_strategies: number;
  };
  portfolio: PortfolioData;
  risk: {
    max_drawdown: number;
    current_exposure: number;
    risk_score: number;
  };
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// Dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff88',
    },
    secondary: {
      main: '#ff4444',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", "Courier New", monospace',
  },
});

function App() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [strategies, setStrategies] = useState<StrategyStatus[]>([]);
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Mock data for charts
  const portfolioHistory = [
    { time: '09:00', value: 100000, pnl: 0 },
    { time: '10:00', value: 101500, pnl: 1500 },
    { time: '11:00', value: 99800, pnl: -200 },
    { time: '12:00', value: 102300, pnl: 2300 },
    { time: '13:00', value: 103100, pnl: 3100 },
    { time: '14:00', value: 101900, pnl: 1900 },
    { time: '15:00', value: 104500, pnl: 4500 },
  ];

  const strategyPerformance = [
    { name: 'Mean Reversion', trades: 45, winRate: 68, pnl: 2300 },
    { name: 'Bollinger Bands', trades: 32, winRate: 72, pnl: 1800 },
    { name: 'VWAP', trades: 28, winRate: 65, pnl: 1200 },
    { name: 'Supply/Demand', trades: 22, winRate: 75, pnl: 1600 },
    { name: 'StochRSI', trades: 38, winRate: 63, pnl: 900 },
  ];

  const riskMetrics = [
    { name: 'Low Risk', value: 35, color: '#00ff88' },
    { name: 'Medium Risk', value: 45, color: '#ffaa00' },
    { name: 'High Risk', value: 20, color: '#ff4444' },
  ];

  // API calls
  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get('/status');
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      toast.error('Failed to fetch system status');
    }
  };

  const fetchStrategies = async () => {
    try {
      const response = await axios.get('/api/v1/strategies/');
      setStrategies(response.data.strategies || []);
    } catch (error) {
      console.error('Failed to fetch strategies:', error);
      toast.error('Failed to fetch strategies');
    }
  };

  const toggleStrategy = async (strategyName: string, enabled: boolean) => {
    try {
      const endpoint = enabled ? 'enable' : 'disable';
      await axios.post(`/api/v1/strategies/${strategyName}/${endpoint}`);
      toast.success(`Strategy ${strategyName} ${enabled ? 'enabled' : 'disabled'}`);
      fetchStrategies();
    } catch (error) {
      toast.error(`Failed to ${enabled ? 'enable' : 'disable'} strategy`);
    }
  };

  const emergencyStop = async () => {
    try {
      await axios.post('/emergency-stop');
      toast.success('Emergency stop executed');
      fetchSystemStatus();
    } catch (error) {
      toast.error('Failed to execute emergency stop');
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchSystemStatus(), fetchStrategies()]);
      setLoading(false);
    };

    loadData();

    // Auto-refresh every 5 seconds
    const interval = setInterval(() => {
      if (autoRefresh) {
        fetchSystemStatus();
        fetchStrategies();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
      case 'healthy':
        return 'success';
      case 'stopped':
      case 'error':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getStatusIcon = (status: boolean) => {
    return status ? <CheckCircle color="success" /> : <Error color="error" />;
  };

  if (loading) {
    return (
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Box sx={{ width: '100%', mt: 4 }}>
          <LinearProgress />
          <Typography variant="h6" align="center" sx={{ mt: 2 }}>
            Loading Autonomous Trading System...
          </Typography>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Toaster position="top-right" />
      
      <AppBar position="static" sx={{ background: 'linear-gradient(45deg, #1a1a1a 30%, #2a2a2a 90%)' }}>
        <Toolbar>
          <TrendingUp sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🚀 Autonomous Trading System
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                color="primary"
              />
            }
            label="Auto Refresh"
            sx={{ mr: 2 }}
          />
          
          <Tooltip title="Refresh Data">
            <IconButton onClick={() => { fetchSystemStatus(); fetchStrategies(); }}>
              <Refresh />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Emergency Stop">
            <IconButton onClick={emergencyStop} color="error">
              <Stop />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 2 }}>
        {/* System Status Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography color="textSecondary" gutterBottom>
                        System Status
                      </Typography>
                      <Typography variant="h6">
                        {systemStatus?.status || 'Unknown'}
                      </Typography>
                    </Box>
                    {getStatusIcon(systemStatus?.services.strategy_engine || false)}
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography color="textSecondary" gutterBottom>
                        Portfolio Value
                      </Typography>
                      <Typography variant="h6" color="primary">
                        ${systemStatus?.portfolio.total_value?.toLocaleString() || '0'}
                      </Typography>
                    </Box>
                    <AccountBalance color="primary" />
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography color="textSecondary" gutterBottom>
                        Daily P&L
                      </Typography>
                      <Typography 
                        variant="h6" 
                        color={(systemStatus?.portfolio.daily_pnl ?? 0) >= 0 ? 'success.main' : 'error.main'}
                      >
                        ${systemStatus?.portfolio.daily_pnl?.toLocaleString() || '0'}
                      </Typography>
                    </Box>
                    {(systemStatus?.portfolio.daily_pnl ?? 0) >= 0 ? 
                      <TrendingUp color="success" /> : 
                      <TrendingDown color="error" />
                    }
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography color="textSecondary" gutterBottom>
                        Active Strategies
                      </Typography>
                      <Typography variant="h6">
                        {systemStatus?.system.active_strategies || 0} / {systemStatus?.system.total_strategies || 0}
                      </Typography>
                    </Box>
                    <ShowChart color="primary" />
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>

        {/* Main Content Tabs */}
        <Card>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
              <Tab label="Dashboard" />
              <Tab label="Strategies" />
              <Tab label="Portfolio" />
              <Tab label="Analytics" />
              <Tab label="Risk Management" />
            </Tabs>
          </Box>

          {/* Dashboard Tab */}
          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Card>
                  <CardHeader title="Portfolio Performance" />
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={portfolioHistory}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <RechartsTooltip />
                        <Area 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#00ff88" 
                          fill="#00ff8820" 
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card>
                  <CardHeader title="Risk Distribution" />
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={riskMetrics}
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {riskMetrics.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <RechartsTooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Strategies Tab */}
          <TabPanel value={tabValue} index={1}>
            <Grid container spacing={3}>
              {strategies.map((strategy, index) => (
                <Grid item xs={12} md={6} lg={4} key={strategy.name}>
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Card>
                      <CardHeader
                        title={strategy.name}
                        action={
                          <Switch
                            checked={strategy.enabled}
                            onChange={(e) => toggleStrategy(strategy.name, e.target.checked)}
                            color="primary"
                          />
                        }
                        subheader={
                          <Chip
                            label={strategy.status}
                            color={getStatusColor(strategy.status)}
                            size="small"
                          />
                        }
                      />
                      <CardContent>
                        <Typography variant="body2" color="textSecondary">
                          Trades: {strategy.performance.total_trades}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Win Rate: {(strategy.performance.win_rate * 100).toFixed(1)}%
                        </Typography>
                        <Typography 
                          variant="body2" 
                          color={strategy.performance.total_pnl >= 0 ? 'success.main' : 'error.main'}
                        >
                          P&L: ${strategy.performance.total_pnl.toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  </motion.div>
                </Grid>
              ))}
            </Grid>
          </TabPanel>

          {/* Portfolio Tab */}
          <TabPanel value={tabValue} index={2}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardHeader title="Portfolio Overview" />
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} md={3}>
                        <Typography variant="h6" color="primary">
                          Total Value
                        </Typography>
                        <Typography variant="h4">
                          ${systemStatus?.portfolio.total_value?.toLocaleString() || '0'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Typography variant="h6" color="textSecondary">
                          Cash Balance
                        </Typography>
                        <Typography variant="h4">
                          ${systemStatus?.portfolio.cash_balance?.toLocaleString() || '0'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Typography variant="h6" color="textSecondary">
                          Open Positions
                        </Typography>
                        <Typography variant="h4">
                          {systemStatus?.portfolio.open_positions || 0}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Typography variant="h6" color="textSecondary">
                          Total P&L
                        </Typography>
                        <Typography 
                          variant="h4"
                          color={(systemStatus?.portfolio.total_pnl ?? 0) >= 0 ? 'success.main' : 'error.main'}
                        >
                          ${systemStatus?.portfolio.total_pnl?.toLocaleString() || '0'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Analytics Tab */}
          <TabPanel value={tabValue} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardHeader title="Strategy Performance Comparison" />
                  <CardContent>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={strategyPerformance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <RechartsTooltip />
                        <Bar dataKey="pnl" fill="#00ff88" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Risk Management Tab */}
          <TabPanel value={tabValue} index={4}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader title="Risk Metrics" />
                  <CardContent>
                    <Box mb={2}>
                      <Typography variant="body2" color="textSecondary">
                        Max Drawdown
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={(systemStatus?.risk.max_drawdown || 0) * 100} 
                        color="error"
                      />
                      <Typography variant="caption">
                        {((systemStatus?.risk.max_drawdown || 0) * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                    
                    <Box mb={2}>
                      <Typography variant="body2" color="textSecondary">
                        Current Exposure
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={(systemStatus?.risk.current_exposure || 0) * 100} 
                        color="warning"
                      />
                      <Typography variant="caption">
                        {((systemStatus?.risk.current_exposure || 0) * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                    
                    <Box>
                      <Typography variant="body2" color="textSecondary">
                        Risk Score
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={(systemStatus?.risk.risk_score || 0) * 100} 
                        color="primary"
                      />
                      <Typography variant="caption">
                        {((systemStatus?.risk.risk_score || 0) * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader title="Emergency Controls" />
                  <CardContent>
                    <Alert severity="warning" sx={{ mb: 2 }}>
                      Emergency controls will immediately stop all trading activities
                    </Alert>
                    <Button
                      variant="contained"
                      color="error"
                      fullWidth
                      startIcon={<Stop />}
                      onClick={emergencyStop}
                    >
                      Emergency Stop All Trading
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </Card>
      </Container>
    </ThemeProvider>
  );
}

export default App; 