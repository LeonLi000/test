"""Configuration for running the ETH/USDT short-term strategy with Jesse.

The configuration mirrors the defaults that work well for research and
backtesting.  Values can be overridden from environment variables in production
setups; however, the constants defined here are adequate for reproducing the
examples described in :mod:`backtest` and the project README.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# Directory that will contain cached data or exported results.  The scripts in
# this repository create the folder on-demand, so no additional setup is
# required.
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"

# Map of exchanges and symbols we intend to trade.  Jesse expects this structure
# when running either live or backtest sessions.
EXCHANGES: Dict[str, Dict[str, str]] = {
    "Binance": {"ETH-USDT": "15m"},
}

# Default strategy routes.  Jesse will instantiate a strategy instance for every
# dictionary defined in the list below.
STRATEGY_ROUTES: List[Dict[str, str]] = [
    {
        "exchange": "Binance",
        "symbol": "ETH-USDT",
        "timeframe": "15m",
        "strategy": "ETHShortTermStrategy",
    }
]

# In Jesse the `config` variable must be available at module import time.  The
# structure below is a trimmed-down version of the official configuration file
# focusing on the values that are relevant for research tasks.
config: Dict[str, object] = {
    "app": {
        "trading_mode": "backtest",
        "debug_mode": False,
        "cache_candles": True,
        "warmup_candles": 240,
        "start_date": "2021-01-01",
        "finish_date": "2021-12-31",
        "exchange": "Binance",
        "symbol": "ETH-USDT",
        "timeframe": "15m",
        "routes": STRATEGY_ROUTES,
    },
    "logging": {
        "log_mode": "file",
        "directory": str(DATA_DIRECTORY / "logs"),
    },
    "databases": {
        "postgres": {
            "driver": "postgres",
            "host": "localhost",
            "database": "jesse",
            "user": "jesse",
            "password": "jesse",
            "port": 5432,
        },
    },
    "metrics": {
        "dashboard": False,
        "tradingview": False,
    },
}

__all__ = [
    "config",
    "EXCHANGES",
    "STRATEGY_ROUTES",
    "DATA_DIRECTORY",
]
