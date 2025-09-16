"""ETHUSDT 15-minute short-term trading strategy for the Jesse framework.

This module implements the strategy described in the project README.  The
implementation favours readability and explicitness so that it can be used as a
reference when adapting the logic for live trading.  Whenever the optional
``jesse`` dependency is available the :class:`ETHShortTermStrategy` class will
inherit from :class:`jesse.strategies.Strategy`; otherwise a very small fallback
base class is used so that the file remains importable for documentation and
static-analysis purposes.

The strategy combines several indicators to locate short-term momentum swings
while aggressively managing risk through stop losses, take profits, and
indicator-based filters.  The goal of the example is to show how the trading
rules from the README translate into Jesse code rather than to provide a
fully-optimised production system.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Optional Jesse imports
# ---------------------------------------------------------------------------
StrategyBase: type
utils_module = None
indicators_module = None

_strategy_spec = importlib.util.find_spec("jesse.strategies")
if _strategy_spec is not None:
    StrategyBase = importlib.import_module("jesse.strategies").Strategy
else:  # pragma: no cover - executed only when Jesse is absent.
    class _FallbackPosition:
        """Very small stand-in for :class:`jesse.models.Position`."""

        def __init__(self) -> None:
            self.is_long: bool = False
            self.is_short: bool = False
            self.qty: float = 0.0
            self.entry_price: float = 0.0

        @property
        def is_open(self) -> bool:
            return self.qty != 0.0

        def reset(self) -> None:
            self.is_long = False
            self.is_short = False
            self.qty = 0.0
            self.entry_price = 0.0

    class StrategyBase:  # type: ignore[override]
        """Fallback base class used when the Jesse framework is unavailable."""

        def __init__(self, *_, **__) -> None:
            self.vars: Dict[str, float] = {}
            self.candles: np.ndarray = np.empty((0, 6), dtype=float)
            self.price: float = np.nan
            self.capital: float = 100000.0
            self.buy = None
            self.sell = None
            self.stop_loss = None
            self.take_profit = None
            self.position = _FallbackPosition()

        @property
        def exchange(self) -> str:
            return "Binance"

        @property
        def symbol(self) -> str:
            return "ETH-USDT"

        @property
        def timeframe(self) -> str:
            return "15m"

_utils_spec = importlib.util.find_spec("jesse.utils")
if _utils_spec is not None:
    utils_module = importlib.import_module("jesse.utils")

_indicators_spec = importlib.util.find_spec("jesse.indicators")
if _indicators_spec is not None:
    indicators_module = importlib.import_module("jesse.indicators")


# ---------------------------------------------------------------------------
# Helper indicator implementations
# ---------------------------------------------------------------------------
def _sma(values: np.ndarray, period: int) -> float:
    if values.size < period:
        return float("nan")
    return float(np.mean(values[-period:]))


def _ema(values: np.ndarray, period: int) -> float:
    if values.size < period:
        return float("nan")
    if indicators_module is not None:
        return float(indicators_module.ema(values, period=period))
    seed = float(np.mean(values[:period]))
    multiplier = 2.0 / (period + 1.0)
    ema_val = seed
    for price in values[period:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return float(ema_val)


def _rsi(values: np.ndarray, period: int) -> float:
    if values.size <= period:
        return float("nan")
    if indicators_module is not None:
        return float(indicators_module.rsi(values, period=period))
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _macd(values: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
    if values.size < slow + signal:
        return np.array([float("nan"), float("nan"), float("nan")])
    if indicators_module is not None:
        macd_line = indicators_module.macd(values, fast=fast, slow=slow)
        signal_line = indicators_module.ema(macd_line, period=signal)
        histogram = macd_line - signal_line
        return np.array([float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])])

    macd_fast = np.array([_ema(values[: i + 1], fast) for i in range(values.size)])
    macd_slow = np.array([_ema(values[: i + 1], slow) for i in range(values.size)])
    macd_line = macd_fast - macd_slow
    signal_line = np.array([_ema(macd_line[: i + 1], signal) for i in range(macd_line.size)])
    histogram = macd_line - signal_line
    return np.array([float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])])


def _bollinger(values: np.ndarray, period: int, stddev: float) -> np.ndarray:
    if values.size < period:
        return np.array([float("nan"), float("nan"), float("nan")])
    if indicators_module is not None:
        upper = indicators_module.bollinger_bands(values, period=period, dev=stddev, direction="upper")
        middle = indicators_module.sma(values, period=period)
        lower = indicators_module.bollinger_bands(values, period=period, dev=stddev, direction="lower")
        return np.array([float(upper[-1]), float(middle[-1]), float(lower[-1])])
    recent = values[-period:]
    mid = float(np.mean(recent))
    std = float(np.std(recent))
    upper = mid + stddev * std
    lower = mid - stddev * std
    return np.array([upper, mid, lower])


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    if close.size <= period:
        return float("nan")
    if indicators_module is not None:
        return float(indicators_module.atr(high, low, close, period=period))
    trs = []
    for i in range(1, close.size):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return float("nan")
    return float(np.mean(trs[-period:]))


@dataclass
class IndicatorSnapshot:
    """Container for the indicator values used in the strategy."""

    close: float
    volume: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    rsi: float
    macd_line: float
    macd_signal: float
    bollinger_upper: float
    bollinger_mid: float
    bollinger_lower: float
    atr: float
    volume_sma: float

    @property
    def is_valid(self) -> bool:
        return not any(np.isnan([
            self.ema_fast,
            self.ema_slow,
            self.ema_trend,
            self.rsi,
            self.macd_line,
            self.macd_signal,
            self.bollinger_upper,
            self.bollinger_mid,
            self.bollinger_lower,
            self.atr,
            self.volume_sma,
        ]))


class ETHShortTermStrategy(StrategyBase):
    """Composite indicator strategy for the ETH/USDT pair on a 15m timeframe."""

    risk_per_trade: float = 0.02
    fixed_stop: float = 0.015
    fixed_take_profit: float = 0.03
    atr_multiplier: float = 1.5
    cooldown_period: int = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vars.setdefault("last_signal_index", None)
        self.vars.setdefault("last_signal_direction", None)
        self.vars.setdefault("initial_stop", None)
        self.vars.setdefault("cooldown_active", False)

    # ------------------------------------------------------------------
    # Indicator access helpers
    # ------------------------------------------------------------------
    def _close_prices(self) -> np.ndarray:
        if self.candles.size == 0:
            return np.array([], dtype=float)
        return self.candles[:, 2].astype(float)

    def _high_prices(self) -> np.ndarray:
        if self.candles.size == 0:
            return np.array([], dtype=float)
        return self.candles[:, 3].astype(float)

    def _low_prices(self) -> np.ndarray:
        if self.candles.size == 0:
            return np.array([], dtype=float)
        return self.candles[:, 4].astype(float)

    def _volumes(self) -> np.ndarray:
        if self.candles.size == 0:
            return np.array([], dtype=float)
        return self.candles[:, 5].astype(float)

    def _indicator_snapshot(self) -> IndicatorSnapshot:
        closes = self._close_prices()
        highs = self._high_prices()
        lows = self._low_prices()
        volumes = self._volumes()

        ema_fast = _ema(closes, period=8)
        ema_slow = _ema(closes, period=21)
        ema_trend = _ema(closes, period=55)
        rsi = _rsi(closes, period=14)
        macd_line, macd_signal, _ = _macd(closes, fast=12, slow=26, signal=9)
        bollinger_upper, bollinger_mid, bollinger_lower = _bollinger(closes, period=20, stddev=2)
        atr = _atr(highs, lows, closes, period=14)
        volume_sma = _sma(volumes, period=20)

        close = float(closes[-1]) if closes.size else float("nan")
        volume = float(volumes[-1]) if volumes.size else float("nan")
        return IndicatorSnapshot(
            close=close,
            volume=volume,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_trend=ema_trend,
            rsi=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            bollinger_upper=bollinger_upper,
            bollinger_mid=bollinger_mid,
            bollinger_lower=bollinger_lower,
            atr=atr,
            volume_sma=volume_sma,
        )

    def _enough_data(self) -> bool:
        return self.candles.shape[0] >= 55  # longest lookback requirement

    def _current_index(self) -> int:
        return int(self.candles.shape[0] - 1)

    def _cooldown_allows_entry(self, direction: str) -> bool:
        last_index = self.vars.get("last_signal_index")
        last_direction = self.vars.get("last_signal_direction")
        if last_index is None:
            return True
        if self._current_index() - last_index < self.cooldown_period:
            return False
        if last_direction == direction:
            return True
        return True

    def _volume_expansion(self, snapshot: IndicatorSnapshot) -> bool:
        if np.isnan(snapshot.volume) or np.isnan(snapshot.volume_sma):
            return False
        return snapshot.volume >= snapshot.volume_sma * 1.2

    # ------------------------------------------------------------------
    # Jesse strategy interface
    # ------------------------------------------------------------------
    def should_long(self) -> bool:
        if not self._enough_data() or self.position.is_open:
            return False
        if not self._cooldown_allows_entry("long"):
            return False

        snapshot = self._indicator_snapshot()
        if not snapshot.is_valid:
            return False
        conditions = [
            snapshot.ema_fast > snapshot.ema_slow,
            snapshot.close > snapshot.ema_trend,
            30.0 <= snapshot.rsi <= 70.0,
            snapshot.macd_line > snapshot.macd_signal,
            snapshot.bollinger_lower < snapshot.close < snapshot.bollinger_mid,
            self._volume_expansion(snapshot),
        ]
        return all(conditions)

    def should_short(self) -> bool:
        if not self._enough_data() or self.position.is_open:
            return False
        if not self._cooldown_allows_entry("short"):
            return False

        snapshot = self._indicator_snapshot()
        if not snapshot.is_valid:
            return False
        conditions = [
            snapshot.ema_fast < snapshot.ema_slow,
            snapshot.close < snapshot.ema_trend,
            30.0 <= snapshot.rsi <= 70.0,
            snapshot.macd_line < snapshot.macd_signal,
            snapshot.bollinger_mid < snapshot.close < snapshot.bollinger_upper,
            self._volume_expansion(snapshot),
        ]
        return all(conditions)

    def should_cancel(self) -> bool:  # pragma: no cover - behaviour delegated to Jesse
        # Cancel resting orders if the entry conditions disappear.
        if self.position.is_open:
            return False
        snapshot = self._indicator_snapshot()
        if not snapshot.is_valid:
            return True
        long_ready = self.should_long()
        short_ready = self.should_short()
        return not (long_ready or short_ready)

    # ------------------------------------------------------------------
    # Order placement and management
    # ------------------------------------------------------------------
    def _position_size(self, price: float) -> float:
        capital = float(getattr(self, "capital", 0.0))
        if capital <= 0.0:
            capital = 100000.0
        allocation = capital * self.risk_per_trade
        if utils_module is not None and hasattr(utils_module, "size_to_qty"):
            return float(utils_module.size_to_qty(allocation, price))
        if price == 0:
            return 0.0
        return allocation / price

    def _stop_loss_offset(self, price: float, snapshot: IndicatorSnapshot) -> float:
        atr_component = 0.0
        if not np.isnan(snapshot.atr) and price != 0.0:
            atr_component = (snapshot.atr / price) * self.atr_multiplier
        if atr_component == 0.0:
            return self.fixed_stop
        return min(self.fixed_stop, atr_component)

    def go_long(self) -> None:
        snapshot = self._indicator_snapshot()
        entry_price = snapshot.close if not np.isnan(snapshot.close) else float(self.price)
        qty = self._position_size(entry_price)
        stop_offset = self._stop_loss_offset(entry_price, snapshot)
        stop_price = entry_price * (1.0 - stop_offset)
        take_profit_price = entry_price * (1.0 + self.fixed_take_profit)

        self.buy = qty, entry_price
        self.stop_loss = qty, stop_price
        self.take_profit = qty, take_profit_price
        self.vars["initial_stop"] = stop_price
        self.vars["last_signal_index"] = self._current_index()
        self.vars["last_signal_direction"] = "long"

    def go_short(self) -> None:
        snapshot = self._indicator_snapshot()
        entry_price = snapshot.close if not np.isnan(snapshot.close) else float(self.price)
        qty = self._position_size(entry_price)
        stop_offset = self._stop_loss_offset(entry_price, snapshot)
        stop_price = entry_price * (1.0 + stop_offset)
        take_profit_price = entry_price * (1.0 - self.fixed_take_profit)

        self.sell = qty, entry_price
        self.stop_loss = qty, stop_price
        self.take_profit = qty, take_profit_price
        self.vars["initial_stop"] = stop_price
        self.vars["last_signal_index"] = self._current_index()
        self.vars["last_signal_direction"] = "short"

    def update_position(self) -> None:  # pragma: no cover - requires live candle updates
        if not self.position.is_open:
            return
        snapshot = self._indicator_snapshot()
        if not snapshot.is_valid or np.isnan(snapshot.atr):
            return
        if self.position.is_long:
            trailing_stop = snapshot.close - snapshot.atr * self.atr_multiplier
            if trailing_stop > float(self.vars.get("initial_stop", 0.0)):
                self.stop_loss = self.position.qty, trailing_stop
                self.vars["initial_stop"] = trailing_stop
        elif self.position.is_short:
            trailing_stop = snapshot.close + snapshot.atr * self.atr_multiplier
            if trailing_stop < float(self.vars.get("initial_stop", float("inf"))):
                self.stop_loss = self.position.qty, trailing_stop
                self.vars["initial_stop"] = trailing_stop

    def on_stop_loss(self) -> None:  # pragma: no cover - dependent on Jesse runtime
        if self.position.is_open:
            return
        self.vars["cooldown_active"] = True

    def on_take_profit(self) -> None:  # pragma: no cover - dependent on Jesse runtime
        if self.position.is_open:
            return
        self.vars["cooldown_active"] = True
