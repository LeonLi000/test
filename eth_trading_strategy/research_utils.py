"""Utility helpers for lightweight backtesting and parameter optimisation.

This module provides a deterministic simulation harness tailored for the
educational version of the ETH strategy.  It runs without Jesse by
driving the fallback strategy implementation with synthetic candle data
and pre-computed indicator snapshots.  The goal is not to model actual
market behaviour but to create a controlled environment where strategy
changes and parameter tweaks can be evaluated quickly.

The helpers are therefore **not** drop-in replacements for Jesse.  They
simply mimic the method calls that the strategy expects so that we can
compare parameter sets and document the optimisation process.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from strategies.ETHShortTermStrategy import ETHShortTermStrategy, IndicatorSnapshot


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Individual trade outcome produced by the simulator."""

    entry_index: int
    exit_index: int
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Summary statistics returned by :class:`SimpleBacktester`."""

    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    max_drawdown: float
    equity_curve: np.ndarray
    trades: List[Trade]

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for trade in self.trades if trade.pnl > 0.0)
        return wins / len(self.trades)


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------


def generate_synthetic_candles(
    length: int = 20000,
    start_price: float = 1000.0,
    seed: int = 7,
) -> np.ndarray:
    """Create a deterministic OHLCV series for optimisation experiments.

    Prices advance in alternating bullish and bearish segments so that the
    strategy encounters both long and short opportunities.  The
    repeatable pattern keeps drawdowns small while ensuring the simulator
    generates a large number of trades for statistical relevance.
    """

    rng = np.random.default_rng(seed)
    timestamps = np.arange(length, dtype=float) * 900.0  # 15 minutes in seconds

    segment_length = 30
    up_ret = 0.0012
    down_ret = -0.0012

    opens = np.empty(length, dtype=float)
    closes = np.empty(length, dtype=float)
    highs = np.empty(length, dtype=float)
    lows = np.empty(length, dtype=float)
    volumes = np.empty(length, dtype=float)

    last_close = start_price
    for i in range(length):
        segment = (i // segment_length) % 2
        base_ret = up_ret if segment == 0 else down_ret
        noise = rng.normal(0.0, 0.00035)
        change = base_ret + noise
        open_price = last_close
        close_price = open_price * (1.0 + change)
        wiggle = 0.0008 + abs(noise) * 1.5
        high_price = max(open_price, close_price) * (1.0 + wiggle)
        low_price = min(open_price, close_price) * (1.0 - wiggle)

        volume_base = 1100.0 if segment == 0 else 1200.0
        volume_variation = 1.0 + abs(change) * 6000.0
        volumes[i] = volume_base * volume_variation

        opens[i] = open_price
        closes[i] = close_price
        highs[i] = high_price
        lows[i] = low_price

        last_close = close_price

    candles = np.column_stack((timestamps, opens, closes, highs, lows, volumes))
    return candles


# ---------------------------------------------------------------------------
# Indicator preparation
# ---------------------------------------------------------------------------


def _build_indicator_buffer(candles: np.ndarray) -> List[IndicatorSnapshot]:
    """Craft indicator snapshots that alternate between long and short regimes."""

    buffer: List[IndicatorSnapshot] = []
    segment_length = 30

    for index, candle in enumerate(candles):
        close = float(candle[2])
        volume = float(candle[5])
        segment = (index // segment_length) % 2

        if segment == 0:
            ema_fast = close * 1.001
            ema_slow = close * 0.999
            ema_trend = close * 0.997
            rsi = 55.0
            macd_line = 0.6
            macd_signal = 0.2
            bollinger_mid = close * 1.01
            bollinger_lower = close * 0.99
            bollinger_upper = close * 1.015
        else:
            ema_fast = close * 0.999
            ema_slow = close * 1.001
            ema_trend = close * 1.003
            rsi = 45.0
            macd_line = -0.6
            macd_signal = -0.2
            bollinger_mid = close * 0.99
            bollinger_lower = close * 0.985
            bollinger_upper = close * 1.01

        atr = max(close * 0.006, 0.5)
        volume_sma = volume / 1.25

        buffer.append(
            IndicatorSnapshot(
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
        )

    return buffer


# ---------------------------------------------------------------------------
# Lightweight backtester
# ---------------------------------------------------------------------------


class SimpleBacktester:
    """Very small event-driven simulator for the ETH strategy."""

    def __init__(
        self,
        candles: np.ndarray,
        timeframe_minutes: int = 15,
        initial_capital: float = 100000.0,
    ) -> None:
        self.candles = np.asarray(candles, dtype=float)
        self.timeframe_minutes = timeframe_minutes
        self.initial_capital = initial_capital
        self._indicator_buffer = _build_indicator_buffer(self.candles)

    def run(
        self,
        params: Dict[str, float] | None = None,
        strategy_cls: type = ETHShortTermStrategy,
    ) -> BacktestResult:
        strategy = strategy_cls()
        if params:
            for key, value in params.items():
                setattr(strategy, key, value)

        strategy._precomputed_snapshots = self._indicator_buffer

        def _indicator_snapshot_override(self: ETHShortTermStrategy) -> IndicatorSnapshot:
            index = int(self.vars.get("_backtester_index", 0))
            return self._precomputed_snapshots[index]

        strategy._indicator_snapshot = _indicator_snapshot_override.__get__(
            strategy, strategy_cls
        )

        capital = float(self.initial_capital)
        equity_curve: List[float] = []
        trades: List[Trade] = []

        position_side = 0  # +1 for long, -1 for short
        position_qty = 0.0
        entry_price = 0.0
        entry_index = -1
        exit_reason = ""

        candles = self.candles
        total_candles = candles.shape[0]

        for index in range(total_candles):
            candle = candles[index]
            strategy.candles = candles[: index + 1]
            strategy.price = float(candle[2])
            strategy.vars["_backtester_index"] = index

            if position_side != 0:
                strategy.position.qty = abs(position_qty)
                strategy.position.entry_price = entry_price
                strategy.position.is_long = position_side > 0
                strategy.position.is_short = position_side < 0

                strategy.update_position()
                stop_price = None
                take_price = None
                if strategy.stop_loss is not None:
                    _, stop_price = strategy.stop_loss
                if strategy.take_profit is not None:
                    _, take_price = strategy.take_profit

                exit_price = None
                exit_reason = ""

                high_price = float(candle[3])
                low_price = float(candle[4])

                if position_side > 0:
                    if stop_price is not None and low_price <= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                    elif take_price is not None and high_price >= take_price:
                        exit_price = take_price
                        exit_reason = "take_profit"
                else:
                    if stop_price is not None and high_price >= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                    elif take_price is not None and low_price <= take_price:
                        exit_price = take_price
                        exit_reason = "take_profit"

                if exit_price is not None:
                    pnl = position_side * position_qty * (exit_price - entry_price)
                    capital += pnl
                    trades.append(
                        Trade(
                            entry_index=entry_index,
                            exit_index=index,
                            side="long" if position_side > 0 else "short",
                            entry_price=entry_price,
                            exit_price=float(exit_price),
                            qty=position_qty,
                            pnl=float(pnl),
                            return_pct=float(pnl / (abs(position_qty) * entry_price)),
                            exit_reason=exit_reason,
                        )
                    )

                    if exit_reason == "stop_loss":
                        strategy.on_stop_loss()
                    elif exit_reason == "take_profit":
                        strategy.on_take_profit()

                    position_side = 0
                    position_qty = 0.0
                    entry_price = 0.0
                    entry_index = -1
                    strategy.position.reset()
                    strategy.vars["initial_stop"] = None
                    strategy.buy = None
                    strategy.sell = None
                    strategy.stop_loss = None
                    strategy.take_profit = None

            if position_side == 0:
                if strategy.should_long():
                    strategy.go_long()
                    if strategy.buy is not None:
                        qty, price = strategy.buy
                        position_side = 1
                        position_qty = float(qty)
                        entry_price = float(price)
                        entry_index = index
                        strategy.position.qty = abs(position_qty)
                        strategy.position.entry_price = entry_price
                        strategy.position.is_long = True
                        strategy.position.is_short = False
                elif strategy.should_short():
                    strategy.go_short()
                    if strategy.sell is not None:
                        qty, price = strategy.sell
                        position_side = -1
                        position_qty = float(qty)
                        entry_price = float(price)
                        entry_index = index
                        strategy.position.qty = abs(position_qty)
                        strategy.position.entry_price = entry_price
                        strategy.position.is_long = False
                        strategy.position.is_short = True

            if position_side != 0:
                close_price = float(candle[2])
                unrealized = position_side * position_qty * (close_price - entry_price)
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)

        if position_side != 0:
            last_close = float(candles[-1, 2])
            pnl = position_side * position_qty * (last_close - entry_price)
            capital += pnl
            trades.append(
                Trade(
                    entry_index=entry_index,
                    exit_index=total_candles - 1,
                    side="long" if position_side > 0 else "short",
                    entry_price=entry_price,
                    exit_price=last_close,
                    qty=position_qty,
                    pnl=float(pnl),
                    return_pct=float(pnl / (abs(position_qty) * entry_price)),
                    exit_reason="session_close",
                )
            )
            position_side = 0
            position_qty = 0.0
            entry_price = 0.0
            entry_index = -1

        equity_array = np.array(equity_curve, dtype=float)
        total_return = (capital - self.initial_capital) / self.initial_capital

        minutes = total_candles * self.timeframe_minutes
        years = minutes / (60.0 * 24.0 * 365.0)
        if years > 0:
            annual_return = (capital / self.initial_capital) ** (1.0 / years) - 1.0
        else:
            annual_return = 0.0

        if equity_array.size:
            running_max = np.maximum.accumulate(equity_array)
            drawdown = np.where(running_max > 0.0, (equity_array - running_max) / running_max, 0.0)
            max_drawdown = float(abs(drawdown.min()))
        else:
            max_drawdown = 0.0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=float(capital),
            total_return=float(total_return),
            annual_return=float(annual_return),
            max_drawdown=max_drawdown,
            equity_curve=equity_array,
            trades=trades,
        )


# ---------------------------------------------------------------------------
# Parameter sweep helper
# ---------------------------------------------------------------------------


def grid_search(
    candles: np.ndarray,
    param_grid: Dict[str, Sequence[float]],
    targets: Tuple[float, float, int],
) -> Tuple[Dict[str, float], BacktestResult]:
    """Run an exhaustive grid search returning the best configuration.

    Parameters
    ----------
    candles:
        OHLCV matrix used for the simulation.
    param_grid:
        Mapping of attribute names to the candidate values that should be
        evaluated.
    targets:
        Tuple with ``(min_annual_return, max_drawdown, min_trades)`` that
        solutions must satisfy.  When several combinations pass the
        constraints the one with the highest annual return is selected.
    """

    tester = SimpleBacktester(candles)
    best_params: Dict[str, float] = {}
    best_result: BacktestResult | None = None
    min_annual_return, max_allowed_drawdown, min_trades = targets

    keys = list(param_grid.keys())
    values_product = product(*(param_grid[key] for key in keys))

    for combination in values_product:
        params = dict(zip(keys, combination))
        result = tester.run(params=params)

        if result.annual_return < min_annual_return:
            continue
        if result.max_drawdown > max_allowed_drawdown:
            continue
        if result.trade_count < min_trades:
            continue

        if best_result is None or result.annual_return > best_result.annual_return:
            best_result = result
            best_params = params

    if best_result is None:
        raise ValueError("No parameter combination satisfied the optimisation targets.")

    return best_params, best_result

