"""Parameter optimisation helper for the ETH short-term strategy.

The optimisation process runs directly against the strategy implementation
bundled with this repository by leveraging the lightweight simulator in
``research_utils``.  The goal is to discover parameter combinations that
achieve the targets stated in the project README:

* annualised return above 50%
* maximum drawdown below 5%
* at least 50 trades during the test period

The optimisation operates on a deterministic synthetic dataset.  While the
data does not represent real ETH/USDT price action it allows us to quickly
iterate on the logic, compare relative improvements, and document the
trade-offs of the strategy configuration.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from research_utils import SimpleBacktester, generate_synthetic_candles


TARGETS = (0.50, 0.05, 50)


def _print_result(label: str, metrics) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Annual return: {metrics.annual_return * 100:.2f}%")
    print(f"Total return:  {metrics.total_return * 100:.2f}%")
    print(f"Max drawdown:  {metrics.max_drawdown * 100:.2f}%")
    print(f"Trades:        {metrics.trade_count}")
    print(f"Win rate:      {metrics.win_rate * 100:.2f}%")


def main() -> int:
    candles = generate_synthetic_candles()
    tester = SimpleBacktester(candles)

    baseline_params = {
        "risk_per_trade": 0.02,
        "fixed_stop": 0.015,
        "fixed_take_profit": 0.03,
        "atr_multiplier": 1.5,
        "cooldown_period": 4,
        "volume_multiplier": 1.2,
    }
    baseline = tester.run(params=baseline_params)
    _print_result("Baseline", baseline)

    param_options: Dict[str, list[float]] = {
        "risk_per_trade": [0.018, 0.02, 0.022, 0.024],
        "fixed_stop": [0.012, 0.014, 0.015, 0.018],
        "fixed_take_profit": [0.028, 0.03, 0.032, 0.035],
        "atr_multiplier": [1.2, 1.4, 1.5, 1.7],
        "cooldown_period": [3, 4, 5],
        "volume_multiplier": [1.15, 1.2, 1.25],
    }

    rng = np.random.default_rng(21)
    best_params: Dict[str, float] = {}
    best_result = None

    trials = 80
    for _ in range(trials):
        candidate = {key: float(rng.choice(values)) for key, values in param_options.items()}
        result = tester.run(params=candidate)
        if result.trade_count < TARGETS[2]:
            continue
        if result.max_drawdown > TARGETS[1]:
            continue
        if result.annual_return < TARGETS[0]:
            continue
        if best_result is None or result.annual_return > best_result.annual_return:
            best_result = result
            best_params = candidate

    if best_result is None:
        print("No configuration met the optimisation targets. Try expanding the search space.")
        return 1

    print("\nOptimised parameters")
    print("--------------------")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    _print_result("Optimised", best_result)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(main())

