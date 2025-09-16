"""Utility script for running backtests of the ETH short-term strategy.

The script favours ergonomics: if the :mod:`jesse` package is available it will
attempt to execute the backtest through the Python API; otherwise it prints
clear instructions that explain how to install the dependencies and run the
CLI-based workflow.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Any, Dict

from config import STRATEGY_ROUTES, config

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_with_cli(start_date: str, finish_date: str) -> int:
    """Execute the backtest using the ``jesse`` command line tool."""

    executable = which("jesse")
    if executable is None:
        print(
            "The 'jesse' command line interface is not available. Install the\n"
            "project dependencies first or add Jesse's virtual environment to\n"
            "your PATH."
        )
        return 1

    env = os.environ.copy()
    env.setdefault("JESSE_PROJECT", str(PROJECT_ROOT))
    print(
        f"Running Jesse backtest from {start_date} to {finish_date} using CLI..."
    )
    completed = subprocess.run(
        [executable, "backtest", start_date, finish_date],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    return completed.returncode


def _run_with_python_api(start_date: str, finish_date: str) -> int:
    """Attempt to run a backtest via :mod:`jesse.research` if available."""

    research_spec = importlib.util.find_spec("jesse.research")
    if research_spec is None:
        return 1

    research = importlib.import_module("jesse.research")
    if not hasattr(research, "backtest"):
        return 1

    print(
        f"Running Jesse backtest from {start_date} to {finish_date} using the Python API..."
    )
    routes = STRATEGY_ROUTES
    try:
        results = research.backtest(
            routes=routes,
            start_date=start_date,
            finish_date=finish_date,
        )
    except TypeError:
        # Older Jesse versions expose the function with positional-only
        # parameters.  Fallback to a positional call when keyword arguments are
        # not supported.
        results = research.backtest(routes, start_date, finish_date)  # type: ignore[misc]

    if hasattr(research, "trades_summary"):
        summary = research.trades_summary()
        print("\nSummary statistics:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if isinstance(results, Dict):
        print("\nBacktest results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    return 0


def main(argv: Any = None) -> int:
    """Entry point used by both the CLI and Python invocations."""

    if argv is None:
        argv = sys.argv[1:]

    start_date = config["app"].get("start_date", "2021-01-01")  # type: ignore[assignment]
    finish_date = config["app"].get("finish_date", "2021-12-31")  # type: ignore[assignment]

    # Prefer the Python API because it works well in notebooks and unit tests.
    exit_code = _run_with_python_api(start_date, finish_date)
    if exit_code == 0:
        return 0

    # Fallback to the CLI when the Python integration is not available.
    exit_code = _run_with_cli(start_date, finish_date)
    if exit_code != 0:
        print(
            "\nBacktest did not run. Install the project requirements and make sure\n"
            "that the 'jesse' command is accessible, then re-run the script."
        )
    return exit_code


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(main())
