"""Common abstract contract for persistent trading strategies."""
from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.state_persistence import PersistentStrategy


class StrategyBase(PersistentStrategy, ABC):
    """Persistent strategy contract for signal generation and risk management."""

    @abstractmethod
    async def _check_signals(self) -> None:
        """Inspect market data and publish any new entry signals."""

    @abstractmethod
    async def _manage_positions(self) -> None:
        """Manage open positions, including exits and protective adjustments."""

    @abstractmethod
    def _get_risk_params(self) -> Dict[str, Any]:
        """Return this strategy's normalized risk parameters."""