"""Baseline policies for the collective crossing environment."""

from .greedy_policy import GreedyPolicy, create_greedy_policy
from .waiting_policy import WaitingPolicy, create_waiting_policy

__all__ = ["GreedyPolicy", "create_greedy_policy", "WaitingPolicy", "create_waiting_policy"]
