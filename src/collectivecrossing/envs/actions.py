"""This file contains the actions for the agents."""
from enum import Enum


class Actions(Enum):
    """Available actions for agents"""

    right = 0
    up = 1
    left = 2
    down = 3
    wait = 4  # Stay in place
