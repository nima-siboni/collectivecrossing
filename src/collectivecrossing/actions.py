"""Actions for the agents in the collective crossing environment."""

from enum import Enum

import numpy as np


class Actions(Enum):
    """Available actions for agents."""

    right = 0
    up = 1
    left = 2
    down = 3
    wait = 4  # Stay in place


ACTION_TO_DIRECTION = {
    Actions.right.value: np.array([1, 0]),
    Actions.up.value: np.array([0, 1]),
    Actions.left.value: np.array([-1, 0]),
    Actions.down.value: np.array([0, -1]),
    Actions.wait.value: np.array([0, 0]),
}
