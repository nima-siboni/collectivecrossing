"""Agent types and data structures for the collective crossing environment."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AgentType(Enum):
    """Types of agents in the environment."""

    BOARDING = "boarding"  # Agents trying to get on the tram
    EXITING = "exiting"  # Agents trying to get off the tram


@dataclass
class Agent:
    """Represents an agent in the environment with all its properties."""

    id: str
    agent_type: AgentType
    position: np.ndarray
    active: bool
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        """Ensure position is a numpy array."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)

    @property
    def x(self) -> int:
        """Get the x coordinate of the agent's position."""
        return int(self.position[0])

    @property
    def y(self) -> int:
        """Get the y coordinate of the agent's position."""
        return int(self.position[1])

    def update_position(self, new_position: np.ndarray) -> None:
        """Update the agent's position."""
        self.position = np.array(new_position)

    def deactivate(self) -> None:
        """Mark the agent as deactivated (set active to False)."""
        if self.active:
            self.active = False
        else:
            raise ValueError("Agent is already deactivated.")

    @property
    def is_boarding(self) -> bool:
        """Check if this is a boarding agent."""
        return self.agent_type == AgentType.BOARDING

    @property
    def is_exiting(self) -> bool:
        """Check if this is an exiting agent."""
        return self.agent_type == AgentType.EXITING

    def terminate(self) -> None:
        """Mark the agent as terminated (set terminated to True)."""
        if self.terminated:
            raise ValueError("Agent is already terminated.")
        self.terminated = True

    def truncate(self) -> None:
        """Mark the agent as truncated (set truncated to True)."""
        if self.truncated:
            raise ValueError("Agent is already truncated.")
        self.truncated = True

    @property
    def is_terminated(self) -> bool:
        """Check if the agent is terminated."""
        return self.terminated

    @property
    def is_truncated(self) -> bool:
        """Check if the agent is truncated."""
        return self.truncated
