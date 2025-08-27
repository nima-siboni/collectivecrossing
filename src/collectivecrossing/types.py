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

    def __post_init__(self):
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

    def update_position(self, new_position: np.ndarray):
        """Update the agent's position."""
        self.position = np.array(new_position)

    def deactivate(self):
        """Mark the agent as deactivated (set active to False)."""
        if not self.active:
            raise ValueError("Agent is already deactivated.")
        self.active = False

    @property
    def is_boarding(self) -> bool:
        """Check if this is a boarding agent."""
        return self.agent_type == AgentType.BOARDING

    @property
    def is_exiting(self) -> bool:
        """Check if this is an exiting agent."""
        return self.agent_type == AgentType.EXITING

    @property
    def is_terminated(self) -> bool:
        """Check if this agent is terminated (not active)."""
        return not self.active
