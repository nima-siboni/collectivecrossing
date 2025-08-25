"""This file contains the agent types for the environment."""
from enum import Enum


class AgentType(Enum):
    """Types of agents in the environment"""

    BOARDING = "boarding"  # Agents trying to get on the tram
    EXITING = "exiting"  # Agents trying to get off the tram
