"""Observation functions for the collective crossing environment."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from collectivecrossing.observation_configs import ObservationConfig

if TYPE_CHECKING:
    from collectivecrossing.collectivecrossing import CollectiveCrossingEnv


class ObservationFunction(ABC):
    """Abstract base class for observation functions."""

    def __init__(self, observation_config: ObservationConfig):
        """Initialize the observation function with configuration."""
        self.observation_config = observation_config

    @abstractmethod
    def get_agent_observation(self, agent_id: str, env: "CollectiveCrossingEnv") -> np.ndarray:
        """
        Get observation for a specific agent.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            The observation array for the agent.

        """
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function implementation."""

    def get_agent_observation(self, agent_id: str, env: "CollectiveCrossingEnv") -> np.ndarray:
        """
        Get observation for a specific agent.

        The observation includes:
        - Agent's own position (x, y)
        - Tram door information (door center x, division line y, door left, door right)
        - Positions of all other agents (with placeholder for self)

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            The observation array for the agent.

        """
        agent_pos = env._get_agent_position(agent_id)

        # Start with agent's own position and tram door information
        tram_door_info = np.array(
            [
                env.config.tram_door_x,  # Door center X
                env.config.division_y,  # Division line Y
                env.tram_door_left,  # Door left boundary
                env.tram_door_right,  # Door right boundary
            ]
        )
        # TODO: add active status of the agent
        obs = np.concatenate([agent_pos, tram_door_info])

        # Add positions of all other agents
        # TODO: is this for all agents or only the active ones?
        for other_id in env._agents.keys():
            if other_id != agent_id:
                other_pos = env._get_agent_position(other_id)
                obs = np.concatenate([obs, other_pos])
            else:
                # Use a placeholder for self (will be masked out)
                obs = np.concatenate([obs, np.array([-1, -1])])

        return obs.astype(np.int32)


# Registry of available observation functions
OBSERVATION_FUNCTIONS: dict[str, type[ObservationFunction]] = {
    "default": DefaultObservationFunction,
}


def get_observation_function(observation_config: ObservationConfig) -> ObservationFunction:
    """
    Get an observation function by configuration.

    Args:
    ----
        observation_config: The observation configuration.

    Returns:
    -------
        The observation function instance.

    Raises:
    ------
        ValueError: If the observation function name is not found.

    """
    name = observation_config.get_observation_function_name()
    if name not in OBSERVATION_FUNCTIONS:
        available = ", ".join(OBSERVATION_FUNCTIONS.keys())
        raise ValueError(f"Unknown observation function '{name}'. Available: {available}")

    return OBSERVATION_FUNCTIONS[name](observation_config)
