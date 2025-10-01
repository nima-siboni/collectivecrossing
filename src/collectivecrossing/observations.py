"""Observation functions for the collective crossing environment."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gymnasium as gym
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
        - For each other agent:
            - position (x, y)
            - agent type (0=boarding, 1=exiting)
            - active status (0=inactive, 1=active)

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
                (env.tram_door_left + env.tram_door_right)
                // 2,  # Door center X (occupied positions)
                env.config.division_y,  # Division line Y
                env.tram_door_left,  # Door left occupied position
                env.tram_door_right,  # Door right occupied position
            ]
        )
        obs = np.concatenate([agent_pos, tram_door_info])

        # Add information for all other agents
        for other_id in env._agents.keys():
            if other_id != agent_id:
                other_agent = env._get_agent(other_id)
                other_pos = other_agent.position
                # Agent type: 0 for boarding, 1 for exiting
                agent_type = 1 if other_agent.is_exiting else 0
                # Active status: 0 for inactive, 1 for active
                active_status = 1 if other_agent.active else 0
                other_info = np.array([other_pos[0], other_pos[1], agent_type, active_status])
                obs = np.concatenate([obs, other_info])
            else:
                # Use placeholders for self (will be masked out)
                obs = np.concatenate([obs, np.array([-1, -1, -1, -1])])

        return obs.astype(np.float32)

    def return_agent_observation_space(
        self, agent_id: str, env: "CollectiveCrossingEnv"
    ) -> gym.Space:
        """
        Return the observation space for a specific agent.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        """
        # Observation structure:
        # - Agent's own position: 2 dimensions
        # - Tram door info: 4 dimensions
        # - For each agent (including self): 4 dimensions (x, y, agent_type, active_status)
        # Total: 2 + 4 + 4 * num_agents
        return gym.spaces.Box(
            low=-1,  # Allow -1 for placeholders
            high=max(env.config.width, env.config.height) - 1,
            shape=(2 + 4 + 4 * len(env._agents),),
            dtype=np.float32,
        )


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
