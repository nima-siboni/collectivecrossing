"""Reward functions for the collective crossing environment."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from collectivecrossing.reward_configs import RewardConfig
from collectivecrossing.types import AgentType
from collectivecrossing.utils.geometry import calculate_distance

if TYPE_CHECKING:
    from collectivecrossing.collectivecrossing import CollectiveCrossingEnv


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    def __init__(self, reward_config: RewardConfig):
        """Initialize the reward function with configuration."""
        self.reward_config = reward_config

    @abstractmethod
    def calculate_reward(self, agent_id: str, env: "CollectiveCrossingEnv") -> float | None:
        """
        Calculate reward for an agent.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            The reward for the agent.

        """
        pass


class DefaultRewardFunction(RewardFunction):
    """Default reward function implementation."""

    def calculate_reward(self, agent_id: str, env: "CollectiveCrossingEnv") -> float | None:
        """
        Calculate reward for an agent.

        The reward is calculated based on the agent's position and type:
        - Boarding agents get positive reward for reaching tram door and boarding destination area
        - Exiting agents get positive reward for reaching exiting destination area
        - Boarding agents get negative reward for moving towards the door
        - Exiting agents get negative reward for moving towards the exit

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            The reward for the agent.

        """
        # if the agent is terminated or truncated, return None
        if env._agents[agent_id].terminated or env._agents[agent_id].truncated:
            return None

        agent_pos = env._get_agent_position(agent_id)
        agent_type = env._agents[agent_id].agent_type

        if agent_type == AgentType.BOARDING:
            # Boarding agents get positive reward for reaching tram door and boarding destination
            # area
            if env.has_agent_reached_destination(agent_id):
                return self.reward_config.boarding_destination_reward
            elif env.is_at_tram_door(agent_id):
                return self.reward_config.tram_door_reward
            elif env.is_in_tram_area(agent_id):
                return self.reward_config.tram_area_reward
            else:
                # Small reward for moving towards the door
                door_center_x = (env.tram_door_left + env.tram_door_right) // 2
                distance_to_door = abs(agent_pos[0] - door_center_x) + (
                    env.config.division_y - agent_pos[1]
                )
                return -distance_to_door * self.reward_config.distance_penalty_factor
        else:  # EXITING
            # Exiting agents get positive reward for reaching exiting destination area
            if env.is_in_exiting_destination_area(agent_id):
                return self.reward_config.boarding_destination_reward  # Use same reward for exiting
            elif not env.is_in_tram_area(agent_id):
                return self.reward_config.tram_area_reward  # Use same reward for progress
            else:
                # Small reward for moving towards exit
                door_center_x = (env.tram_door_left + env.tram_door_right) // 2
                distance_to_exit = abs(agent_pos[0] - door_center_x) + (
                    agent_pos[1] - env.config.division_y
                )
                return distance_to_exit * self.reward_config.distance_penalty_factor


class SimpleDistanceRewardFunction(RewardFunction):
    """Simple distance-based reward function."""

    def calculate_reward(self, agent_id: str, env: "CollectiveCrossingEnv") -> float | None:
        """
        Calculate reward based on distance to goal.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            The reward for the agent.

        """
        # if the agent is terminated or truncated, return None
        if env._agents[agent_id].terminated or env._agents[agent_id].truncated:
            return None

        agent_pos = env._get_agent_position(agent_id)

        goal_pos = env.get_agent_destination_position(agent_id)
        distance = calculate_distance(agent_pos, goal_pos)
        return (
            -distance * self.reward_config.distance_penalty_factor
        )  # Negative reward proportional to distance


class BinaryRewardFunction(RewardFunction):
    """Binary reward function - only rewards for goal completion."""

    def calculate_reward(self, agent_id: str, env: "CollectiveCrossingEnv") -> float | None:
        """
        Calculate binary reward - only positive reward for goal completion.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            The reward for the agent.

        """
        # if the agent is terminated or truncated, return None
        if env._agents[agent_id].terminated or env._agents[agent_id].truncated:
            return None

        agent_pos = env._get_agent_position(agent_id)

        goal_pos = env.get_agent_destination_position(agent_id)
        if np.array_equal(agent_pos, goal_pos):
            return self.reward_config.goal_reward  # Positive reward for reaching goal
        else:
            return self.reward_config.no_goal_reward  # No reward otherwise


class ConstantNegativeRewardFunction(RewardFunction):
    """Constant negative reward function - provides a fixed negative reward per step."""

    def calculate_reward(self, agent_id: str, env: "CollectiveCrossingEnv") -> float | None:
        """
        Calculate constant negative reward - same negative value every step.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            The constant negative reward for the agent.

        """
        # if the agent is terminated or truncated, return None
        if env._agents[agent_id].terminated or env._agents[agent_id].truncated:
            return None
        return self.reward_config.step_penalty


# Registry of available reward functions
REWARD_FUNCTIONS: dict[str, type[RewardFunction]] = {
    "default": DefaultRewardFunction,
    "simple_distance": SimpleDistanceRewardFunction,
    "binary": BinaryRewardFunction,
    "constant_negative": ConstantNegativeRewardFunction,
}


def get_reward_function(reward_config: RewardConfig) -> RewardFunction:
    """
    Get a reward function by configuration.

    Args:
    ----
        reward_config: The reward configuration.

    Returns:
    -------
        The reward function instance.

    Raises:
    ------
        ValueError: If the reward function name is not found.

    """
    name = reward_config.get_reward_function_name()
    if name not in REWARD_FUNCTIONS:
        available = ", ".join(REWARD_FUNCTIONS.keys())
        raise ValueError(f"Unknown reward function '{name}'. Available: {available}")

    return REWARD_FUNCTIONS[name](reward_config)
