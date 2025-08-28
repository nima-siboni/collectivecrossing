"""Reward function configurations for the collective crossing environment."""

from abc import abstractmethod
from typing import Any

from pydantic import Field

from collectivecrossing.utils.pydantic import ConfigClass


class RewardConfig(ConfigClass):
    """Base configuration class for reward functions."""

    reward_function: str = Field(
        description="Name of the reward function to use",
        examples=["default", "simple_distance", "binary", "custom"],
    )

    @abstractmethod
    def get_reward_function_name(self) -> str:
        """Get the name of the reward function this config corresponds to."""
        return self.reward_function


class DefaultRewardConfig(RewardConfig):
    """Configuration for the default reward function."""

    reward_function: str = Field(default="default", description="Default reward function")

    # Reward values for different states
    boarding_destination_reward: float = Field(
        default=15.0,
        description="Reward for reaching boarding destination area",
        ge=-100.0,
        le=100.0,
    )
    tram_door_reward: float = Field(
        default=10.0,
        description="Reward for reaching tram door",
        ge=-100.0,
        le=100.0,
    )
    tram_area_reward: float = Field(
        default=5.0,
        description="Reward for being in tram area",
        ge=-100.0,
        le=100.0,
    )
    distance_penalty_factor: float = Field(
        default=0.1,
        description="Factor for distance-based penalty",
        ge=0.0,
        le=10.0,
    )

    def get_reward_function_name(self) -> str:
        """Get the name of the reward function this config corresponds to."""
        return "default"


class SimpleDistanceRewardConfig(RewardConfig):
    """Configuration for the simple distance-based reward function."""

    reward_function: str = Field(
        default="simple_distance", description="Simple distance reward function"
    )

    distance_penalty_factor: float = Field(
        default=0.1,
        description="Factor for distance-based penalty",
        ge=0.0,
        le=10.0,
    )

    def get_reward_function_name(self) -> str:
        """Get the name of the reward function this config corresponds to."""
        return "simple_distance"


class BinaryRewardConfig(RewardConfig):
    """Configuration for the binary reward function."""

    reward_function: str = Field(default="binary", description="Binary reward function")

    goal_reward: float = Field(
        default=1.0,
        description="Reward for reaching goal",
        ge=0.0,
        le=100.0,
    )
    no_goal_reward: float = Field(
        default=0.0,
        description="Reward when not at goal",
        ge=-100.0,
        le=100.0,
    )

    def get_reward_function_name(self) -> str:
        """Get the name of the reward function this config corresponds to."""
        return "binary"


class ConstantNegativeRewardConfig(RewardConfig):
    """Configuration for the constant negative reward function."""

    reward_function: str = Field(
        default="constant_negative", description="Constant negative reward function"
    )

    step_penalty: float = Field(
        default=-1.0,
        description="Constant negative reward per step",
        ge=-100.0,
        le=0.0,
    )

    def get_reward_function_name(self) -> str:
        """Get the name of the reward function this config corresponds to."""
        return "constant_negative"


class CustomRewardConfig(RewardConfig):
    """Configuration for custom reward functions."""

    reward_function: str = Field(description="Name of the custom reward function")

    # Generic parameters that can be used by custom reward functions
    time_penalty: float = Field(
        default=0.0,
        description="Penalty per time step",
        ge=-10.0,
        le=0.0,
    )
    goal_bonus: float = Field(
        default=0.0,
        description="Bonus for reaching goal",
        ge=0.0,
        le=100.0,
    )
    collision_penalty: float = Field(
        default=0.0,
        description="Penalty for collisions",
        ge=-100.0,
        le=0.0,
    )
    efficiency_bonus: float = Field(
        default=0.0,
        description="Bonus for efficient paths",
        ge=0.0,
        le=100.0,
    )

    def get_reward_function_name(self) -> str:
        """Get the name of the reward function this config corresponds to."""
        return self.reward_function


# Registry of reward configurations
REWARD_CONFIGS = {
    "default": DefaultRewardConfig,
    "simple_distance": SimpleDistanceRewardConfig,
    "binary": BinaryRewardConfig,
    "constant_negative": ConstantNegativeRewardConfig,
    "custom": CustomRewardConfig,
}


def get_reward_config(reward_function_name: str, **kwargs: Any) -> RewardConfig:
    """
    Get a reward configuration by name.

    Args:
    ----
        reward_function_name: The name of the reward function.
        **kwargs: Additional configuration parameters.

    Returns:
    -------
        The reward configuration instance.

    Raises:
    ------
        ValueError: If the reward function name is not found.

    """
    if reward_function_name not in REWARD_CONFIGS:
        available = ", ".join(REWARD_CONFIGS.keys())
        raise ValueError(
            f"Unknown reward function '{reward_function_name}'. Available: {available}"
        )

    config_class = REWARD_CONFIGS[reward_function_name]

    # Remove reward_function from kwargs if it exists to avoid duplicate parameter
    kwargs.pop("reward_function", None)

    return config_class(reward_function=reward_function_name, **kwargs)
