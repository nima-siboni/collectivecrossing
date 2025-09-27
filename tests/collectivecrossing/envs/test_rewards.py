"""Tests for reward functions in the collective crossing environment."""

import numpy as np
import pytest

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig


@pytest.fixture
def basic_config() -> CollectiveCrossingConfig:
    """Create a basic configuration for testing."""
    from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

    return CollectiveCrossingConfig(
        width=10,
        height=8,
        division_y=4,
        tram_door_left=4,
        tram_door_right=5,
        tram_length=8,
        num_boarding_agents=1,
        num_exiting_agents=1,
        exiting_destination_area_y=1,
        boarding_destination_area_y=6,
        truncated_config=MaxStepsTruncatedConfig(max_steps=100),
    )


def test_default_reward_function(basic_config: CollectiveCrossingConfig) -> None:
    """Test default reward function."""
    from collectivecrossing.reward_configs import DefaultRewardConfig

    config = basic_config.model_copy(update={"reward_config": DefaultRewardConfig()})
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Check that rewards are returned for all agents
    assert len(rewards) == 2
    assert "boarding_0" in rewards
    assert "exiting_0" in rewards
    # Default rewards should be floats
    assert isinstance(rewards["boarding_0"], float | np.floating)
    assert isinstance(rewards["exiting_0"], float | np.floating)


def test_simple_distance_reward_function(basic_config: CollectiveCrossingConfig) -> None:
    """Test simple distance reward function."""
    from collectivecrossing.reward_configs import SimpleDistanceRewardConfig

    config = basic_config.model_copy(
        update={"reward_config": SimpleDistanceRewardConfig(distance_penalty_factor=0.2)}
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Simple distance rewards should be negative (distance penalties)
    assert rewards["boarding_0"] < 0
    assert rewards["exiting_0"] < 0


def test_binary_reward_function(basic_config: CollectiveCrossingConfig) -> None:
    """Test binary reward function."""
    from collectivecrossing.reward_configs import BinaryRewardConfig

    config = basic_config.model_copy(
        update={"reward_config": BinaryRewardConfig(goal_reward=10.0, no_goal_reward=-1.0)}
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Binary rewards should be exactly the configured values
    assert rewards["boarding_0"] == -1.0
    assert rewards["exiting_0"] == -1.0


def test_constant_negative_reward_function(basic_config: CollectiveCrossingConfig) -> None:
    """Test constant negative reward function."""
    from collectivecrossing.reward_configs import ConstantNegativeRewardConfig

    config = basic_config.model_copy(
        update={"reward_config": ConstantNegativeRewardConfig(step_penalty=-2.5)}
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Constant negative rewards should be exactly the configured penalty
    assert rewards["boarding_0"] == -2.5
    assert rewards["exiting_0"] == -2.5


def test_constant_negative_reward_function_default(basic_config: CollectiveCrossingConfig) -> None:
    """Test constant negative reward function with default penalty."""
    from collectivecrossing.reward_configs import ConstantNegativeRewardConfig

    config = basic_config.model_copy(update={"reward_config": ConstantNegativeRewardConfig()})
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Default constant negative reward should be -1.0
    assert rewards["boarding_0"] == -1.0
    assert rewards["exiting_0"] == -1.0


def test_constant_negative_reward_consistency(basic_config: CollectiveCrossingConfig) -> None:
    """Test that constant negative reward is consistent across steps."""
    from collectivecrossing.reward_configs import ConstantNegativeRewardConfig

    config = basic_config.model_copy(
        update={"reward_config": ConstantNegativeRewardConfig(step_penalty=-3.0)}
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Test multiple steps
    for _step in range(3):
        actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Rewards should be consistent across steps
        assert rewards["boarding_0"] == -3.0
        assert rewards["exiting_0"] == -3.0


def test_get_agent_destination_position(basic_config: CollectiveCrossingConfig) -> None:
    """Test the get_agent_destination_position method."""
    from collectivecrossing.reward_configs import DefaultRewardConfig

    config = basic_config.model_copy(update={"reward_config": DefaultRewardConfig()})
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Test destination positions for both agent types
    boarding_dest = env.get_agent_destination_position("boarding_0")
    exiting_dest = env.get_agent_destination_position("exiting_0")

    # Boarding agents should go to boarding destination area
    assert boarding_dest == (None, 6)
    # Exiting agents should go to exiting destination area
    assert exiting_dest == (None, 1)


def test_custom_default_reward_config(basic_config: CollectiveCrossingConfig) -> None:
    """Test custom default reward configuration."""
    from collectivecrossing.reward_configs import DefaultRewardConfig

    config = basic_config.model_copy(
        update={
            "reward_config": DefaultRewardConfig(
                boarding_destination_reward=50.0,
                tram_door_reward=25.0,
                tram_area_reward=10.0,
                distance_penalty_factor=0.05,
            )
        }
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Should work without errors
    assert len(rewards) == 2
    assert "boarding_0" in rewards
    assert "exiting_0" in rewards


def test_invalid_reward_function(basic_config: CollectiveCrossingConfig) -> None:
    """Test that invalid reward function names raise errors."""
    from collectivecrossing.reward_configs import CustomRewardConfig

    config = basic_config.model_copy(
        update={"reward_config": CustomRewardConfig(reward_function="invalid_function")}
    )

    with pytest.raises(ValueError, match="Unknown reward function"):
        CollectiveCrossingEnv(config)
