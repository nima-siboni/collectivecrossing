"""Tests for reward functions in the collective crossing environment."""

import numpy as np
import pytest

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig

# Note: We use equality comparisons (== True, == False) instead of truthiness checks
# because the environment returns numpy boolean types (np.True_, np.False_) which
# require explicit equality comparison for proper testing.


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
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
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
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
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
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
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
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
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
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
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
        actions = {
            agent_id: env.action_spaces[agent_id].sample()
            for agent_id in obs.keys()
            if env._agents[agent_id].active
        }
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
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
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


def test_rewards_not_issued_for_terminated_agents_individual_termination(
    basic_config: CollectiveCrossingConfig,
) -> None:
    """Test that rewards are not issued for terminated agents with individual termination."""
    from collectivecrossing.reward_configs import SimpleDistanceRewardConfig
    from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig

    config = basic_config.model_copy(
        update={
            "reward_config": SimpleDistanceRewardConfig(distance_penalty_factor=0.2),
            "terminated_config": IndividualAtDestinationTerminatedConfig(),
        }
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Move boarding agent to its destination
    boarding_dest = env.get_agent_destination_position("boarding_0")
    # Force the agent to its destination
    env._agents["boarding_0"].position = np.array([5, boarding_dest[1]], dtype=np.int32)
    # Deactivate the agent so it doesn't move during the step
    env._agents["boarding_0"].deactivate()

    # Take first step - the boarding agent should be terminated
    # Only provide actions for active agents
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Boarding agent should be terminated after first step
    assert terminateds["boarding_0"]

    # Take second step - the terminated boarding agent should not receive rewards
    # Only provide actions for active agents
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Boarding agent should still be terminated and not receive rewards
    assert terminateds["boarding_0"]
    assert "boarding_0" not in rewards  # No reward for terminated agent

    # Exiting agent should still receive rewards
    assert not terminateds["exiting_0"]
    assert "exiting_0" in rewards
    assert isinstance(rewards["exiting_0"], float)


def test_rewards_not_issued_for_terminated_agents_all_termination(
    basic_config: CollectiveCrossingConfig,
) -> None:
    """Test that rewards are not issued for terminated agents with all-at-destination terms."""
    from collectivecrossing.reward_configs import BinaryRewardConfig
    from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig

    config = basic_config.model_copy(
        update={
            "reward_config": BinaryRewardConfig(goal_reward=10.0, no_goal_reward=-1.0),
            "terminated_config": AllAtDestinationTerminatedConfig(),
        }
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Move both agents to their destinations
    boarding_dest = env.get_agent_destination_position("boarding_0")
    exiting_dest = env.get_agent_destination_position("exiting_0")
    env._agents["boarding_0"].position = np.array([5, boarding_dest[1]], dtype=np.int32)
    env._agents["exiting_0"].position = np.array([5, exiting_dest[1]], dtype=np.int32)
    # Deactivate both agents so they don't move during the step
    env._agents["boarding_0"].deactivate()
    env._agents["exiting_0"].deactivate()

    # Take first step - both agents should be terminated
    # Only provide actions for active agents
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Both agents should be terminated after first step
    assert terminateds["boarding_0"]
    assert terminateds["exiting_0"]
    assert terminateds["__all__"]

    # Take second step - no rewards should be issued for terminated agents
    # Only provide actions for active agents
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # No rewards should be issued for terminated agents
    assert "boarding_0" not in rewards
    assert "exiting_0" not in rewards
    assert len(rewards) == 0


def test_rewards_not_issued_for_truncated_agents(basic_config: CollectiveCrossingConfig) -> None:
    """Test that rewards are not issued for truncated agents."""
    from collectivecrossing.reward_configs import ConstantNegativeRewardConfig
    from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

    # Set max steps to 1 to force truncation
    config = basic_config.model_copy(
        update={
            "reward_config": ConstantNegativeRewardConfig(step_penalty=-2.5),
            "truncated_config": MaxStepsTruncatedConfig(max_steps=1),
        }
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Take first step - should trigger truncation
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Both agents should be truncated after first step
    assert truncateds["boarding_0"]
    assert truncateds["exiting_0"]
    assert truncateds["__all__"]

    # Take second step - no rewards should be issued for truncated agents
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # No rewards should be issued for truncated agents
    assert "boarding_0" not in rewards
    assert "exiting_0" not in rewards
    assert len(rewards) == 0


def test_default_reward_function_respects_termination(
    basic_config: CollectiveCrossingConfig,
) -> None:
    """Test that DefaultRewardFunction respects termination status."""
    from collectivecrossing.reward_configs import DefaultRewardConfig
    from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig

    config = basic_config.model_copy(
        update={
            "reward_config": DefaultRewardConfig(),
            "terminated_config": IndividualAtDestinationTerminatedConfig(),
        }
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Move boarding agent to its destination
    boarding_dest = env.get_agent_destination_position("boarding_0")
    env._agents["boarding_0"].position = np.array([5, boarding_dest[1]], dtype=np.int32)
    # Deactivate the agent so it doesn't move during the step
    env._agents["boarding_0"].deactivate()

    # Take first step - boarding agent should be terminated
    # Only provide actions for active agents
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Boarding agent should be terminated after first step
    assert terminateds["boarding_0"]

    # Take second step - terminated boarding agent should not receive rewards
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Boarding agent should still be terminated and not receive rewards
    assert terminateds["boarding_0"]
    assert "boarding_0" not in rewards

    # Exiting agent should still receive rewards
    assert not terminateds["exiting_0"]
    assert "exiting_0" in rewards


def test_mixed_termination_states(basic_config: CollectiveCrossingConfig) -> None:
    """Test rewards when some agents are terminated and others are not."""
    from collectivecrossing.reward_configs import SimpleDistanceRewardConfig
    from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig

    config = basic_config.model_copy(
        update={
            "reward_config": SimpleDistanceRewardConfig(distance_penalty_factor=0.2),
            "terminated_config": IndividualAtDestinationTerminatedConfig(),
        }
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Move only the boarding agent to its destination
    boarding_dest = env.get_agent_destination_position("boarding_0")
    env._agents["boarding_0"].position = np.array([5, boarding_dest[1]], dtype=np.int32)
    # Deactivate the boarding agent so it doesn't move during the step
    env._agents["boarding_0"].deactivate()

    # Take first step - boarding agent should be terminated
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Boarding agent should be terminated after first step
    assert terminateds["boarding_0"]

    # Take second step - terminated boarding agent should not receive rewards
    actions = {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in obs.keys()
        if env._agents[agent_id].active
    }
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Boarding agent should still be terminated and not receive rewards
    assert terminateds["boarding_0"]
    assert "boarding_0" not in rewards

    # Exiting agent should not be terminated and should receive rewards
    assert not terminateds["exiting_0"]
    assert "exiting_0" in rewards
    assert isinstance(rewards["exiting_0"], float)
    assert rewards["exiting_0"] < 0  # Should be negative distance penalty

    # Only one reward should be issued
    assert len(rewards) == 1


def test_reward_function_returns_none_for_terminated_agent() -> None:
    """Test that reward functions return None for terminated agents."""
    from collectivecrossing.reward_configs import SimpleDistanceRewardConfig
    from collectivecrossing.rewards import SimpleDistanceRewardFunction

    # Create a mock environment with a terminated agent
    class MockEnv:
        def __init__(self) -> None:
            self._agents = {
                "agent_0": type("Agent", (), {"terminated": True, "truncated": False})()
            }

        def _get_agent_position(self, agent_id: str) -> np.ndarray:
            return np.array([0, 0])

        def get_agent_destination_position(self, agent_id: str) -> np.ndarray:
            return np.array([5, 5])

    reward_config = SimpleDistanceRewardConfig(distance_penalty_factor=0.2)
    reward_function = SimpleDistanceRewardFunction(reward_config)
    mock_env = MockEnv()

    # Should return None for terminated agent
    reward = reward_function.calculate_reward("agent_0", mock_env)
    assert reward is None


def test_reward_function_returns_none_for_truncated_agent() -> None:
    """Test that reward functions return None for truncated agents."""
    from collectivecrossing.reward_configs import BinaryRewardConfig
    from collectivecrossing.rewards import BinaryRewardFunction

    # Create a mock environment with a truncated agent
    class MockEnv:
        def __init__(self) -> None:
            self._agents = {
                "agent_0": type("Agent", (), {"terminated": False, "truncated": True})()
            }

        def _get_agent_position(self, agent_id: str) -> np.ndarray:
            return np.array([0, 0])

        def get_agent_destination_position(self, agent_id: str) -> np.ndarray:
            return np.array([5, 5])

    reward_config = BinaryRewardConfig(goal_reward=10.0, no_goal_reward=-1.0)
    reward_function = BinaryRewardFunction(reward_config)
    mock_env = MockEnv()

    # Should return None for truncated agent
    reward = reward_function.calculate_reward("agent_0", mock_env)
    assert reward is None
