"""Tests for the collective crossing environment."""

import numpy as np
import pytest

from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig


def test_environment_initialization() -> None:
    """Test that the environment initializes correctly."""
    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=8,
            division_y=4,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
            tram_length=8,
            num_boarding_agents=3,
            num_exiting_agents=2,
            exiting_destination_area_y=0,
            boarding_destination_area_y=7,
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
            render_mode="human",
        )
    )

    assert env.config.width == 10
    assert env.config.height == 8
    assert env.config.division_y == 4
    assert env.config.tram_door_left == 3
    assert env.config.tram_door_right == 4
    assert env.config.num_boarding_agents == 3
    assert env.config.num_exiting_agents == 2
    assert env.config.exiting_destination_area_y == 0
    assert env.config.boarding_destination_area_y == 7


def test_environment_reset() -> None:
    """Test that the environment resets correctly."""
    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=8,
            division_y=4,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
            tram_length=8,
            num_boarding_agents=3,
            num_exiting_agents=2,
            exiting_destination_area_y=0,
            boarding_destination_area_y=7,
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
            render_mode="human",
        )
    )

    observations, infos = env.reset(seed=42)

    # Check that all agents are present
    assert len(observations) == 5  # 3 boarding + 2 exiting
    assert len(infos) == 5

    # Check that boarding agents are in the waiting area
    boarding_agents = [k for k in observations.keys() if k.startswith("boarding")]
    assert len(boarding_agents) == 3

    # Check that exiting agents are in the tram area
    exiting_agents = [k for k in observations.keys() if k.startswith("exiting")]
    assert len(exiting_agents) == 2


def test_agent_movement() -> None:
    """Test that agents can move correctly."""
    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=8,
            division_y=4,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
            tram_length=8,
            num_boarding_agents=3,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=7,
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
            render_mode="human",
        )
    )

    observations, _ = env.reset(seed=42)

    # Get initial positions
    initial_positions = {}
    for agent_id in observations.keys():
        obs = observations[agent_id]
        initial_positions[agent_id] = obs[:2]  # First two values are x, y position

    # Take a step with wait action (action 4)
    actions = dict.fromkeys(observations.keys(), 4)
    new_observations, rewards, terminated, truncated, infos = env.step(actions)

    # Check that agents are still present
    assert len(new_observations) == 4

    # Check that positions haven't changed (wait action)
    for agent_id in new_observations.keys():
        new_pos = new_observations[agent_id][:2]
        initial_pos = initial_positions[agent_id]
        assert np.array_equal(new_pos, initial_pos)


def test_agent_termination() -> None:
    """Test that agents terminate when reaching their destination areas."""
    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=8,
            height=6,
            division_y=3,
            tram_door_left=3,
            tram_door_right=4,
            tram_length=8,
            num_boarding_agents=1,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=5,
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
            render_mode="human",
        )
    )

    observations, _ = env.reset(seed=42)

    # Manually place agents at their destination areas
    # This is a bit of a hack, but it tests the termination logic
    for agent_id in observations.keys():
        if agent_id.startswith("boarding"):
            env._agents[agent_id].update_position(np.array([4, 5]))  # At boarding destination
        else:
            env._agents[agent_id].update_position(np.array([4, 0]))  # At exiting destination

    # Take a step
    actions = dict.fromkeys(observations.keys(), 4)
    new_observations, rewards, terminated, truncated, infos = env.step(actions)

    # Check that agents are terminated
    for agent_id in terminated.keys():
        if agent_id != "__all__":
            assert terminated[agent_id]


def test_rendering() -> None:
    """Test that rendering works without errors."""
    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=6,
            division_y=3,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
            tram_length=8,
            num_boarding_agents=2,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=4,
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
            render_mode="human",
        )
    )

    observations, _ = env.reset(seed=42)

    # Test rgb_array rendering
    rgb_array = env.render()
    assert rgb_array.shape == (800, 1200, 3)  # Based on figsize=(12, 8), dpi=100
    assert rgb_array.dtype == np.uint8


# def test_observation_space():
#     """Test that observations are within the expected space"""
#     env = CollectiveCrossingEnv(
#         width=10,
#         height=6,
#         division_y=3,
#         tram_door_x=5,
#         tram_door_width=2,
#         tram_length=8,
#         num_boarding_agents=2,
#         num_exiting_agents=1,
#     )

#     observations, _ = env.reset(seed=42)

#     for _, obs in observations.items():
#         # Check observation shape
#         assert obs.shape == env.observation_space.shape
#         # Check observation type
#         assert obs.dtype == env.observation_space.dtype
#         # Check observation bounds
#         assert np.all(obs >= env.observation_space.low)
#         assert np.all(obs <= env.observation_space.high)


def test_action_space() -> None:
    """Test that actions are within the expected space."""
    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=6,
            division_y=3,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
            tram_length=8,
            num_boarding_agents=2,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=4,
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
            render_mode="human",
        )
    )

    observations, _ = env.reset(seed=42)

    # Test valid actions
    valid_actions = dict.fromkeys(observations.keys(), 0)  # Right action
    new_observations, rewards, terminated, truncated, infos = env.step(valid_actions)

    # Test invalid action
    invalid_actions = dict.fromkeys(observations.keys(), 10)  # Invalid action
    with pytest.raises(ValueError):
        env.step(invalid_actions)


def test_observation_config() -> None:
    """Test that observation configuration works correctly."""
    from collectivecrossing.observation_configs import (
        DefaultObservationConfig,
        get_observation_config,
    )

    # Test default observation config
    config = DefaultObservationConfig()
    assert config.observation_function == "default"
    assert config.get_observation_function_name() == "default"

    # Test getting observation config by name
    config_from_name = get_observation_config("default")
    assert isinstance(config_from_name, DefaultObservationConfig)
    assert config_from_name.observation_function == "default"

    # Test invalid observation function name
    with pytest.raises(ValueError, match="Unknown observation function"):
        get_observation_config("invalid_function")


def test_default_observation_function() -> None:
    """Test that the default observation function works correctly."""
    from collectivecrossing.observation_configs import DefaultObservationConfig
    from collectivecrossing.observations import DefaultObservationFunction, get_observation_function

    # Test creating observation function
    config = DefaultObservationConfig()
    obs_function = get_observation_function(config)
    assert isinstance(obs_function, DefaultObservationFunction)
    assert obs_function.observation_config.get_observation_function_name() == "default"

    # Test invalid observation function name by creating a mock config
    class InvalidObservationConfig(DefaultObservationConfig):
        def get_observation_function_name(self) -> str:
            return "invalid_function"

    with pytest.raises(ValueError, match="Unknown observation function"):
        get_observation_function(InvalidObservationConfig())


def test_observation_structure() -> None:
    """Test that observations have the correct structure."""
    from collectivecrossing.observation_configs import DefaultObservationConfig

    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=6,
            division_y=3,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
            tram_length=8,
            num_boarding_agents=2,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=4,
            observation_config=DefaultObservationConfig(),
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
        )
    )

    observations, _ = env.reset(seed=42)

    # Check that all agents have observations
    assert len(observations) == 3  # 2 boarding + 1 exiting

    # Check observation structure for each agent
    for _agent_id, obs in observations.items():
        # Observation should be a numpy array
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

        # Calculate expected observation size:
        # 2 (agent position) +
        # 4 (tram door info) +
        # 4 * num_agents (x, y, agent_type, active_status for each agent)
        expected_size = 2 + 4 + 4 * 3  # 3 agents total
        assert obs.shape == (expected_size,)

        # Check that agent position is at the beginning
        agent_pos = obs[:2]
        assert 0 <= agent_pos[0] < env.config.width
        assert 0 <= agent_pos[1] < env.config.height

        # Check tram door info (positions 2-5)
        tram_door_info = obs[2:6]
        assert tram_door_info[0] == (env.tram_door_left + env.tram_door_right) // 2  # Door center X
        assert tram_door_info[1] == env.config.division_y  # Division line Y
        assert tram_door_info[2] == env.tram_door_left  # Door left boundary
        assert tram_door_info[3] == env.tram_door_right  # Door right boundary

        # Check other agent information (positions 6 onwards)
        # Each agent has 4 values: x, y, agent_type, active_status
        other_agent_info = obs[6:]
        assert len(other_agent_info) == 4 * 3  # 4 values per agent, 3 agents total

        # Check that agent types are valid (0 or 1, or -1 for self placeholder)
        for i in range(3):  # 3 agents
            agent_type_idx = i * 4 + 2  # agent_type is at index 2 of each agent's 4 values
            assert other_agent_info[agent_type_idx] in [
                -1,
                0,
                1,
            ]  # -1=self placeholder, 0=boarding, 1=exiting

        # Check that active statuses are valid (0 or 1, or -1 for self placeholder)
        for i in range(3):  # 3 agents
            active_status_idx = i * 4 + 3  # active_status is at index 3 of each agent's 4 values
            assert other_agent_info[active_status_idx] in [
                -1,
                0,
                1,
            ]  # -1=self placeholder, 0=inactive, 1=active


def test_observation_consistency() -> None:
    """Test that observations are consistent across steps."""
    from collectivecrossing.observation_configs import DefaultObservationConfig

    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=6,
            division_y=3,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
            tram_length=8,
            num_boarding_agents=1,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=4,
            observation_config=DefaultObservationConfig(),
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
        )
    )

    observations, _ = env.reset(seed=42)

    # Get initial observations
    initial_obs = observations.copy()

    # Take a step
    actions = dict.fromkeys(observations.keys(), 4)  # Wait action
    new_observations, _, _, _, _ = env.step(actions)

    # Check that observation structure remains the same
    for agent_id in initial_obs.keys():
        assert initial_obs[agent_id].shape == new_observations[agent_id].shape
        assert initial_obs[agent_id].dtype == new_observations[agent_id].dtype

        # Check that tram door info remains constant
        initial_tram_info = initial_obs[agent_id][2:6]
        new_tram_info = new_observations[agent_id][2:6]
        assert np.array_equal(initial_tram_info, new_tram_info)


def test_observation_function_integration() -> None:
    """Test that the observation function integrates correctly with the environment."""
    from collectivecrossing.observation_configs import DefaultObservationConfig
    from collectivecrossing.observations import DefaultObservationFunction

    # Create environment with custom observation config
    config = CollectiveCrossingConfig(
        width=10,
        height=6,
        division_y=3,
        tram_door_left=4,
        tram_door_right=5,
        tram_length=8,
        num_boarding_agents=1,
        num_exiting_agents=1,
        exiting_destination_area_y=0,
        boarding_destination_area_y=4,
        observation_config=DefaultObservationConfig(),
        truncated_config=MaxStepsTruncatedConfig(max_steps=100),
    )

    env = CollectiveCrossingEnv(config=config)

    # Check that the environment uses the correct observation function
    assert isinstance(env._observation_function, DefaultObservationFunction)
    assert env._observation_function.observation_config.get_observation_function_name() == "default"

    # Test that observations are generated correctly
    observations, _ = env.reset(seed=42)

    for agent_id, obs in observations.items():
        # Verify that the observation matches what the function would produce
        expected_obs = env._observation_function.get_agent_observation(agent_id, env)
        assert np.array_equal(obs, expected_obs)
