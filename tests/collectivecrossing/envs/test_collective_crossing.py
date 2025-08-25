import numpy as np
import pytest
from collectivecrossing import CollectiveCrossingEnv


def test_environment_initialization():
    """Test that the environment initializes correctly"""
    env = CollectiveCrossingEnv(
        width=10,
        height=8,
        division_y=4,
        tram_door_x=5,
        tram_door_width=2,
        tram_length=8,
        num_boarding_agents=3,
        num_exiting_agents=2,
        exiting_destination_area_y=0,
        boarding_destination_area_y=7,
    )

    assert env.width == 10
    assert env.height == 8
    assert env.division_y == 4
    assert env.tram_door_x == 5
    assert env.tram_door_width == 2
    assert env.num_boarding_agents == 3
    assert env.num_exiting_agents == 2
    assert env.exiting_destination_area_y == 0
    assert env.boarding_destination_area_y == 7


def test_environment_reset():
    """Test that the environment resets correctly"""
    env = CollectiveCrossingEnv(
        width=12,
        height=6,
        division_y=3,
        tram_door_x=6,
        tram_door_width=2,
        tram_length=10,
        num_boarding_agents=2,
        num_exiting_agents=1,
    )

    observations, infos = env.reset(seed=42)

    # Check that all agents are present
    assert len(observations) == 3  # 2 boarding + 1 exiting
    assert len(infos) == 3

    # Check that boarding agents are in the waiting area
    boarding_agents = [k for k in observations.keys() if k.startswith("boarding")]
    assert len(boarding_agents) == 2

    # Check that exiting agents are in the tram area
    exiting_agents = [k for k in observations.keys() if k.startswith("exiting")]
    assert len(exiting_agents) == 1


def test_agent_movement():
    """Test that agents can move correctly"""
    env = CollectiveCrossingEnv(
        width=10,
        height=6,
        division_y=3,
        tram_door_x=5,
        tram_door_width=2,
        tram_length=8,
        num_boarding_agents=1,
        num_exiting_agents=1,
    )

    observations, _ = env.reset(seed=42)

    # Get initial positions
    initial_positions = {}
    for agent_id in observations.keys():
        obs = observations[agent_id]
        initial_positions[agent_id] = obs[:2]  # First two values are x, y position

    # Take a step with wait action (action 4)
    actions = {agent_id: 4 for agent_id in observations.keys()}
    new_observations, rewards, terminated, truncated, infos = env.step(actions)

    # Check that agents are still present
    assert len(new_observations) == 2

    # Check that positions haven't changed (wait action)
    for agent_id in new_observations.keys():
        new_pos = new_observations[agent_id][:2]
        initial_pos = initial_positions[agent_id]
        assert np.array_equal(new_pos, initial_pos)


def test_agent_termination():
    """Test that agents terminate when reaching their destination areas"""
    env = CollectiveCrossingEnv(
        width=8,
        height=6,
        division_y=3,
        tram_door_x=4,
        tram_door_width=2,
        tram_length=6,
        num_boarding_agents=1,
        num_exiting_agents=1,
        exiting_destination_area_y=0,
        boarding_destination_area_y=5,
    )

    observations, _ = env.reset(seed=42)

    # Manually place agents at their destination areas
    # This is a bit of a hack, but it tests the termination logic
    for agent_id in observations.keys():
        if agent_id.startswith("boarding"):
            env.boarding_agents[agent_id] = np.array([4, 5])  # At boarding destination
        else:
            env.exiting_agents[agent_id] = np.array([4, 0])  # At exiting destination

    # Take a step
    actions = {agent_id: 4 for agent_id in observations.keys()}
    new_observations, rewards, terminated, truncated, infos = env.step(actions)

    # Check that agents are terminated
    for agent_id in terminated.keys():
        if agent_id != "__all__":
            assert terminated[agent_id]


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate errors"""
    # Test tram length exceeding width
    with pytest.raises(ValueError, match="Tram length.*cannot exceed grid width"):
        CollectiveCrossingEnv(width=10, tram_length=15)

    # Test invalid tram door position
    with pytest.raises(ValueError, match="Tram door x-coordinate.*must be within tram boundaries"):
        CollectiveCrossingEnv(tram_door_x=15, tram_length=10)

    # Test invalid exiting destination area
    with pytest.raises(
        ValueError, match="Exiting destination area y-coordinate.*must be within waiting area"
    ):
        CollectiveCrossingEnv(division_y=4, exiting_destination_area_y=5)


def test_rendering():
    """Test that rendering works without errors"""
    env = CollectiveCrossingEnv(
        width=10,
        height=6,
        division_y=3,
        tram_door_x=5,
        tram_door_width=2,
        tram_length=8,
        num_boarding_agents=2,
        num_exiting_agents=1,
    )

    observations, _ = env.reset(seed=42)

    # Test rgb_array rendering
    rgb_array = env.render()
    assert rgb_array.shape == (600, 1000, 3)  # Based on figsize=(12, 8), dpi=100
    assert rgb_array.dtype == np.uint8


def test_observation_space():
    """Test that observations are within the expected space"""
    env = CollectiveCrossingEnv(
        width=10,
        height=6,
        division_y=3,
        tram_door_x=5,
        tram_door_width=2,
        tram_length=8,
        num_boarding_agents=2,
        num_exiting_agents=1,
    )

    observations, _ = env.reset(seed=42)

    for _, obs in observations.items():
        # Check observation shape
        assert obs.shape == env.observation_space.shape
        # Check observation type
        assert obs.dtype == env.observation_space.dtype
        # Check observation bounds
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)


def test_action_space():
    """Test that actions are within the expected space"""
    env = CollectiveCrossingEnv(
        width=10,
        height=6,
        division_y=3,
        tram_door_x=5,
        tram_door_width=2,
        tram_length=8,
        num_boarding_agents=2,
        num_exiting_agents=1,
    )

    observations, _ = env.reset(seed=42)

    # Test valid actions
    valid_actions = {agent_id: 0 for agent_id in observations.keys()}  # Right action
    new_observations, rewards, terminated, truncated, infos = env.step(valid_actions)

    # Test invalid action
    invalid_actions = {agent_id: 10 for agent_id in observations.keys()}  # Invalid action
    with pytest.raises(ValueError):
        env.step(invalid_actions)
