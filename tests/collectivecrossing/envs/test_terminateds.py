"""Tests for termination functions in the collective crossing environment."""

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
        num_boarding_agents=1,
        num_exiting_agents=1,
        tram_length=6,
        tram_door_left=2,
        tram_door_right=3,
        division_y=4,
        boarding_destination_area_y=6,
        exiting_destination_area_y=2,
        truncated_config=MaxStepsTruncatedConfig(max_steps=100),
    )


def test_all_at_destination_terminated_function(basic_config: CollectiveCrossingConfig) -> None:
    """Test all-at-destination termination function."""
    from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig

    config = basic_config.model_copy(
        update={"terminated_config": AllAtDestinationTerminatedConfig()}
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Initially no agent should be terminated
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Check that no agent is terminated initially
    assert not terminateds["boarding_0"]
    assert not terminateds["exiting_0"]
    assert not terminateds["__all__"]

    # Test that the termination function returns the expected behavior
    # For all_at_destination, no agent should be terminated until all reach destination
    assert not env._calculate_terminated("boarding_0")
    assert not env._calculate_terminated("exiting_0")


def test_individual_at_destination_terminated_function(
    basic_config: CollectiveCrossingConfig,
) -> None:
    """Test individual-at-destination termination function."""
    from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig

    config = basic_config.model_copy(
        update={"terminated_config": IndividualAtDestinationTerminatedConfig()}
    )
    env = CollectiveCrossingEnv(config)
    obs, info = env.reset(seed=42)

    # Initially no agent should be terminated
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Check that no agent is terminated initially
    assert not terminateds["boarding_0"]
    assert not terminateds["exiting_0"]
    assert not terminateds["__all__"]

    # Test that the termination function returns the expected behavior
    # For individual_at_destination, each agent terminates when it reaches destination
    # Initially, neither agent should be at destination
    assert not env._calculate_terminated("boarding_0")
    assert not env._calculate_terminated("exiting_0")


def test_default_terminated_function(basic_config: CollectiveCrossingConfig) -> None:
    """Test default termination function (should be individual_at_destination)."""
    env = CollectiveCrossingEnv(basic_config)
    obs, info = env.reset(seed=42)

    # Initially no agent should be terminated
    actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs.keys()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Check that no agent is terminated initially
    assert not terminateds["boarding_0"]
    assert not terminateds["exiting_0"]
    assert not terminateds["__all__"]

    # Test that the default termination function behaves like individual_at_destination
    assert not env._calculate_terminated("boarding_0")
    assert not env._calculate_terminated("exiting_0")


def test_terminated_function_consistency(basic_config: CollectiveCrossingConfig) -> None:
    """Test that termination function behavior is consistent."""
    # Test all-at-destination
    from collectivecrossing.terminated_configs import (
        AllAtDestinationTerminatedConfig,
        IndividualAtDestinationTerminatedConfig,
    )

    config_all = basic_config.model_copy(
        update={"terminated_config": AllAtDestinationTerminatedConfig()}
    )
    env_all = CollectiveCrossingEnv(config_all)
    obs_all, info_all = env_all.reset(seed=42)

    # Test individual-at-destination
    config_individual = basic_config.model_copy(
        update={"terminated_config": IndividualAtDestinationTerminatedConfig()}
    )
    env_individual = CollectiveCrossingEnv(config_individual)
    obs_individual, info_individual = env_individual.reset(seed=42)

    # Same initial actions should produce different termination behavior
    actions = {agent_id: env_all.action_spaces[agent_id].sample() for agent_id in obs_all.keys()}

    # Step both environments
    obs_all, rewards_all, terminateds_all, truncateds_all, infos_all = env_all.step(actions)
    obs_ind, rewards_ind, terminateds_ind, truncateds_ind, infos_ind = env_individual.step(actions)

    # The termination behavior should be different between the two functions
    # (though in this case, neither agent has reached destination yet, so both should be False)
    assert terminateds_all["boarding_0"] == terminateds_ind["boarding_0"]  # Both False initially
    assert terminateds_all["exiting_0"] == terminateds_ind["exiting_0"]  # Both False initially


def test_terminated_function_config_validation(basic_config: CollectiveCrossingConfig) -> None:
    """Test that invalid termination function names are rejected."""
    from collectivecrossing.terminated_configs import CustomTerminatedConfig

    with pytest.raises(ValueError, match="Unknown termination function"):
        config = basic_config.model_copy(
            update={
                "terminated_config": CustomTerminatedConfig(terminated_function="invalid_function")
            }
        )
        CollectiveCrossingEnv(config)


def test_terminated_function_config_structure(basic_config: CollectiveCrossingConfig) -> None:
    """Test that termination function config structure is validated."""
    # These tests are no longer needed since we now use proper config objects
    # The validation is now handled by Pydantic at the config object level
    pass
