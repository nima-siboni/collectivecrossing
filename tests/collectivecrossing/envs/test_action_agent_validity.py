#!/usr/bin/env python3
"""
Tests for action and agent validity checking in CollectiveCrossingEnv.
"""

import pytest
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig


class TestActionAgentValidity:
    """Test cases for the _check_action_agent_validity method."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        config = CollectiveCrossingConfig(
            width=10,
            height=8,
            division_y=4,
            tram_door_x=5,
            tram_door_width=2,
            tram_length=8,
            num_boarding_agents=2,
            num_exiting_agents=1,
            max_steps=100,
            exiting_destination_area_y=1,
            boarding_destination_area_y=7,
        )
        return CollectiveCrossingEnv(config=config)

    def test_valid_agent_valid_action(self, env):
        """Test that valid agent ID and action pass validation."""
        # Reset environment to get valid agent IDs
        env.reset(seed=42)

        # Test with a valid boarding agent and valid action
        agent_id = "boarding_0"
        action = 0  # Move right

        # Should not raise any exception
        env._check_action_agent_validity(agent_id, action)

    def test_valid_agent_invalid_action(self, env):
        """Test that valid agent ID with invalid action raises ValueError."""
        env.reset(seed=42)

        agent_id = "boarding_0"
        invalid_action = 999  # Invalid action

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(agent_id, invalid_action)

        assert "Invalid action" in str(exc_info.value)
        assert str(invalid_action) in str(exc_info.value)
        assert agent_id in str(exc_info.value)

    def test_invalid_agent_valid_action(self, env):
        """Test that invalid agent ID with valid action raises ValueError."""
        env.reset(seed=42)

        invalid_agent_id = "invalid_agent"
        action = 0  # Valid action

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(invalid_agent_id, action)

        assert "Unknown agent ID" in str(exc_info.value)
        assert invalid_agent_id in str(exc_info.value)
        assert "all_agent_ids" in str(exc_info.value)

    def test_invalid_agent_invalid_action(self, env):
        """Test that invalid agent ID with invalid action raises ValueError."""
        env.reset(seed=42)

        invalid_agent_id = "invalid_agent"
        invalid_action = 999

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(invalid_agent_id, invalid_action)

        # Should fail on agent ID check first
        assert "Unknown agent ID" in str(exc_info.value)
        assert invalid_agent_id in str(exc_info.value)

    def test_all_valid_actions(self, env):
        """Test that all valid actions (0-4) pass validation."""
        env.reset(seed=42)

        agent_id = "boarding_0"
        valid_actions = [0, 1, 2, 3, 4]  # All valid actions

        for action in valid_actions:
            # Should not raise any exception
            env._check_action_agent_validity(agent_id, action)

    def test_all_agent_types(self, env):
        """Test that both boarding and exiting agents pass validation."""
        env.reset(seed=42)

        action = 0  # Valid action

        # Test boarding agents
        for agent_id in env.boarding_agent_ids:
            env._check_action_agent_validity(agent_id, action)

        # Test exiting agents
        for agent_id in env.exiting_agent_ids:
            env._check_action_agent_validity(agent_id, action)

    def test_negative_action(self, env):
        """Test that negative action values raise ValueError."""
        env.reset(seed=42)

        agent_id = "boarding_0"
        negative_action = -1

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(agent_id, negative_action)

        assert "Invalid action" in str(exc_info.value)
        assert str(negative_action) in str(exc_info.value)

    def test_large_action(self, env):
        """Test that very large action values raise ValueError."""
        env.reset(seed=42)

        agent_id = "boarding_0"
        large_action = 1000000

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(agent_id, large_action)

        assert "Invalid action" in str(exc_info.value)
        assert str(large_action) in str(exc_info.value)

    def test_empty_string_agent_id(self, env):
        """Test that empty string agent ID raises ValueError."""
        env.reset(seed=42)

        empty_agent_id = ""
        action = 0

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(empty_agent_id, action)

        assert "Unknown agent ID" in str(exc_info.value)
        assert empty_agent_id in str(exc_info.value)

    def test_none_agent_id(self, env):
        """Test that None agent ID raises ValueError."""
        env.reset(seed=42)

        none_agent_id = None
        action = 0

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(none_agent_id, action)

        assert "Unknown agent ID" in str(exc_info.value)

    def test_agent_id_with_spaces(self, env):
        """Test that agent ID with spaces raises ValueError."""
        env.reset(seed=42)

        spaced_agent_id = "boarding 0"
        action = 0

        with pytest.raises(ValueError) as exc_info:
            env._check_action_agent_validity(spaced_agent_id, action)

        assert "Unknown agent ID" in str(exc_info.value)
        assert spaced_agent_id in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
