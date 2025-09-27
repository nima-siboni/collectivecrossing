#!/usr/bin/env python3
"""Tests for truncation logic in CollectiveCrossingEnv."""

import pytest

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig


class TestTruncationLogic:
    """Test cases for the truncation function."""

    @pytest.mark.parametrize(
        "current_step,max_steps,expected",
        [
            # Normal cases
            (5, 10, False),  # Below max steps
            (9, 10, False),  # At max steps minus 1
            (10, 10, True),  # At max steps
            (15, 10, True),  # Above max steps
            # Edge cases
            (0, 1, False),  # Zero step count, min max_steps
            (1, 1, True),  # At max steps
            (0, 10, False),  # Zero step count
            (100000, 100000, True),  # Large numbers within bounds
        ],
    )
    def test_calculate_truncated(self, current_step: int, max_steps: int, expected: bool) -> None:
        """Test the truncation function with various inputs."""
        # Create a basic config
        config = CollectiveCrossingConfig(
            width=10,
            height=8,
            division_y=4,
            tram_door_left=4,
            tram_door_right=5,
            tram_length=6,
            num_boarding_agents=1,
            num_exiting_agents=1,
            exiting_destination_area_y=2,
            boarding_destination_area_y=6,
            truncated_config=MaxStepsTruncatedConfig(max_steps=max_steps),
        )

        env = CollectiveCrossingEnv(config)
        # Set the step count to simulate the current step
        env._step_count = current_step

        # Test with a valid agent ID
        agent_id = "boarding_0"
        result = env._truncated_function.calculate_truncated(agent_id, env)
        assert result is expected, (
            f"Expected {expected} for step {current_step} with max {max_steps}, got {result}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
