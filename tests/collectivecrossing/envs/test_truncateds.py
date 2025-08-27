#!/usr/bin/env python3
"""Tests for termination and truncation logic in CollectiveCrossingEnv."""

import pytest
from collectivecrossing.collectivecrossing import CollectiveCrossingEnv


class TestTruncationLogic:
    """Test cases for the _is_truncated static method."""

    @pytest.mark.parametrize(
        "current_step,max_steps,expected",
        [
            # Normal cases
            (5, 10, False),  # Below max steps
            (9, 10, False),  # At max steps minus 1
            (10, 10, True),  # At max steps
            (15, 10, True),  # Above max steps
            # Edge cases
            (0, 0, True),  # Zero max steps
            (5, 0, True),  # Above zero max steps
            (0, 10, False),  # Zero step count
            (1000000, 999999, True),  # Large numbers
        ],
    )
    def test_is_truncated(self, current_step, max_steps, expected):
        """Test the _is_truncated method with various inputs."""
        result = CollectiveCrossingEnv._is_truncated(current_step, max_steps)
        assert (
            result is expected
        ), f"Expected {expected} for step {current_step} with max {max_steps}, got {result}"


if __name__ == "__main__":
    pytest.main([__file__])
