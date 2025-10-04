#!/usr/bin/env python3
"""Test script to verify wall collision detection is working correctly."""

import numpy as np

from src.collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from src.collectivecrossing.configs import CollectiveCrossingConfig


def test_wall_collision() -> bool:
    """Test that agents cannot pass through walls."""
    # Create a simple configuration
    config = CollectiveCrossingConfig(
        width=10,
        height=8,
        division_y=4,  # Division line at y=4
        tram_door_left=1,  # Door from x=1 to x=2 (relative to tram)
        tram_door_right=2,
        tram_length=6,  # Tram from x=2 to x=7 (centered)
        num_boarding_agents=2,
        num_exiting_agents=2,
        exiting_destination_area_y=0,
        boarding_destination_area_y=6,
    )

    env = CollectiveCrossingEnv(config)

    print("Configuration:")
    print(f"  Grid: {config.width}x{config.height}")
    print(f"  Division line: y={config.division_y}")
    print(f"  Tram boundaries: x={env.tram_left} to x={env.tram_right}")
    print(f"  Door boundaries: x={env.tram_door_left} to x={env.tram_door_right}")
    print()

    # Test cases for wall collision detection
    test_cases = [
        # Test crossing division line through door (should be allowed)
        {
            "name": "Cross division line through door",
            "current": np.array([3, 3]),  # Below division, at door x
            "new": np.array([3, 4]),  # Above division, at door x
            "should_block": False,
        },
        # Test crossing division line through wall (should be blocked)
        {
            "name": "Cross division line through wall",
            "current": np.array([1, 3]),  # Below division, outside door
            "new": np.array([1, 4]),  # Above division, outside door
            "should_block": True,
        },
        # Test crossing tram left wall (should be blocked)
        {
            "name": "Cross tram left wall",
            "current": np.array([2, 5]),  # Inside tram
            "new": np.array([1, 5]),  # Outside tram
            "should_block": True,
        },
        # Test crossing tram right wall (should be blocked)
        {
            "name": "Cross tram right wall",
            "current": np.array([8, 5]),  # At tram boundary
            "new": np.array([9, 5]),  # Outside tram
            "should_block": True,
        },
        # Test normal movement within tram (should be allowed)
        {
            "name": "Normal movement within tram",
            "current": np.array([4, 5]),  # Inside tram
            "new": np.array([5, 5]),  # Inside tram
            "should_block": False,
        },
        # Test normal movement in waiting area (should be allowed)
        {
            "name": "Normal movement in waiting area",
            "current": np.array([3, 2]),  # In waiting area
            "new": np.array([4, 2]),  # In waiting area
            "should_block": False,
        },
        # Test movement within tram at boundary (should be allowed)
        {
            "name": "Movement within tram at boundary",
            "current": np.array([7, 5]),  # Inside tram
            "new": np.array([8, 5]),  # At tram boundary (still inside)
            "should_block": False,
        },
    ]

    print("Testing wall collision detection:")
    all_passed = True

    for test_case in test_cases:
        current_pos = test_case["current"]
        new_pos = test_case["new"]
        expected_block = test_case["should_block"]

        # Test the collision detection
        would_cross = env._would_hit_tram_wall(current_pos, new_pos)

        # Check if result matches expectation
        passed = would_cross == expected_block
        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"  {test_case['name']}: {status}")
        print(
            f"    Current: ({current_pos[0]}, {current_pos[1]}) -> "
            f"New: ({new_pos[0]}, {new_pos[1]})"
            f"Would cross: {would_cross}"
            f"Expected to block: {expected_block}"
            f"Actual blocks: {would_cross}"
        )
        print(f"    Expected to block: {expected_block}, Actually blocks: {would_cross}")

        if not passed:
            all_passed = False
        print()

    print(f"Overall result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    test_wall_collision()
