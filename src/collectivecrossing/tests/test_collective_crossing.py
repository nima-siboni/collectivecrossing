#!/usr/bin/env python3
"""Test script for the CollectiveCrossingEnv multi-agent environment."""

import numpy as np
from collectivecrossing import CollectiveCrossingEnv


def test_collective_crossing_env():
    """Test the collective crossing environment with random actions."""
    # Create environment with rectangular geometry
    env = CollectiveCrossingEnv(
        width=12,
        height=8,
        division_y=4,  # Horizontal division at y=4
        tram_door_x=6,  # Door center at x=6
        tram_door_width=2,  # Door width of 2
        tram_length=12,  # Tram length (horizontal dimension)
        num_boarding_agents=5,
        num_exiting_agents=3,
        max_steps=50,
        termination_strip_y=0,  # Termination strip at bottom border
    )

    print("=== Collective Crossing Environment Test ===\n")

    # Reset environment
    observations, infos = env.reset(seed=42)

    print("Environment configuration:")
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Division line: y = {env.division_y}")
    print(f"  Tram length: {env.tram_length} (x=0 to x={env.tram_right})")
    print(f"  Tram door: center at x={env.tram_door_x}, width={env.tram_door_width}")
    print(f"  Door boundaries: x={env.tram_door_left} to {env.tram_door_right}")
    print(f"  Termination strip: y = {env.termination_strip_y}")
    print(f"  Boarding agents: {list(env.boarding_agents.keys())}")
    print(f"  Exiting agents: {list(env.exiting_agents.keys())}")
    print()

    # Print initial render
    print("Initial environment state:")
    print(env.render())
    print()

    # Run a few steps with random actions
    for step in range(15):
        print(f"=== Step {step + 1} ===")

        # Generate random actions for active agents only
        actions = {}
        for agent_id in observations.keys():
            actions[agent_id] = np.random.randint(0, 5)  # Random action

        print(f"Actions: {actions}")

        # Step the environment
        observations, rewards, terminated, truncated, infos = env.step(actions)

        print(f"Rewards: {rewards}")
        print(f"Terminated: {terminated}")

        # Print current state
        print("Current environment state:")
        print(env.render())

        # Check if episode is done
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            print("Episode finished!")
            break

        print()

    print("Test completed!")


def test_agent_observation():
    """Test agent observation structure."""
    env = CollectiveCrossingEnv(
        width=10,
        height=6,
        division_y=3,
        tram_door_x=5,
        tram_door_width=2,
        tram_length=10,
        num_boarding_agents=3,
        num_exiting_agents=2,
        termination_strip_y=0,
    )

    observations, infos = env.reset(seed=123)

    print("=== Agent Observation Test ===")
    print(f"Total agents: {len(observations)}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print()

    for agent_id in observations.keys():
        obs = observations[agent_id]
        agent_type = infos[agent_id]["agent_type"]
        print(f"Agent {agent_id} ({agent_type}):")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Agent position: [{obs[0]}, {obs[1]}]")
        print(
            f"  Tram door info: center_x={obs[2]}, division_y={obs[3]}, left={obs[4]}, "
            f"right={obs[5]}"
        )
        print()


def test_geometry_configurations():
    """Test different geometry configurations."""
    configs = [
        {
            "name": "Wide tram, narrow door",
            "width": 15,
            "height": 8,
            "division_y": 4,
            "tram_door_x": 7,
            "tram_door_width": 1,
        },
        {
            "name": "Narrow tram, wide door",
            "width": 8,
            "height": 10,
            "division_y": 5,
            "tram_door_x": 4,
            "tram_door_width": 3,
        },
        {
            "name": "Asymmetric division",
            "width": 12,
            "height": 8,
            "division_y": 6,
            "tram_door_x": 6,
            "tram_door_width": 2,
        },
    ]

    for config in configs:
        print(f"\n=== Testing: {config['name']} ===")

        env = CollectiveCrossingEnv(
            width=config["width"],
            height=config["height"],
            division_y=config["division_y"],
            tram_door_x=config["tram_door_x"],
            tram_door_width=config["tram_door_width"],
            num_boarding_agents=2,
            num_exiting_agents=1,
            termination_strip_y=0,
        )

        observations, infos = env.reset(seed=42)

        print(f"Grid: {env.width} x {env.height}")
        print(f"Division: y = {env.division_y}")
        print(f"Door: x = {env.tram_door_x}, width = {env.tram_door_width}")
        print(f"Door boundaries: x = {env.tram_door_left} to {env.tram_door_right}")
        print()
        print(env.render())


if __name__ == "__main__":
    print("Running Collective Crossing Environment Tests...\n")

    test_agent_observation()
    print("\n" + "=" * 50 + "\n")
    test_geometry_configurations()
    print("\n" + "=" * 50 + "\n")
    test_collective_crossing_env()
