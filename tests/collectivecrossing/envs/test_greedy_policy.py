"""Test script for the greedy baseline policy."""

import os
import sys

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.observation_configs import DefaultObservationConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from baseline_policies.greedy_policy import create_greedy_policy


def test_greedy_policy() -> None:
    """Test the greedy policy with a simple scenario."""
    # Create environment
    config = CollectiveCrossingConfig(
        width=10,
        height=6,
        division_y=3,
        tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
        tram_door_right=4,  # Relative to tram (tram_left + 4 = 5)
        tram_length=8,
        num_boarding_agents=2,
        num_exiting_agents=1,
        exiting_destination_area_y=0,
        boarding_destination_area_y=5,
        observation_config=DefaultObservationConfig(),
        truncated_config=MaxStepsTruncatedConfig(max_steps=50),
    )

    env = CollectiveCrossingEnv(config)
    policy = create_greedy_policy()

    # Reset environment
    observations, infos = env.reset(seed=42)
    terminateds = dict.fromkeys(observations.keys(), False)

    print("Initial state:")
    for agent_id, _obs in observations.items():
        agent = env._get_agent(agent_id)
        print(f"  {agent_id}: pos=({agent.x}, {agent.y}), type={agent.agent_type.value}")

    print("\nRunning greedy policy for 20 steps...")

    # Run policy for several steps
    for step in range(20):
        actions = {}

        # Get actions for all active agents
        for agent_id in env.agents:
            if agent_id in observations and not terminateds.get(agent_id, False):
                action = policy.get_action(agent_id, observations[agent_id], env)
                actions[agent_id] = action

        # Execute actions
        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        # Print step information
        print(f"\nStep {step + 1}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")

        # Check if any agents reached their destination
        for agent_id in env.agents:
            if env.has_agent_reached_destination(agent_id):
                agent = env._get_agent(agent_id)
                print(f"  🎯 {agent_id} reached destination at ({agent.x}, {agent.y})")

        # Check if episode is done
        if all(terminateds.values()) or all(truncateds.values()):
            print(f"\nEpisode finished at step {step + 1}")
            break

    print("\nFinal state:")
    for agent_id in env.agents:
        agent = env._get_agent(agent_id)
        destination = env.get_agent_destination_position(agent_id)
        reached = env.has_agent_reached_destination(agent_id)
        print(
            f"  {agent_id}: pos=({agent.x}, {agent.y}), dest=({destination[0]},"
            f"{destination[1]}), reached={reached}"
        )


def test_policy_consistency() -> None:
    """Test that the policy produces consistent actions for the same state."""
    config = CollectiveCrossingConfig(
        width=8,
        height=4,
        division_y=2,
        tram_door_left=4,
        tram_door_right=4,
        tram_length=6,
        num_boarding_agents=1,
        num_exiting_agents=1,
        exiting_destination_area_y=0,
        boarding_destination_area_y=3,
        observation_config=DefaultObservationConfig(),
        truncated_config=MaxStepsTruncatedConfig(max_steps=20),
    )

    env = CollectiveCrossingEnv(config)
    policy = create_greedy_policy()

    # Reset with same seed multiple times
    for trial in range(3):
        observations, _ = env.reset(seed=42)

        # Get actions for first agent
        first_agent_id = list(observations.keys())[0]
        action1 = policy.get_action(first_agent_id, observations[first_agent_id], env)

        # Reset again with same seed
        observations, _ = env.reset(seed=42)
        action2 = policy.get_action(first_agent_id, observations[first_agent_id], env)

        assert action1 == action2, f"Policy inconsistent: {action1} != {action2}"
        print(f"Trial {trial + 1}: Policy consistent (action={action1})")

    print("✅ Policy consistency test passed!")


def test_greedy_policy_simple() -> None:
    """Simple test of the greedy policy."""
    # Create environment
    config = CollectiveCrossingConfig(
        width=8,
        height=4,
        division_y=2,
        tram_door_left=4,
        tram_door_right=4,
        tram_length=6,
        num_boarding_agents=1,
        num_exiting_agents=1,
        exiting_destination_area_y=0,
        boarding_destination_area_y=3,
        observation_config=DefaultObservationConfig(),
        truncated_config=MaxStepsTruncatedConfig(max_steps=20),
    )

    env = CollectiveCrossingEnv(config)
    policy = create_greedy_policy()

    # Reset environment
    observations, infos = env.reset(seed=42)

    print("Initial state:")
    for agent_id, _obs in observations.items():
        agent = env._get_agent(agent_id)
        destination = env.get_agent_destination_position(agent_id)
        print(
            f"  {agent_id}: pos=({agent.x}, {agent.y}), dest=({destination[0]},"
            f"{destination[1]}), type={agent.agent_type.value}"
        )

    print("\nRunning greedy policy...")

    # Run for a few steps
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")

        # Get actions for all agents
        actions = {}
        for agent_id in env.agents:
            if agent_id in observations:
                action = policy.get_action(agent_id, observations[agent_id], env)
                actions[agent_id] = action
                print(f"  {agent_id}: action={action}")

        # Execute actions
        try:
            observations, rewards, terminateds, truncateds, infos = env.step(actions)

            print(f"  Rewards: {rewards}")
            print(f"  Terminated: {terminateds}")

            # Check if any agents reached their destination
            for agent_id in env.agents:
                if env.has_agent_reached_destination(agent_id):
                    agent = env._get_agent(agent_id)
                    print(f"  🎯 {agent_id} reached destination at ({agent.x}, {agent.y})")

            # Check if episode is done
            if all(terminateds.values()) or all(truncateds.values()):
                print(f"\nEpisode finished at step {step + 1}")
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

    print("\nFinal state:")
    for agent_id in env.agents:
        agent = env._get_agent(agent_id)
        destination = env.get_agent_destination_position(agent_id)
        reached = env.has_agent_reached_destination(agent_id)
        print(
            f"  {agent_id}: pos=({agent.x}, {agent.y}), dest=({destination[0]}"
            f", {destination[1]}), reached={reached}"
        )


if __name__ == "__main__":
    print("Testing greedy baseline policy...")
    print("=" * 50)

    test_policy_consistency()
    print("\n" + "=" * 50)
    test_greedy_policy()

    print("\n✅ All tests completed!")
