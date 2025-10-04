#!/usr/bin/env python3
"""
Example script demonstrating the WaitingPolicy baseline.

This script shows how to use the new WaitingPolicy that implements
coordinated movement where outside agents wait for inside agents to exit.
"""

import logging

from baseline_policies import create_waiting_policy
from collectivecrossing import CollectiveCrossingEnv, Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run a demonstration of the WaitingPolicy."""
    # Create environment configuration
    config = Config(
        width=20,
        height=15,
        num_boarding_agents=3,
        num_exiting_agents=2,
        tram_width=6,
        tram_door_width=2,
        boarding_destination_area_y=14,
        exiting_destination_area_y=0,
    )

    # Create environment
    env = CollectiveCrossingEnv(config)

    # Create waiting policy
    waiting_policy = create_waiting_policy(epsilon=0.1)

    # Run simulation
    logger.info("Starting WaitingPolicy demonstration...")

    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        # Get actions for all agents using the waiting policy
        actions = {}
        for agent_id in env.agents:
            if agent_id in obs:
                action = waiting_policy.get_action(agent_id, obs[agent_id], env)
                actions[agent_id] = action

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Track rewards
        for _agent_id, reward in rewards.items():
            total_reward += reward

        step_count += 1

        # Check if all agents are done
        if all(terminated.values()) or all(truncated.values()):
            logger.info(f"Simulation completed after {step_count} steps")
            break

        # Log progress every 100 steps
        if step_count % 100 == 0:
            logger.info(f"Step {step_count}: Total reward = {total_reward:.2f}")

    logger.info("Final results:")
    logger.info(f"  Total steps: {step_count}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Average reward per step: {total_reward / step_count:.4f}")


if __name__ == "__main__":
    main()
