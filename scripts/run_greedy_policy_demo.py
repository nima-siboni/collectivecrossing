#!/usr/bin/env python3
"""Demo script for running the greedy baseline policy with statistics and animation."""

from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from baseline_policies.greedy_policy import create_greedy_policy
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.reward_configs import ConstantNegativeRewardConfig
from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig


def run_greedy_policy_with_stats(
    epsilon: float,
) -> tuple[CollectiveCrossingEnv, list[dict[str, tuple[int, int, str]]], list[float]]:
    """Run the greedy policy and collect statistics."""
    print("🚇 Running Greedy Policy Demo")
    print("=" * 50)

    # Create environment
    config = CollectiveCrossingConfig(
        width=15,
        height=8,
        division_y=4,
        tram_door_left=4,  # Left boundary of tram door (occupied position)
        tram_door_right=7,  # Right boundary of tram door (occupied position)
        tram_length=10,
        num_boarding_agents=5,
        num_exiting_agents=10,
        exiting_destination_area_y=0,
        boarding_destination_area_y=8,
        truncated_config=MaxStepsTruncatedConfig(max_steps=200),
        reward_config=ConstantNegativeRewardConfig(step_penalty=-1.0),
        terminated_config=IndividualAtDestinationTerminatedConfig(),
    )

    env = CollectiveCrossingEnv(config)
    policy = create_greedy_policy(epsilon=epsilon)

    # Reset environment
    observations, infos = env.reset()

    # Statistics tracking
    total_steps = 0
    agents_completed = {"boarding": 0, "exiting": 0}
    step_rewards = []
    agent_positions_history = []

    print("Initial state:")
    print(f"  Boarding agents: {config.num_boarding_agents}")
    print(f"  Exiting agents: {config.num_exiting_agents}")
    print(f"  Environment size: {config.width}x{config.height}")
    print(
        f"  Tram door at: ({config.tram_door_left}-{config.tram_door_right}, {config.division_y})"
    )
    print(f"  Epsilon (randomness): {epsilon}")

    # Run the policy
    terminateds = dict.fromkeys(observations.keys(), False)
    max_steps = config.truncated_config.max_steps

    while total_steps < max_steps:
        total_steps += 1

        # Get actions for all active agents
        actions = {}
        for agent_id in env.agents:
            if agent_id in observations and not terminateds.get(agent_id, False):
                # Double check agent is still active
                if agent_id in env._agents and env._agents[agent_id].active:
                    action = policy.get_action(agent_id, observations[agent_id], env)
                    actions[agent_id] = action

        # Execute actions
        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        # Track rewards
        step_rewards.append(sum(rewards.values()))

        # Track agent positions for animation
        positions = {}
        for agent_id in env.agents:
            agent = env._get_agent(agent_id)
            positions[agent_id] = (agent.x, agent.y, agent.agent_type.value)
        agent_positions_history.append(positions)

        # Check if episode is done
        if all(terminateds.values()) or all(truncateds.values()):
            print(f"\nEpisode finished at step {total_steps}")

            # Check for completed agents
            for agent_id in env.possible_agents:
                if env.has_agent_reached_destination(agent_id):
                    agent = env._get_agent(agent_id)
                    if agent.agent_type.value == "boarding":
                        agents_completed["boarding"] += 1
                    else:
                        agents_completed["exiting"] += 1
                    print(
                        f"  Step {total_steps}: 🎯 {agent_id} reached destination at"
                        f" ({agent.x}, {agent.y})"
                    )

            break

    # Print final statistics
    print("\n📊 Final Statistics:")
    print(f"  Total steps: {total_steps}")
    print(
        f"  Boarding agents completed: {agents_completed['boarding']}/{config.num_boarding_agents}"
    )
    print(f"  Exiting agents completed: {agents_completed['exiting']}/{config.num_exiting_agents}")
    print(
        f"  Total agents completed: {sum(agents_completed.values())}/{
            config.num_boarding_agents + config.num_exiting_agents
        }"
    )
    print(f"  Average reward per step: {np.mean(step_rewards):.2f}")
    print(f"  Total reward: {sum(step_rewards):.2f}")

    return env, agent_positions_history, step_rewards


def create_greedy_policy_animation(
    env: CollectiveCrossingEnv,
    epsilon: float,
) -> str:
    """Create animation of the greedy policy run."""
    print("\n🎬 Creating animation...")

    # Reset environment for animation
    observations, infos = env.reset()

    # Create figure and axis (single plot like test_animation)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    # Initial frame
    rgb_array = env.render()
    img = ax.imshow(rgb_array)

    # Store observations for animation function
    current_observations = observations

    # Create greedy policy with epsilon
    greedy_policy = create_greedy_policy(epsilon=epsilon)

    # Track episode completion
    episode_completed = False

    def animate(frame: int) -> list[Any]:
        """Animation function."""
        nonlocal current_observations, episode_completed

        # If episode is completed, reset environment and start new episode
        if episode_completed:
            current_observations, infos = env.reset()
            episode_completed = False

        # Generate actions using greedy policy for active agents only
        actions = {}
        for agent_id in current_observations.keys():
            # Only get action if agent is still active
            if agent_id in env._agents and env._agents[agent_id].active:
                actions[agent_id] = greedy_policy.get_action(
                    agent_id, current_observations[agent_id], env
                )

        # Step the environment
        current_observations, rewards, terminated, truncated, infos = env.step(actions)

        # Render new frame
        rgb_array = env.render()
        img.set_array(rgb_array)

        # Update title with current step
        ax.set_title(f"Greedy Policy - Step {frame + 1}")

        # Check if episode is done
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            episode_completed = True
            ax.set_title(f"Greedy Policy - Step {frame + 1} (Episode Complete)")

        return [img]

    # Create animation - use the same max_steps as the environment
    max_frames = env.config.truncated_config.max_steps
    anim = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=200, blit=True, repeat=True
    )

    plt.tight_layout()

    # Save animation as GIF
    gif_filename = "greedy_policy_demo.gif"
    print(f"Saving animation to {gif_filename}...")

    # Save with PillowWriter for better GIF quality
    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=5, metadata={"loop": 0})  # 5 FPS, infinite loop
    anim.save(gif_filename, writer=writer, dpi=100)

    print(f"Animation saved successfully to {gif_filename}")

    # Display the animation
    plt.show()

    return gif_filename


def main() -> None:
    """Run the greedy policy demo."""
    # Set epsilon value for randomness (0.0 = pure greedy, 1.0 = pure random)
    epsilon = 0.0  # 10% random actions

    # Run greedy policy and collect statistics
    env, agent_positions_history, step_rewards = run_greedy_policy_with_stats(epsilon=epsilon)

    # Create animation
    gif_filename = create_greedy_policy_animation(env, epsilon=epsilon)

    print("\n✅ Demo completed successfully!")
    print(f"📁 Animation saved as: {gif_filename}")
    print(f"🎲 Epsilon used: {epsilon}")


if __name__ == "__main__":
    main()
