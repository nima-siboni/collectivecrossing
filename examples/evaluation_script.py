"""
Evaluation script for the CollectiveCrossing environment.

This script loads a trained MultiRLModule and runs an evaluation episode while
creating an animation GIF of the environment roll-out, modeled after
`scripts/run_greedy_policy_demo.py`.
"""

import logging
import os
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.reward_configs import ConstantNegativeRewardConfig
from collectivecrossing.terminated_configs import (
    AllAtDestinationTerminatedConfig,
)
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Evaluating the CollectiveCrossing environment...")

marl_module_checkpoint_path = os.path.join(os.getcwd(), "marl_module_checkpoints", "e463")
# load the MultiRLModule
marl_module = MultiRLModule.from_checkpoint(marl_module_checkpoint_path)

logger.info("MultiRLModule loaded successfully")


def convert_observations_for_marl_module(
    observations: dict,
) -> tuple[dict, dict, list[str], list[str]]:
    """
    Convert the observations to two separate dictionaries for the boarding / exiting agents.

    Args:
    ----
        observations: The observations to convert. This is a dictionary with agent ids as keys and
        observations as values, straight from the step function.

    Returns:
    -------
        A tuple of boarding observations, exiting observations, boarding agent ids,
        exiting agent ids.

    """
    boarding_observations: dict[str, np.ndarray] = {
        k: v for k, v in observations.items() if "boarding" in k
    }
    exiting_observations: dict[str, np.ndarray] = {
        k: v for k, v in observations.items() if "exiting" in k
    }

    # stack the values of each of the dictionaries on top of each other
    boarding_agent_ids = (
        list(boarding_observations.keys()) if len(boarding_observations) > 0 else []
    )
    exiting_agent_ids = list(exiting_observations.keys()) if len(exiting_observations) > 0 else []

    if len(boarding_observations) > 0:
        boarding_observations = np.stack(list(boarding_observations.values()))

    if len(exiting_observations) > 0:
        exiting_observations = np.stack(list(exiting_observations.values()))

    # convert the observations to torch tensors
    if len(boarding_observations) > 0:
        boarding_observations = {Columns.OBS: torch.from_numpy(boarding_observations)}
    else:
        boarding_observations = {}
    if len(exiting_observations) > 0:
        exiting_observations = {Columns.OBS: torch.from_numpy(exiting_observations)}
    else:
        exiting_observations = {}

    return boarding_observations, exiting_observations, boarding_agent_ids, exiting_agent_ids


env_config = {
    "width": 15,
    "height": 8,
    "division_y": 4,
    "tram_door_left": 4,  # Left boundary of tram door (occupied position)
    "tram_door_right": 6,  # Right boundary of tram door (occupied position)
    "tram_length": 10,
    "num_boarding_agents": 2,
    "num_exiting_agents": 2,
    "exiting_destination_area_y": 0,
    "boarding_destination_area_y": 8,
    "truncated_config": MaxStepsTruncatedConfig(max_steps=100),
    "reward_config": ConstantNegativeRewardConfig(step_penalty=-1.0),
    "terminated_config": AllAtDestinationTerminatedConfig(),
}


env = CollectiveCrossingEnv(config=CollectiveCrossingConfig(**env_config))

observations, infos = env.reset()

# Prepare Matplotlib figure for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")

# Initial render
rgb_array = env.render()
img = ax.imshow(rgb_array)

# Keep mutable state for the animation function
current_observations = observations
boarding_obss, exiting_obss, boarding_agent_ids, exiting_agent_ids = (
    convert_observations_for_marl_module(current_observations)
)
episode_completed = False


def animate(frame: int) -> list[Any]:
    """Animate the environment."""
    global \
        current_observations, \
        boarding_obss, \
        exiting_obss, \
        boarding_agent_ids, \
        exiting_agent_ids, \
        episode_completed

    # If episode is completed, reset environment and start new episode
    if episode_completed:
        current_observations, infos = env.reset()
        boarding_obss, exiting_obss, boarding_agent_ids, exiting_agent_ids = (
            convert_observations_for_marl_module(current_observations)
        )
        episode_completed = False

    # Inference for both agent groups

    if len(boarding_obss) > 0:
        boarding_logits = (
            marl_module["boarding"].forward_inference(boarding_obss)["action_dist_inputs"].numpy()
        )
    else:
        boarding_logits = None

    if len(exiting_obss) > 0:
        exiting_logits = (
            marl_module["exiting"].forward_inference(exiting_obss)["action_dist_inputs"].numpy()
        )
    else:
        exiting_logits = None

    if boarding_logits is not None:
        boarding_action = np.argmax(boarding_logits, axis=1)
    else:
        boarding_action = None
    if exiting_logits is not None:
        exiting_action = np.argmax(exiting_logits, axis=1)
    else:
        exiting_action = None

    actions = {boarding_agent_ids[i]: boarding_action[i] for i in range(len(boarding_agent_ids))}
    actions.update({exiting_agent_ids[i]: exiting_action[i] for i in range(len(exiting_agent_ids))})

    # Step environment
    next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
    print(f"Step {frame + 1} - actions: {actions}")
    # Render and update frame
    rgb = env.render()
    img.set_array(rgb)
    ax.set_title(f"Evaluation - Step {frame + 1}")

    # Update observations for next step
    (
        boarding_next,
        exiting_next,
        boarding_ids_next,
        exiting_ids_next,
    ) = convert_observations_for_marl_module(next_observations)

    # Assign back to outer scope vars
    current_observations = next_observations
    boarding_obss = boarding_next
    exiting_obss = exiting_next
    boarding_agent_ids = boarding_ids_next
    exiting_agent_ids = exiting_ids_next

    # Check done
    if terminateds.get("__all__", False) or truncateds.get("__all__", False):
        episode_completed = True
        ax.set_title(f"Evaluation - Step {frame + 1} (Episode Complete)")

    return [img]


# Create and save animation GIF similar to the demo script
max_frames = env.config.truncated_config.max_steps
anim = animation.FuncAnimation(
    fig, animate, frames=max_frames, interval=200, blit=True, repeat=True
)

plt.tight_layout()

gif_filename = "evaluation.gif"
logger.info(f"Saving animation to {gif_filename}...")
writer = PillowWriter(fps=5, metadata={"loop": 0})  # 5 FPS, infinite loop
anim.save(gif_filename, writer=writer, dpi=100)
logger.info(f"Animation saved successfully to {gif_filename}")

# Optionally display the animation window (can be commented out in headless runs)
plt.show()

env.close()
