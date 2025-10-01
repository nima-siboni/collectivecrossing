"""Evaluation script for the CollectiveCrossing environment."""

import logging
import os

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Evaluating the CollectiveCrossing environment...")

marl_module_checkpoint_path = os.path.join(os.getcwd(), "marl_module_checkpoints", "8366")
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
    boarding_observations = {k: v for k, v in observations.items() if "boarding" in k}
    exiting_observations = {k: v for k, v in observations.items() if "exiting" in k}

    # stack the values of each of the dictionaries on top of each other
    boarding_agent_ids = list(boarding_observations.keys())
    exiting_agent_ids = list(exiting_observations.keys())

    boarding_observations = np.stack(list(boarding_observations.values()))
    exiting_observations = np.stack(list(exiting_observations.values()))

    # convert the observations to torch tensors
    boarding_observations = {Columns.OBS: torch.from_numpy(boarding_observations)}
    exiting_observations = {Columns.OBS: torch.from_numpy(exiting_observations)}

    return boarding_observations, exiting_observations, boarding_agent_ids, exiting_agent_ids


env_config = {
    "width": 12,
    "height": 8,
    "division_y": 4,
    "tram_door_left": 4,
    "tram_door_right": 6,
    "tram_length": 10,
    "num_boarding_agents": 10,
    "num_exiting_agents": 10,
    "exiting_destination_area_y": 1,
    "boarding_destination_area_y": 7,
}

env = CollectiveCrossingEnv(config=CollectiveCrossingConfig(**env_config))

observations, infos = env.reset()

# convert the observations to two separate dictionaries


boarding_obss, exiting_obss, boarding_agent_ids, exiting_agent_ids = (
    convert_observations_for_marl_module(observations)
)

terminated_all = False
truncated_all = False

iteration_count = 0

while not (terminated_all or truncated_all):
    iteration_count += 1
    logger.info(f" Iteration {iteration_count}")
    boarding_action = np.argmax(
        marl_module["boarding"].forward_inference(boarding_obss)["action_dist_inputs"].numpy(),
        axis=1,
    )
    exiting_action = np.argmax(
        marl_module["exiting"].forward_inference(exiting_obss)["action_dist_inputs"].numpy(), axis=1
    )

    actions = {boarding_agent_ids[i]: boarding_action[i] for i in range(len(boarding_agent_ids))}
    actions.update({exiting_agent_ids[i]: exiting_action[i] for i in range(len(exiting_agent_ids))})

    observations, rewards, terminateds, truncateds, infos = env.step(actions)
    env.render()

    boarding_obss, exiting_obss, boarding_agent_ids, exiting_agent_ids = (
        convert_observations_for_marl_module(observations)
    )

    terminated_all = terminateds.get("__all__", False)
    truncated_all = truncateds.get("__all__", False)

env.close()
