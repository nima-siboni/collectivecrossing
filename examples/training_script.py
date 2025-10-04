"""RLlib training script for the CollectiveCrossing multi-agent environment."""

import logging
import os
import shutil
import uuid

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import (
    CollectiveCrossingConfig,
)
from collectivecrossing.reward_configs import ConstantNegativeRewardConfig
from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Register the environment with RLlib using a factory that builds the env
register_env(
    "collective_crossing",
    lambda env_config: CollectiveCrossingEnv(config=CollectiveCrossingConfig(**env_config)),
)


def policy_mapping_fn(agent_id: str, *args, **kwargs) -> str:  # type: ignore
    """
    Map the agent id to the policy.

    Args:
    ----
        agent_id: The id of the agent.
        args: The arguments to the function.
        kwargs: The keyword arguments to the function.

    Returns:
    -------
        The policy to map the agent to.

    """
    return "boarding" if agent_id.startswith("boarding_") else "exiting"


env_config = {
    "width": 15,
    "height": 8,
    "division_y": 4,
    "tram_door_left": 2,  # Left boundary of tram door (occupied position)
    "tram_door_right": 8,  # Right boundary of tram door (occupied position)
    "tram_length": 10,
    "num_boarding_agents": 4,
    "num_exiting_agents": 4,
    "exiting_destination_area_y": 0,
    "boarding_destination_area_y": 8,
    "truncated_config": MaxStepsTruncatedConfig(max_steps=100),
    "reward_config": ConstantNegativeRewardConfig(step_penalty=-1.0),
    "terminated_config": AllAtDestinationTerminatedConfig(),
}

ray.init()


algo = (
    DQNConfig()
    .environment(
        env="collective_crossing",
        env_config=env_config,
    )
    .multi_agent(policies={"boarding", "exiting"}, policy_mapping_fn=policy_mapping_fn)
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "boarding": RLModuleSpec(),
                "exiting": RLModuleSpec(),
            }
        )
    )
    .env_runners(num_env_runners=4)
    .training(train_batch_size=500)
).build()

# Run a minimal training iteration to verify integration
for i in range(30):
    result = algo.train()

    logger.info(
        " Iteration %d Episode rewards: %.1f, %.1f, %.1f",
        i,
        result["env_runners"]["episode_return_min"],
        result["env_runners"]["episode_return_mean"],
        result["env_runners"]["episode_return_max"],
    )

# save the algo
save_dir = os.path.join(os.getcwd(), "marl_module_checkpoints", str(uuid.uuid4().hex[:4]))
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir
logger.info(f" Saving the MultiRLModule to {save_path}")
algo.env_runner.module.save_to_path(save_path)
ray.shutdown()
