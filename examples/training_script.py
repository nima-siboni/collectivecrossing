"""Training script for the collective crossing environment."""

import numpy as np

from collectivecrossing.collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig, MaxStepsTruncatedConfig
from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig

env = CollectiveCrossingEnv(
    config=CollectiveCrossingConfig(
        width=20,
        height=8,
        division_y=3,
        tram_door_x=5,
        tram_door_width=2,
        tram_length=12,
        num_boarding_agents=4,
        num_exiting_agents=20,
        render_mode="rgb_array",
        exiting_destination_area_y=0,
        boarding_destination_area_y=7,
        truncated_config=MaxStepsTruncatedConfig(max_steps=20),
        terminated_config=AllAtDestinationTerminatedConfig(),
    )
)


observations, infos = env.reset(seed=123)

actions = {}
for agent_id in observations.keys():
    actions[agent_id] = np.random.randint(0, 5)

observations, rewards, terminated, truncated, infos = env.step(actions)
