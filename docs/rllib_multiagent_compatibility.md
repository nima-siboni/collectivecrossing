# RLlib MultiAgentEnv Compatibility

This page explains how `CollectiveCrossingEnv` aligns with Ray RLlib's `MultiAgentEnv` API and how to plug it into RLlib training.

Reference: [RLlib MultiAgentEnv API](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html).

## API Conformance

`CollectiveCrossingEnv` follows RLlib's multi-agent return signatures:

- Observations: `Dict[AgentID, obs]`
- Rewards: `Dict[AgentID, float]`
- Terminated: `Dict[AgentID, bool]` with global key "__all__"
- Truncated: `Dict[AgentID, bool]` with global key "__all__"
- Infos: `Dict[AgentID, dict]`

Agent IDs are stable strings like `boarding_0`, `boarding_1`, `exiting_0`. After `reset`, active agent IDs are available via `env.agents`.

### possible_agents, agents, and _agents

- possible_agents: The superset of agent IDs that can appear for a given configuration (e.g., all boarding/exiting indices). This is static for a fixed config and useful for pre-declaring spaces.
- agents: The dynamic set of currently active agent IDs. Populated on `reset` and may shrink as agents terminate. Step/return dicts are keyed by this set. This matches RLlib’s expectation that returns are dictionaries keyed by active agents each step.
- _agents: Internal storage used by the environment to track active agents. Treat this as private; external code should use `agents`/`possible_agents`.

Compatibility note: RLlib does not require `possible_agents`, but fully supports dynamic agent sets via dict-based returns. The environment’s use of `agents` for live IDs and the `"__all__"` key in `terminated`/`truncated` conforms to the [RLlib MultiAgentEnv API](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html).

Observation/action spaces are exposed per-agent via helpers like `env.get_observation_space(agent_id)` and `env.get_action_space(agent_id)` and are gymnasium-compatible, as expected by RLlib.

## Reset and Step

- `reset(seed) -> (obs_dict, info_dict)` returns initial observations and infos for all agents.
- `step(actions_dict) -> (obs, rewards, terminated, truncated, infos)` returns per-agent dicts and sets `terminated["__all__"]`/`truncated["__all__"]` accordingly, matching RLlib's requirements.

See RLlib docs for the exact dictionary structures: [Multi-agent envs](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html).

## Termination and Truncation

- Episode termination policies (all agents vs. individual) are configured via `TerminatedConfig` and surfaced through per-agent flags plus "__all__".
- Truncation policies (e.g., max steps) are configured via `TruncatedConfig` and surfaced similarly.

RLlib expects both termination and truncation dictionaries; this environment provides both.

## Policy Mapping

Standard RLlib policy mapping works out-of-the-box. For example, you can map boarding vs. exiting agents to different policies using `policy_mapping_fn`.

For training orchestration details, see RLlib: [Running actual training experiments](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html#running-actual-training-experiments-with-a-multiagentenv).

## Minimal RLlib Example

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig

# Register env factory for RLlib
register_env(
    "collective_crossing",
    lambda env_config: CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(**env_config)
    ),
)

# Map boarding vs exiting to different policies (example)
def policy_mapping_fn(agent_id, *args, **kwargs):
    return "boarding" if agent_id.startswith("boarding_") else "exiting"

algo = (
    PPOConfig()
    .environment(env="collective_crossing", env_config={
        "width": 12,
        "height": 8,
        "division_y": 4,
        "tram_door_left": 5,
        "tram_door_right": 6,
        "tram_length": 10,
        "num_boarding_agents": 5,
        "num_exiting_agents": 3,
        "exiting_destination_area_y": 1,
        "boarding_destination_area_y": 7,
    })
    .multi_agent(policy_mapping_fn=policy_mapping_fn)
    .build()
)
```

For agent grouping, policy modules, and more advanced multi-agent features, consult RLlib's docs: [Multi-agent envs](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html).
