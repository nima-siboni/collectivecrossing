# Usage Guide

## Basic Usage

### Quick Start Example

```python
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig

# Create configuration
config = CollectiveCrossingConfig(
    width=12, height=8, division_y=4,
    tram_door_x=6, tram_door_width=2, tram_length=10,
    num_boarding_agents=5, num_exiting_agents=3,
    max_steps=100, render_mode="rgb_array"
)

# Create environment
env = CollectiveCrossingEnv(config=config)

# Reset environment
observations, infos = env.reset(seed=42)

# Take actions for all agents
actions = {
    "boarding_0": 0,  # Move right
    "boarding_1": 1,  # Move up
    "boarding_2": 2,  # Move left
    "boarding_3": 3,  # Move down
    "boarding_4": 4,  # Wait
    "exiting_0": 0,   # Move right
    "exiting_1": 1,   # Move up
    "exiting_2": 2,   # Move left
}

# Step the environment
observations, rewards, terminated, truncated, infos = env.step(actions)

# Render the environment
rgb_array = env.render()
```

## Configuration System

### Configuration Building

The project uses a **type-safe configuration system** with automatic validation:

```python
from collectivecrossing.configs import CollectiveCrossingConfig

# Create a configuration with automatic validation
config = CollectiveCrossingConfig(
    width=12,                    # Environment width
    height=8,                    # Environment height
    division_y=4,                # Y-coordinate of tram/waiting area division
    tram_door_x=6,               # X-coordinate of tram door center
    tram_door_width=2,           # Width of the tram door
    tram_length=10,              # Length of the tram
    num_boarding_agents=5,       # Number of agents trying to board
    num_exiting_agents=3,        # Number of agents trying to exit
    max_steps=100,               # Maximum steps per episode
    exiting_destination_area_y=1,    # Y-coordinate for exit destination
    boarding_destination_area_y=7,   # Y-coordinate for boarding destination
    render_mode="rgb_array"      # Rendering mode
)
```

### Automatic Validation

The configuration system automatically validates:
- **Tram parameters** 🚇 (door position, width, length)
- **Destination areas** 🎯 (within valid boundaries)
- **Environment bounds** 📐 (grid dimensions)
- **Agent counts** 👥 (reasonable limits)
- **Render modes** 🎨 (valid options)

```python
# Invalid configuration will raise descriptive errors
try:
    config = CollectiveCrossingConfig(
        width=10, tram_length=15  # Error: tram length > width
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Visualization

### RGB Rendering

```python
import matplotlib.pyplot as plt

# Create environment with RGB rendering
config = CollectiveCrossingConfig(
    width=12, height=8, division_y=4,
    tram_door_x=6, tram_door_width=2, tram_length=10,
    num_boarding_agents=5, num_exiting_agents=3,
    render_mode="rgb_array"
)
env = CollectiveCrossingEnv(config=config)

# Reset and render
observations, infos = env.reset(seed=42)
rgb_array = env.render()

# Display
plt.figure(figsize=(12, 8))
plt.imshow(rgb_array)
plt.axis('off')
plt.title('Collective Crossing Environment')
plt.show()
```

### ASCII Rendering

```python
# Create environment with ASCII rendering
config = CollectiveCrossingConfig(
    width=12, height=8, division_y=4,
    tram_door_x=6, tram_door_width=2, tram_length=10,
    num_boarding_agents=5, num_exiting_agents=3,
    render_mode="ansi"
)
env = CollectiveCrossingEnv(config=config)

# Reset and render
observations, infos = env.reset(seed=42)
ascii_frame = env.render()

# Print ASCII representation
print(ascii_frame)
```

## Environment Wrappers

The project includes several wrappers for modifying environment behavior:

### Reward Clipping

```python
from collectivecrossing.wrappers import ClipRewardWrapper

env = CollectiveCrossingEnv(config=config)
env = ClipRewardWrapper(env, min_reward=-1.0, max_reward=1.0)
```

### Discrete Actions

```python
from collectivecrossing.wrappers import DiscreteActionsWrapper

env = CollectiveCrossingEnv(config=config)
env = DiscreteActionsWrapper(env)
```

### Weighted Rewards

```python
from collectivecrossing.wrappers import ReacherWeightedRewardWrapper

env = CollectiveCrossingEnv(config=config)
env = ReacherWeightedRewardWrapper(env, weight=2.0)
```

### Relative Positioning

```python
from collectivecrossing.wrappers import RelativePositionWrapper

env = CollectiveCrossingEnv(config=config)
env = RelativePositionWrapper(env)
```

## Action Space

The environment supports the following actions:

- `0`: Move right
- `1`: Move up
- `2`: Move left
- `3`: Move down
- `4`: Wait (no movement)

## Observation Space

Observations include:
- Agent positions
- Goal positions
- Environment state
- Collision information

## Reward System

Rewards are based on:
- Distance to goal
- Successful goal completion
- Collision penalties
- Time penalties

## Multi-Agent Environment

The environment follows the Ray RLlib MultiAgentEnv API:

```python
# Get action space for all agents
action_spaces = env.action_spaces

# Get observation space for all agents
observation_spaces = env.observation_spaces

# Get agent IDs
agent_ids = list(env.agents)
```

## Examples

Check the `examples/` directory for complete usage examples:

```bash
# Run example
uv run python examples/collectivecrossing_example.py
```
