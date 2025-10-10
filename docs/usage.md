# Usage Guide

## Basic Usage

### Quick Start Example

```python
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.reward_configs import DefaultRewardConfig
from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

# Create configuration with configurable systems
reward_config = DefaultRewardConfig(
    boarding_destination_reward=15.0,
    tram_door_reward=10.0,
    tram_area_reward=5.0,
    distance_penalty_factor=0.1
)

terminated_config = IndividualAtDestinationTerminatedConfig()
truncated_config = MaxStepsTruncatedConfig(max_steps=100)

config = CollectiveCrossingConfig(
    width=12, height=8, division_y=4,
    tram_door_left=4, tram_door_right=6, tram_length=10,
    num_boarding_agents=5, num_exiting_agents=3,
    exiting_destination_area_y=0, boarding_destination_area_y=8,
    render_mode="rgb_array",
    reward_config=reward_config,
    terminated_config=terminated_config,
    truncated_config=truncated_config
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
    tram_door_left=5,            # Left boundary of tram door
    tram_door_right=6,           # Right boundary of tram door
    tram_length=10,              # Length of the tram
    num_boarding_agents=5,       # Number of agents trying to board
    num_exiting_agents=3,        # Number of agents trying to exit
    # Maximum steps are configured in truncated_config
    exiting_destination_area_y=1,    # Y-coordinate for exit destination
    boarding_destination_area_y=7,   # Y-coordinate for boarding destination
    render_mode="rgb_array"      # Rendering mode
)
```

### Automatic Validation

The configuration system automatically validates:
- **Tram parameters** (door position, width, length)
- **Destination areas** (within valid boundaries)
- **Environment bounds** (grid dimensions)
- **Agent counts** (reasonable limits)
- **Render modes** (valid options)

```python
# Invalid configuration will raise descriptive errors
try:
    config = CollectiveCrossingConfig(
        width=10, tram_length=15  # Error: tram length > width
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Reward Configuration

The environment supports multiple reward strategies with configurable parameters:

### Default Reward System

```python
from collectivecrossing.reward_configs import DefaultRewardConfig

# Configure default reward system
reward_config = DefaultRewardConfig(
    boarding_destination_reward=15.0,  # Reward for reaching boarding destination
    tram_door_reward=10.0,            # Reward for reaching tram door
    tram_area_reward=5.0,             # Reward for being in tram area
    distance_penalty_factor=0.1       # Distance-based penalty factor
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    reward_config=reward_config
)
```

### Simple Distance Reward

```python
from collectivecrossing.reward_configs import SimpleDistanceRewardConfig

# Configure simple distance-based rewards
reward_config = SimpleDistanceRewardConfig(
    distance_penalty_factor=0.1  # Penalty based on distance to goal
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    reward_config=reward_config
)
```

### Binary Reward System

```python
from collectivecrossing.reward_configs import BinaryRewardConfig

# Configure binary rewards (goal reached or not)
reward_config = BinaryRewardConfig(
    goal_reward=1.0,      # Reward when goal is reached
    no_goal_reward=0.0    # Reward when goal is not reached
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    reward_config=reward_config
)
```

## Termination Configuration

Configure when episodes should terminate:

### All Agents at Destination

```python
from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig

# Episode terminates only when ALL agents reach their destinations
terminated_config = AllAtDestinationTerminatedConfig()

config = CollectiveCrossingConfig(
    # ... other parameters ...
    terminated_config=terminated_config
)
```

### Individual Agent Termination

```python
from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig

# Each agent terminates individually when reaching its destination
terminated_config = IndividualAtDestinationTerminatedConfig()

config = CollectiveCrossingConfig(
    # ... other parameters ...
    terminated_config=terminated_config
)
```

### Custom Termination

```python
from collectivecrossing.terminated_configs import CustomTerminatedConfig

# Custom termination with specific parameters
terminated_config = CustomTerminatedConfig(
    terminated_function="custom_termination",
    max_steps_per_agent=1000,
    require_all_completion=False,
    timeout_penalty=True
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    terminated_config=terminated_config
)
```

## Truncation Configuration

Configure episode truncation policies:

### Max Steps Truncation

```python
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

# Episode truncates after maximum steps
truncated_config = MaxStepsTruncatedConfig(
    max_steps=1000  # Maximum steps before truncation
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    truncated_config=truncated_config
)
```

### Custom Truncation

```python
from collectivecrossing.truncated_configs import CustomTruncatedConfig

# Custom truncation with advanced parameters
truncated_config = CustomTruncatedConfig(
    truncated_function="custom_truncation",
    max_steps=1000,
    early_truncation_threshold=0.8,  # Truncate early if 80% complete
    require_all_agents_active=False
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    truncated_config=truncated_config
)
```

## Complete Configuration Example

```python
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.reward_configs import DefaultRewardConfig
from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

# Create comprehensive configuration
reward_config = DefaultRewardConfig(
    boarding_destination_reward=20.0,
    tram_door_reward=15.0,
    tram_area_reward=8.0,
    distance_penalty_factor=0.05
)

terminated_config = AllAtDestinationTerminatedConfig()
truncated_config = MaxStepsTruncatedConfig(max_steps=500)

config = CollectiveCrossingConfig(
    width=15, height=10, division_y=5,
    tram_door_left=6, tram_door_right=8, tram_length=12,
    num_boarding_agents=6, num_exiting_agents=4,
    exiting_destination_area_y=1, boarding_destination_area_y=8,
    render_mode="rgb_array",
    reward_config=reward_config,
    terminated_config=terminated_config,
    truncated_config=truncated_config
)

# Create and use environment
env = CollectiveCrossingEnv(config=config)
observations, infos = env.reset(seed=42)
```

## Visualization

### RGB Rendering

```python
import matplotlib.pyplot as plt

# Create environment with RGB rendering
config = CollectiveCrossingConfig(
    width=12, height=8, division_y=4,
    tram_door_left=5, tram_door_right=6, tram_length=10,
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
    tram_door_left=5, tram_door_right=6, tram_length=10,
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

## Environment Extensions

The environment supports various extensions through its configuration system:

### Custom Reward Functions

```python
from collectivecrossing.reward_configs import CustomRewardConfig

# Create custom reward configuration
reward_config = CustomRewardConfig(
    reward_function="custom",
    # Add your custom parameters here
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    reward_config=reward_config
)
```

### Custom Termination Functions

```python
from collectivecrossing.terminated_configs import CustomTerminatedConfig

# Create custom termination configuration
terminated_config = CustomTerminatedConfig(
    terminated_function="custom",
    max_steps_per_agent=1000,
    require_all_completion=False
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    terminated_config=terminated_config
)
```

### Custom Truncation Functions

```python
from collectivecrossing.truncated_configs import CustomTruncatedConfig

# Create custom truncation configuration
truncated_config = CustomTruncatedConfig(
    truncated_function="custom",
    max_steps=1000,
    early_truncation_threshold=0.8
)

config = CollectiveCrossingConfig(
    # ... other parameters ...
    truncated_config=truncated_config
)
```

## Action Space

The environment supports the following actions:

- `0`: Move right
- `1`: Move up
- `2`: Move left
- `3`: Move down
- `4`: Wait (no movement)

## Observation Space

The environment provides configurable observation functions that can be customized for different use cases.

### Default Observation Function

The default observation function provides:
- **Agent's own position** (x, y coordinates)
- **Tram door information** (door center x, division line y, door boundaries)
- **Other agents' positions** (positions of all other agents in the environment)

### Observation Structure

```python
from collectivecrossing.observation_configs import DefaultObservationConfig

# Create observation configuration
observation_config = DefaultObservationConfig()

config = CollectiveCrossingConfig(
    # ... other parameters ...
    observation_config=observation_config
)
```

### Custom Observation Functions

You can create custom observation functions by extending the `ObservationFunction` base class:

```python
from collectivecrossing.observation_configs import ObservationConfig
from collectivecrossing.observations import ObservationFunction

class CustomObservationConfig(ObservationConfig):
    observation_function: str = "custom"
    
    def get_observation_function_name(self) -> str:
        return "custom"

class CustomObservationFunction(ObservationFunction):
    def get_agent_observation(self, agent_id: str, env: "CollectiveCrossingEnv") -> np.ndarray:
        # Implement your custom observation logic here
        # Return observation as numpy array
        pass
```

### Observation Space Properties

```python
# Get observation space for all agents
observation_spaces = env.observation_spaces

# Get observation space for a specific agent
agent_obs_space = env.get_observation_space("boarding_0")
```

## Reward System

Rewards are based on:
- Distance to goal
- Successful goal completion
- Collision penalties
- Time penalties
- Configurable reward strategies

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
