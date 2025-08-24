# Collective Crossing

A multi-agent reinforcement learning project for crowd simulation and collective behavior modeling.

## Environments

This project includes several multi-agent environments:

1. **CrowdCrossing** - A simple single-agent grid world environment
2. **CollectiveCrossingEnv** - A multi-agent environment simulating collective crossing scenarios

### Collective Crossing Environment

The `CollectiveCrossingEnv` simulates a scenario where multiple agents interact in a tram boarding/exiting situation with a rectangular geometry:

#### Geometry:
- **Rectangular domain** divided by a horizontal line
- **Upper part**: Tram area where exiting agents start
- **Lower part**: Waiting area where boarding agents start
- **Configurable tram door**: Position and width can be customized
- **Configurable division line**: Y-coordinate of the horizontal division

#### Features:
- **Boarding agents**: Start in lower area, try to reach tram door
- **Exiting agents**: Start in upper area, try to exit tram area
- **Collision avoidance**: Agents cannot occupy the same position
- **Configurable parameters**: Grid size, door position, number of agents
- **Ray RLlib compatible**: Uses the [MultiAgentEnv API](https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html)
- **Multiple rendering modes**: ASCII text and RGB array for matplotlib visualization

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd collectivecrossing
   ```

2. **Install dependencies**
   ```bash
   # Install main dependencies
   uv sync
   
   # Install development dependencies
   uv sync --dev
   ```

3. **Set up pre-commit hooks**
   ```bash
   # Install pre-commit hooks (automatically configured to use tool-config.toml)
   uv run pre-commit install
   ```

## Development

### Running the project

```bash
# Run with uv
uv run python -m collectivecrossing
```

### Using the Environments

#### Basic Usage

```python
from crowdcrossing.envs import CollectiveCrossingEnv

# Create environment with rectangular geometry
env = CollectiveCrossingEnv(
    width=12,
    height=8,
    division_y=4,  # Horizontal division at y=4
    tram_door_x=6,  # Door center at x=6
    tram_door_width=2,  # Door width of 2
    num_boarding_agents=5,
    num_exiting_agents=3
)

# Reset environment
observations, infos = env.reset(seed=42)

# Take actions
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

observations, rewards, terminated, truncated, infos = env.step(actions)

# Render the environment
print(env.render())

# For RGB rendering with matplotlib
env_rgb = CollectiveCrossingEnv(
    width=12, height=8, division_y=4, tram_door_x=6, tram_door_width=2,
    num_boarding_agents=5, num_exiting_agents=3, render_mode="rgb_array"
)
observations, infos = env_rgb.reset(seed=42)
rgb_array = env_rgb.render()

import matplotlib.pyplot as plt
plt.imshow(rgb_array)
plt.axis('off')
plt.show()
```

#### Testing the Environment

```bash
# Run the test script
uv run python src/crowdcrossing/envs/test_collective_crossing.py

# Run the RGB rendering test
uv run python src/crowdcrossing/envs/test_rgb_rendering.py
```

### Code Quality

This project uses:
- **Ruff** for linting and formatting
- **Pre-commit** for automated code quality checks

#### Configuration Structure

The project uses a **separated configuration approach**:

- **`pyproject.toml`** - Project-specific settings (dependencies, metadata, build config)
- **`tool-config.toml`** - Reusable development tool configurations (ruff, pre-commit settings)

This separation allows you to easily copy `tool-config.toml` to other projects for consistent tooling.

#### Running Code Quality Tools

**Pre-commit hooks** (automatically configured):
```bash
# Pre-commit automatically uses tool-config.toml
# Just commit your changes - hooks will run automatically
git add .
git commit -m "your message"
```

**Manual tool execution**:
```bash
# Run linting with separate config
uv run ruff check . --config tool-config.toml

# Run formatting with separate config
uv run ruff format . --config tool-config.toml

# Run pre-commit manually on all files
uv run pre-commit run --all-files
```

### Adding dependencies

```bash
# Add a main dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

## License

[Add license information here]
