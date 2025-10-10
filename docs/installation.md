# Installation Guide

## Prerequisites

- **Python 3.10+** 🐍
- **[uv](https://docs.astral.sh/uv/) package manager** ⚡

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd collectivecrossing
```

### 2. Install Dependencies

```bash
# Install main dependencies
uv sync

# Install development dependencies
uv sync --dev
```

#### Optional: Demo / Training Dependencies

The core package does not require RL training libraries. To run the RLlib demo and training examples, install the optional `demo` group (includes `ray[rllib]` and `torch`):

```bash
uv sync --group demo
```

### 3. Set Up Development Environment

```bash
# Set up pre-commit hooks
uv run pre-commit install
```

### 4. Verify Installation

```bash
# Run tests to verify everything works
uv run pytest

# Check code quality
uv run ruff check . --config tool-config.toml

# Run type checking
uv run mypy src/collectivecrossing/
```

## Alternative Installation Methods

### Using pip (not recommended)

```bash
pip install -e .
```

### Using conda

```bash
conda create -n collectivecrossing python=3.10
conda activate collectivecrossing
uv sync
```

## Testing the Installation

After installation, you can test that everything works correctly:

```python
# Test basic environment creation
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.reward_configs import DefaultRewardConfig
from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig
from collectivecrossing.observation_configs import DefaultObservationConfig

# Create a simple configuration with configurable systems
reward_config = DefaultRewardConfig(
    boarding_destination_reward=15.0,
    tram_door_reward=10.0,
    tram_area_reward=5.0,
    distance_penalty_factor=0.1
)

terminated_config = IndividualAtDestinationTerminatedConfig()
truncated_config = MaxStepsTruncatedConfig(max_steps=50)
observation_config = DefaultObservationConfig()

config = CollectiveCrossingConfig(
    width=10, height=8, division_y=4,
    tram_door_left=4, tram_door_right=6, tram_length=8,
    num_boarding_agents=3, num_exiting_agents=2,
    exiting_destination_area_y=0, boarding_destination_area_y=8,
    reward_config=reward_config,
    terminated_config=terminated_config,
    truncated_config=truncated_config,
    observation_config=observation_config
)

# Create and test environment
env = CollectiveCrossingEnv(config=config)
observations, infos = env.reset(seed=42)
print(f"Environment created successfully with {len(observations)} agents")
```

## Troubleshooting

### Common Issues

1. **uv not found**: Install uv from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
2. **Python version**: Ensure you have Python 3.10 or higher
3. **Permission errors**: Use `uv sync --user` or check your Python environment
4. **Import errors**: Make sure all dependencies are properly installed
5. **Test failures**: Run `uv run pytest -v` for detailed error information

### Getting Help

If you encounter issues during installation, please:
1. Check the [GitHub Issues](https://github.com/nima-siboni/collectivecrossing/issues)
2. Create a new issue with your error details
3. Include your Python version and operating system
4. Provide the complete error message and stack trace
