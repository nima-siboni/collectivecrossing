# Development Guide

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/collectivecrossing/envs/test_collective_crossing.py

# Run with coverage
uv run pytest --cov=collectivecrossing

# Run with verbose output
uv run pytest -v

# Run tests in parallel
uv run pytest -n auto
```

### Test Structure

```
tests/
├── collectivecrossing/
│   └── envs/
│       ├── test_collective_crossing.py    # Main environment tests
│       ├── test_action_agent_validity.py  # Action validation tests
│       ├── test_dummy.py                  # Dummy environment tests
│       ├── test_rewards.py                # Reward function tests
│       ├── test_terminateds.py            # Termination function tests
│       ├── test_truncateds.py             # Truncation function tests
│       └── test_trajectory_vcr.py         # Trajectory tests
└── fixtures/
    └── trajectories/
        ├── current/                       # Current test data
        └── golden/                        # Golden test data
```

### Writing Tests

```python
import pytest
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.reward_configs import DefaultRewardConfig
from collectivecrossing.terminated_configs import IndividualAtDestinationTerminatedConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig
from collectivecrossing.observation_configs import DefaultObservationConfig

def test_basic_environment():
    # Create configuration with configurable systems
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
    
    env = CollectiveCrossingEnv(config=config)
    observations, infos = env.reset(seed=42)
    
    assert len(observations) == 5  # 3 boarding + 2 exiting agents
    assert not env.terminated
    assert not env.truncated
```

### Testing New Features

#### Testing Reward Functions

```python
from collectivecrossing.rewards import DefaultRewardFunction
from collectivecrossing.reward_configs import DefaultRewardConfig

def test_default_reward_function():
    config = DefaultRewardConfig(
        boarding_destination_reward=15.0,
        tram_door_reward=10.0,
        tram_area_reward=5.0,
        distance_penalty_factor=0.1
    )
    
    reward_func = DefaultRewardFunction(config)
    # Test reward computation
    reward = reward_func.compute_reward(agent_state, action, next_state)
    assert isinstance(reward, float)
```

#### Testing Termination Functions

```python
from collectivecrossing.terminateds import AllAtDestinationTerminatedFunction
from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig

def test_all_at_destination_termination():
    config = AllAtDestinationTerminatedConfig()
    terminated_func = AllAtDestinationTerminatedFunction(config)
    
    # Test termination logic
    terminated = terminated_func.check_termination(agent_states, episode_info)
    assert isinstance(terminated, bool)
```

#### Testing Truncation Functions

```python
from collectivecrossing.truncateds import MaxStepsTruncatedFunction
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

def test_max_steps_truncation():
    config = MaxStepsTruncatedConfig(max_steps=100)
    truncated_func = MaxStepsTruncatedFunction(config)
    
    # Test truncation logic
    truncated = truncated_func.calculate_truncated(agent_id, env)
    assert isinstance(truncated, bool)
```

#### Testing Observation Functions

```python
from collectivecrossing.observations import DefaultObservationFunction
from collectivecrossing.observation_configs import DefaultObservationConfig

def test_default_observation_function():
    config = DefaultObservationConfig()
    observation_func = DefaultObservationFunction(config)
    
    # Test observation generation
    observation = observation_func.get_agent_observation(agent_id, env)
    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.int32
```

### Trajectory Testing

The project includes comprehensive trajectory testing to ensure environment behavior consistency:

```bash
# Run trajectory tests
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py

# Update golden trajectories (if needed)
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py --update-golden
```

## Code Quality Tools

This project uses modern development tools:

- **🦀 Ruff** - Fast Python linter and formatter
- **🔒 Pre-commit** - Automated code quality checks
- **📋 Pytest** - Testing framework
- **🔍 Coverage** - Code coverage reporting
- **🔍 MyPy** - Static type checking

### Running Code Quality Tools

```bash
# Pre-commit hooks (run automatically on commit)
git add .
git commit -m "Your commit message"

# Manual linting
uv run ruff check . --config tool-config.toml

# Manual formatting
uv run ruff format . --config tool-config.toml

# Run pre-commit manually
uv run pre-commit run --all-files

# Type checking
uv run mypy src/collectivecrossing/
```

### Pre-commit Configuration

The project uses pre-commit hooks to ensure code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

## Project Structure

```
collectivecrossing/
├── 📁 src/collectivecrossing/
│   ├── 🎮 collectivecrossing.py      # Main environment implementation
│   ├── ⚙️ configs.py                 # Configuration classes with validation
│   ├── 🎯 actions.py                 # Action definitions and mappings
│   ├── 🏷️ types.py                   # Type definitions (AgentType, etc.)
│   ├── 🎁 reward_configs.py          # Reward function configurations
│   ├── 🎁 rewards.py                 # Reward function implementations
│   ├── ⏹️ terminated_configs.py      # Termination function configurations
│   ├── ⏹️ terminateds.py             # Termination function implementations
│   ├── ⏱️ truncated_configs.py       # Truncation function configurations
│   ├── ⏱️ truncateds.py              # Truncation function implementations
│   ├── 📁 utils/
│   │   ├── 📐 geometry.py            # Geometry utilities (TramBoundaries)
│   │   └── 🔧 pydantic.py            # Pydantic configuration utilities
│   └── 📁 tests/                     # Environment-specific tests
├── 📁 tests/                         # Main test suite
├── 📁 examples/                      # Usage examples
├── ⚙️ pyproject.toml                 # Project configuration
├── 🔧 tool-config.toml               # Development tools configuration
└── 📋 uv.lock                        # Dependency lock file
```

## Adding Dependencies

```bash
# Add main dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add dependency with specific version
uv add "package-name>=1.0.0,<2.0.0"

# Remove dependency
uv remove package-name
```

## Building and Publishing

```bash
# Build the package
uv run build

# Check the built package
uv run twine check dist/*

# Upload to PyPI (if you have access)
uv run twine upload dist/*
```

## Contributing

### Development Workflow

1. **Fork the repository** 🍴
2. **Create a feature branch** 🌿
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** ✏️
4. **Run tests** 🧪
   ```bash
   uv run pytest
   uv run ruff check . --config tool-config.toml
   ```
5. **Commit your changes** 💾
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```
6. **Push to your fork** 📤
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Submit a pull request** 🔄

### Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Commit Message Format

Use conventional commit messages:

```
feat: add new reward function configuration
fix: resolve termination logic bug
docs: update usage examples
test: add tests for truncation functions
refactor: improve configuration validation
```

### Adding New Features

When adding new features, follow these guidelines:

1. **Configuration First** - Add configuration classes for new features
2. **Type Safety** - Use Pydantic for configuration validation
3. **Testing** - Write comprehensive tests for new functionality
4. **Documentation** - Update documentation with examples
5. **Backward Compatibility** - Maintain compatibility with existing APIs

### Adding New Reward Functions

```python
# 1. Create reward configuration
class CustomRewardConfig(RewardConfig):
    custom_parameter: float = Field(default=1.0, description="Custom parameter")

# 2. Create reward function
class CustomRewardFunction:
    def __init__(self, config: CustomRewardConfig):
        self.config = config
    
    def compute_reward(self, agent_state, action, next_state):
        # Implement reward logic
        return reward_value

# 3. Add to registry
REWARD_CONFIGS["custom"] = CustomRewardConfig
REWARD_FUNCTIONS["custom"] = CustomRewardFunction
```

### Adding New Termination Functions

```python
# 1. Create termination configuration
class CustomTerminatedConfig(TerminatedConfig):
    custom_parameter: bool = Field(default=True, description="Custom parameter")

# 2. Create termination function
class CustomTerminatedFunction:
    def __init__(self, config: CustomTerminatedConfig):
        self.config = config
    
    def check_termination(self, agent_states, episode_info):
        # Implement termination logic
        return terminated

# 3. Add to registry
TERMINATED_CONFIGS["custom"] = CustomTerminatedConfig
TERMINATED_FUNCTIONS["custom"] = CustomTerminatedFunction
```

### Adding New Truncation Functions

```python
# 1. Create truncation configuration
class CustomTruncatedConfig(TruncatedConfig):
    custom_parameter: int = Field(default=100, description="Custom parameter")

# 2. Create truncation function
class CustomTruncatedFunction:
    def __init__(self, config: CustomTruncatedConfig):
        self.config = config
    
    def check_truncation(self, step_count, episode_info):
        # Implement truncation logic
        return truncated

# 3. Add to registry
TRUNCATED_CONFIGS["custom"] = CustomTruncatedConfig
TRUNCATED_FUNCTIONS["custom"] = CustomTruncatedFunction
```

## Troubleshooting

### Common Issues

1. **Test failures** - Check if golden trajectories need updating
2. **Configuration errors** - Verify Pydantic validation rules
3. **Import errors** - Ensure all dependencies are installed
4. **Type checking errors** - Add proper type hints

### Getting Help

- Check existing issues on GitHub
- Create a new issue with detailed error information
- Include your Python version and operating system
- Provide minimal reproduction examples
