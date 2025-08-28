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
│       ├── test_trajectory_vcr.py         # Trajectory tests
│       └── test_truncateds.py             # Truncation tests
└── fixtures/
    └── trajectories/
        └── golden/                        # Golden test data
```

### Writing Tests

```python
import pytest
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig

def test_basic_environment():
    config = CollectiveCrossingConfig(
        width=10, height=8, division_y=4,
        tram_door_x=5, tram_door_width=2, tram_length=8,
        num_boarding_agents=3, num_exiting_agents=2,
        max_steps=50
    )
    
    env = CollectiveCrossingEnv(config=config)
    observations, infos = env.reset(seed=42)
    
    assert len(observations) == 5  # 3 boarding + 2 exiting agents
    assert not env.terminated
    assert not env.truncated
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
│   ├── 📁 utils/
│   │   ├── 📐 geometry.py            # Geometry utilities (TramBoundaries)
│   │   └── 🔧 pydantic.py            # Pydantic configuration utilities
│   ├── 📁 wrappers/
│   │   ├── 🎁 clip_reward.py         # Reward clipping wrapper
│   │   ├── 🎲 discrete_actions.py    # Discrete action space wrapper
│   │   ├── ⚖️ reacher_weighted_reward.py  # Weighted reward wrapper
│   │   └── 📍 relative_position.py   # Relative positioning wrapper
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
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(env): add new rendering mode`
- `fix(config): validate tram door position`
- `docs(readme): update installation instructions`
- `test(env): add collision detection tests`

### Pull Request Guidelines

- Include a clear description of the changes
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass
- Follow the existing code style

## Debugging

### Common Issues

1. **Import errors**: Make sure you're in the correct Python environment
2. **Configuration errors**: Check parameter validation messages
3. **Test failures**: Run tests with `-v` flag for verbose output

### Debug Tools

```bash
# Run with debug logging
uv run python -m pytest --log-cli-level=DEBUG

# Use pdb for debugging
uv run python -m pdb -m pytest test_file.py::test_function

# Profile code performance
uv run python -m cProfile -o profile.stats your_script.py
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Your code here
env = CollectiveCrossingEnv(config)
for _ in range(1000):
    env.step(actions)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Usage

```python
import tracemalloc

tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```
