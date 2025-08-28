# 🚇 Collective Crossing

<p align="center">
  <img src="docs/images/collective_crossing_animation.gif" alt="Collective Crossing Animation" width="50%">
</p>

[![Tests](https://github.com/nima-siboni/collectivecrossing/workflows/Run%20Tests/badge.svg)](https://github.com/nima-siboni/collectivecrossing/actions)
[![Coverage](https://img.shields.io/badge/coverage-76%25-green)](https://github.com/nima-siboni/collectivecrossing/actions/workflows/test.yml)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/nima-siboni/collectivecrossing)

A multi-agent reinforcement learning environment for simulating collective behavior in tram boarding/exiting scenarios. This project provides a grid-world environment where multiple agents interact to achieve their goals while sharing some resources together.

## 🎯 Overview

The `CollectiveCrossingEnv` simulates a minimal tram boarding scenario where coordination is essential to find the optimal collective behavior:
- **Boarding agents** 🚶‍♂️ start in the platform area and navigate to the tram door
- **Exiting agents** 🚶‍♀️ start inside the tram and navigate to the exit
- **Simple collision avoidance** 🛡️ prevents agents from occupying the same space, which makes the passing through the tram door a bottleneck and a challenge
- **Configurable geometry** 🏗️ allows customization of tram size, door position, and environment
- **Flexible reward system** 🎁 supports multiple reward strategies (default, simple distance, binary)
- **Customizable termination** ⏹️ configurable episode termination conditions
- **Adaptive truncation** ⏱️ flexible episode truncation policies

## 🚀 Quick Start

```python
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.reward_configs import DefaultRewardConfig
from collectivecrossing.terminated_configs import AllAtDestinationTerminatedConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig

# Create environment with configurable systems
reward_config = DefaultRewardConfig(
    boarding_destination_reward=15.0,
    tram_door_reward=10.0,
    tram_area_reward=5.0,
    distance_penalty_factor=0.1
)

terminated_config = AllAtDestinationTerminatedConfig()
truncated_config = MaxStepsTruncatedConfig(max_steps=100)

config = CollectiveCrossingConfig(
    width=12, height=8, division_y=4,
    tram_door_x=6, tram_door_width=2, tram_length=10,
    num_boarding_agents=5, num_exiting_agents=3,
    render_mode="rgb_array",
    reward_config=reward_config,
    terminated_config=terminated_config,
    truncated_config=truncated_config
)

env = CollectiveCrossingEnv(config=config)
observations, infos = env.reset(seed=42)
```

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Usage Guide](docs/usage.md)** - Complete usage examples and configuration
- **[Development Guide](docs/development.md)** - Testing, contributing, and development
- **[Features Overview](docs/features.md)** - Comprehensive feature descriptions
- **[Local Deployment](docs/setup_local_deployment.md)** - Simple deployment guide

## 🎮 Key Features

- **Multi-agent simulation** with boarding and exiting agents
- **Collision avoidance** prevents agents from overlapping
- **Configurable geometry** customizable tram and door positions
- **Ray RLlib compatible** uses MultiAgentEnv API
- **Multiple rendering modes** ASCII and RGB visualization
- **Type-safe configuration** using Pydantic v2
- **Flexible reward system** multiple reward strategies with custom configurations
- **Customizable termination** configurable episode ending conditions
- **Adaptive truncation** flexible episode timeout policies

## 🛠️ Installation

```bash
# Clone and install
git clone <repository-url>
cd collectivecrossing
uv sync
```

See [Installation Guide](docs/installation.md) for detailed instructions.

## 🚀 Quick Deploy

```bash
# Deploy documentation to GitHub Pages
./scripts/docs.sh deploy
```

See [Local Deployment Guide](docs/setup_local_deployment.md) for details.

## 🧪 Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=collectivecrossing
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

See [Development Guide](docs/development.md) for detailed contribution guidelines.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

**Happy simulating! 🚇✨**
