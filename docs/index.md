# 🚇 Collective Crossing

<p align="center">
  <img src="./images/collective_crossing_animation.gif" alt="Collective Crossing Animation" width="50%">
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

## 🚀 Quick Start

```python
from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig

# Create environment
config = CollectiveCrossingConfig(
    width=12, height=8, division_y=4,
    tram_door_x=6, tram_door_width=2, tram_length=10,
    num_boarding_agents=5, num_exiting_agents=3,
    max_steps=100, render_mode="rgb_array"
)

env = CollectiveCrossingEnv(config=config)
observations, infos = env.reset(seed=42)
```

## 🎮 Key Features

- **Multi-agent simulation** with boarding and exiting agents
- **Collision avoidance** prevents agents from overlapping
- **Configurable geometry** customizable tram and door positions
- **Ray RLlib compatible** uses MultiAgentEnv API
- **Multiple rendering modes** ASCII and RGB visualization
- **Type-safe configuration** using Pydantic v2

## 📚 Documentation

### Getting Started
- **[Installation Guide](installation.md)** - Complete setup instructions and troubleshooting
- **[Usage Guide](usage.md)** - How to use the environment and configuration examples

### Development
- **[Development Guide](development.md)** - Testing, contributing, and development guidelines
- **[Features Overview](features.md)** - Comprehensive feature descriptions

### Deployment
- **[Local Deployment](setup_local_deployment.md)** - Simple deployment guide

## 🛠️ Installation

```bash
# Clone and install
git clone https://github.com/nima-siboni/collectivecrossing.git
cd collectivecrossing
uv sync
```

See [Installation Guide](installation.md) for detailed instructions.

## 🚀 Quick Deploy

```bash
# Deploy documentation to GitHub Pages
./scripts/docs.sh deploy
```

See [Local Deployment Guide](setup_local_deployment.md) for details.

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

See [Development Guide](development.md) for detailed contribution guidelines.

## 📄 License

This project is licensed under the [Apache License 2.0](https://github.com/nima-siboni/collectivecrossing/blob/main/LICENSE).

---

**Happy simulating! 🚇✨**
