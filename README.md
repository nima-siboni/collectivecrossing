# Collective Crossing

A new project.

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
