"""Truncation function configurations for the collective crossing environment."""

from abc import abstractmethod
from typing import Any

from pydantic import Field

from collectivecrossing.utils.pydantic import ConfigClass


class TruncatedConfig(ConfigClass):
    """Base configuration class for truncation functions."""

    truncated_function: str = Field(
        description="Name of the truncation function to use",
        examples=["max_steps", "custom"],
    )

    @abstractmethod
    def get_truncated_function_name(self) -> str:
        """Get the name of the truncation function this config corresponds to."""
        return self.truncated_function


class MaxStepsTruncatedConfig(TruncatedConfig):
    """Configuration for the max-steps truncation function."""

    truncated_function: str = Field(
        default="max_steps",
        description="Episode is truncated when max_steps is reached",
    )

    max_steps: int = Field(
        default=1000,
        description="Maximum number of steps before truncation",
        ge=1,
        le=100000,
    )

    def get_truncated_function_name(self) -> str:
        """Get the name of the truncation function this config corresponds to."""
        return "max_steps"


class CustomTruncatedConfig(TruncatedConfig):
    """Configuration for custom truncation functions."""

    truncated_function: str = Field(description="Name of the custom truncation function")

    # Generic parameters that can be used by custom truncation functions
    max_steps: int = Field(
        default=1000,
        description="Maximum steps before truncation",
        ge=1,
        le=100000,
    )
    early_truncation_threshold: float = Field(
        default=0.0,
        description="Threshold for early truncation (0.0 = no early truncation)",
        ge=0.0,
        le=1.0,
    )
    require_all_agents_active: bool = Field(
        default=False,
        description="Whether to require all agents to be active for truncation",
    )

    def get_truncated_function_name(self) -> str:
        """Get the name of the truncation function this config corresponds to."""
        return self.truncated_function


# Registry of truncation configurations
TRUNCATED_CONFIGS = {
    "max_steps": MaxStepsTruncatedConfig,
    "custom": CustomTruncatedConfig,
}


def get_truncated_config(truncated_function_name: str, **kwargs: Any) -> TruncatedConfig:
    """
    Get a truncation configuration by name.

    Args:
    ----
        truncated_function_name: The name of the truncation function.
        **kwargs: Additional configuration parameters.

    Returns:
    -------
        The truncation configuration instance.

    Raises:
    ------
        ValueError: If the truncation function name is not found.

    """
    if truncated_function_name not in TRUNCATED_CONFIGS:
        available = ", ".join(TRUNCATED_CONFIGS.keys())
        raise ValueError(
            f"Unknown truncation function '{truncated_function_name}'. Available: {available}"
        )

    config_class = TRUNCATED_CONFIGS[truncated_function_name]

    # Remove truncated_function from kwargs if it exists to avoid duplicate parameter
    kwargs.pop("truncated_function", None)

    return config_class(truncated_function=truncated_function_name, **kwargs)
