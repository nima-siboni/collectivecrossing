"""Termination function configurations for the collective crossing environment."""

from abc import abstractmethod
from typing import Any

from pydantic import Field

from collectivecrossing.utils.pydantic import ConfigClass


class TerminatedConfig(ConfigClass):
    """Base configuration class for termination functions."""

    terminated_function: str = Field(
        description="Name of the termination function to use",
        examples=["all_at_destination", "individual_at_destination", "custom"],
    )

    @abstractmethod
    def get_terminated_function_name(self) -> str:
        """Get the name of the termination function this config corresponds to."""
        return self.terminated_function


class AllAtDestinationTerminatedConfig(TerminatedConfig):
    """Configuration for the all-at-destination termination function."""

    terminated_function: str = Field(
        default="all_at_destination",
        description="All agents must reach destination before any termination",
    )

    def get_terminated_function_name(self) -> str:
        """Get the name of the termination function this config corresponds to."""
        return "all_at_destination"


class IndividualAtDestinationTerminatedConfig(TerminatedConfig):
    """Configuration for the individual-at-destination termination function."""

    terminated_function: str = Field(
        default="individual_at_destination",
        description="Each agent terminates when it reaches its destination",
    )

    def get_terminated_function_name(self) -> str:
        """Get the name of the termination function this config corresponds to."""
        return "individual_at_destination"


class CustomTerminatedConfig(TerminatedConfig):
    """Configuration for custom termination functions."""

    terminated_function: str = Field(description="Name of the custom termination function")

    # Generic parameters that can be used by custom termination functions
    max_steps_per_agent: int = Field(
        default=1000,
        description="Maximum steps before individual agent termination",
        ge=1,
        le=10000,
    )
    require_all_completion: bool = Field(
        default=False,
        description="Whether all agents must complete before any termination",
    )
    timeout_penalty: bool = Field(
        default=False,
        description="Whether to apply timeout penalty for incomplete agents",
    )

    def get_terminated_function_name(self) -> str:
        """Get the name of the termination function this config corresponds to."""
        return self.terminated_function


# Registry of termination configurations
TERMINATED_CONFIGS = {
    "all_at_destination": AllAtDestinationTerminatedConfig,
    "individual_at_destination": IndividualAtDestinationTerminatedConfig,
    "custom": CustomTerminatedConfig,
}


def get_terminated_config(terminated_function_name: str, **kwargs: Any) -> TerminatedConfig:
    """
    Get a termination configuration by name.

    Args:
    ----
        terminated_function_name: The name of the termination function.
        **kwargs: Additional configuration parameters.

    Returns:
    -------
        The termination configuration instance.

    Raises:
    ------
        ValueError: If the termination function name is not found.

    """
    if terminated_function_name not in TERMINATED_CONFIGS:
        available = ", ".join(TERMINATED_CONFIGS.keys())
        raise ValueError(
            f"Unknown termination function '{terminated_function_name}'. Available: {available}"
        )

    config_class = TERMINATED_CONFIGS[terminated_function_name]

    # Remove terminated_function from kwargs if it exists to avoid duplicate parameter
    kwargs.pop("terminated_function", None)

    return config_class(terminated_function=terminated_function_name, **kwargs)
