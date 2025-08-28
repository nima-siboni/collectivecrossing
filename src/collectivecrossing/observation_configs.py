"""Observation function configurations for the collective crossing environment."""

from abc import abstractmethod
from typing import Any

from pydantic import Field

from collectivecrossing.utils.pydantic import ConfigClass


class ObservationConfig(ConfigClass):
    """Base configuration class for observation functions."""

    observation_function: str = Field(
        description="Name of the observation function to use",
        examples=["default"],
    )

    @abstractmethod
    def get_observation_function_name(self) -> str:
        """Get the name of the observation function this config corresponds to."""
        return self.observation_function


class DefaultObservationConfig(ObservationConfig):
    """Configuration for the default observation function."""

    observation_function: str = Field(
        default="default",
        description="Default observation function providing agent position, tram info,"
        "and position of other agents",
    )

    def get_observation_function_name(self) -> str:
        """Get the name of the observation function this config corresponds to."""
        return "default"


# Registry of observation configurations
OBSERVATION_CONFIGS = {
    "default": DefaultObservationConfig,
}


def get_observation_config(observation_function_name: str, **kwargs: Any) -> ObservationConfig:
    """
    Get an observation configuration by name.

    Args:
    ----
        observation_function_name: The name of the observation function.
        **kwargs: Additional configuration parameters.

    Returns:
    -------
        The observation configuration instance.

    Raises:
    ------
        ValueError: If the observation function name is not found.

    """
    if observation_function_name not in OBSERVATION_CONFIGS:
        available = ", ".join(OBSERVATION_CONFIGS.keys())
        raise ValueError(
            f"Unknown observation function '{observation_function_name}'. Available: {available}"
        )

    config_class = OBSERVATION_CONFIGS[observation_function_name]

    # Remove observation_function from kwargs if it exists to avoid duplicate parameter
    kwargs.pop("observation_function", None)

    return config_class(observation_function=observation_function_name, **kwargs)
