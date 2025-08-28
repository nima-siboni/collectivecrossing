"""Truncation functions for the collective crossing environment."""

from abc import ABC, abstractmethod

from collectivecrossing.truncated_configs import TruncatedConfig


class TruncatedFunction(ABC):
    """Base class for truncation functions."""

    def __init__(self, truncated_config: TruncatedConfig):
        """Initialize the truncation function with configuration."""
        self.truncated_config = truncated_config

    @abstractmethod
    def is_truncated(self, current_step_count: int) -> bool:
        """
        Determine if the episode should be truncated.

        Args:
        ----
            current_step_count: The current step count.

        Returns:
        -------
            True if the episode should be truncated, False otherwise.

        """
        pass


class MaxStepsTruncatedFunction(TruncatedFunction):
    """Truncation function that truncates when max_steps is reached."""

    def is_truncated(self, current_step_count: int) -> bool:
        """
        Return True if the episode is truncated, False otherwise.

        The logic is that the episode is truncated if the step count is greater than or equal to
        the max steps.

        Args:
        ----
            current_step_count: The current step count.

        Returns:
        -------
            True if the episode is truncated, False otherwise.

        """
        return current_step_count >= self.truncated_config.max_steps


class CustomTruncatedFunction(TruncatedFunction):
    """Custom truncation function that can be extended with custom logic."""

    def is_truncated(self, current_step_count: int) -> bool:
        """
        Return truncated.

        Args:
        ----
            current_step_count: The current step count.

        Returns:
        -------
            True if the episode should be truncated, False otherwise.

        """
        # Basic max steps logic
        if current_step_count >= self.truncated_config.max_steps:
            return True

        # Add custom logic here if needed
        # For example, early truncation based on threshold
        # if self.truncated_config.early_truncation_threshold > 0:
        #     # Custom early truncation logic
        #     pass

        return False


# Registry of truncation functions
TRUNCATED_FUNCTIONS = {
    "max_steps": MaxStepsTruncatedFunction,
    "custom": CustomTruncatedFunction,
}


def get_truncated_function(truncated_config: TruncatedConfig) -> TruncatedFunction:
    """
    Get a truncation function by configuration.

    Args:
    ----
        truncated_config: The truncation configuration.

    Returns:
    -------
        The truncation function instance.

    Raises:
    ------
        ValueError: If the truncation function name is not found.

    """
    name = truncated_config.get_truncated_function_name()

    if name not in TRUNCATED_FUNCTIONS:
        available = ", ".join(TRUNCATED_FUNCTIONS.keys())
        raise ValueError(f"Unknown truncation function '{name}'. Available: {available}")

    return TRUNCATED_FUNCTIONS[name](truncated_config)
