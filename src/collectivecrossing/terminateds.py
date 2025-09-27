"""Termination functions for the collective crossing environment."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from collectivecrossing.terminated_configs import TerminatedConfig

if TYPE_CHECKING:
    from collectivecrossing.collectivecrossing import CollectiveCrossingEnv


class TerminatedFunction(ABC):
    """Abstract base class for termination functions."""

    def __init__(self, terminated_config: TerminatedConfig):
        """Initialize the termination function with configuration."""
        self.terminated_config = terminated_config

    @abstractmethod
    def calculate_terminated(self, agent_id: str, env: "CollectiveCrossingEnv") -> bool | None:
        """
        Calculate termination status for an agent.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            True if the agent should be terminated, False otherwise.

        """
        pass


class AllAtDestinationTerminatedFunction(TerminatedFunction):
    """All-at-destination termination function implementation."""

    def calculate_terminated(self, agent_id: str, env: "CollectiveCrossingEnv") -> bool | None:
        """
        Calculate termination status for an agent.

        No agent is terminated until every agent is at its destination.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            True if all agents have reached their destinations, False otherwise.

        """
        # Check if all agents have reached their destinations
        for other_agent_id in env._agents.keys():
            if not env.has_agent_reached_destination(other_agent_id):
                return False
        return True


class IndividualAtDestinationTerminatedFunction(TerminatedFunction):
    """Individual-at-destination termination function implementation."""

    def calculate_terminated(self, agent_id: str, env: "CollectiveCrossingEnv") -> bool:
        """
        Calculate termination status for an agent.

        Each agent is terminated as soon as it reaches its destination.

        Args:
        ----
            agent_id: The ID of the agent.
            env: The environment instance.

        Returns:
        -------
            True if the agent has reached its destination, False otherwise.

        """
        return env.has_agent_reached_destination(agent_id)


# Registry of available termination functions
TERMINATED_FUNCTIONS: dict[str, type[TerminatedFunction]] = {
    "all_at_destination": AllAtDestinationTerminatedFunction,
    "individual_at_destination": IndividualAtDestinationTerminatedFunction,
}


def get_terminated_function(terminated_config: TerminatedConfig) -> TerminatedFunction:
    """
    Get a termination function by configuration.

    Args:
    ----
        terminated_config: The termination configuration.

    Returns:
    -------
        The termination function instance.

    Raises:
    ------
        ValueError: If the termination function name is not found.

    """
    name = terminated_config.get_terminated_function_name()
    if name not in TERMINATED_FUNCTIONS:
        available = ", ".join(TERMINATED_FUNCTIONS.keys())
        raise ValueError(f"Unknown termination function '{name}'. Available: {available}")

    return TERMINATED_FUNCTIONS[name](terminated_config)
