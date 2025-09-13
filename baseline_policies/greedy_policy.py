"""
Greedy baseline policy for the collective crossing environment.

This policy implements a simple greedy approach where:
- Boarding agents move directly toward the tram door and then to their destination
- Exiting agents move directly toward the tram door and then to their destination
- Agents don't consider other agents and move as directly as possible
"""

from typing import Any

import numpy as np

from collectivecrossing.actions import Actions
from collectivecrossing.types import AgentType


class GreedyPolicy:
    """Greedy baseline policy that moves agents directly toward their goals."""

    def __init__(self) -> None:
        """Initialize the greedy policy."""
        pass

    def get_action(self, agent_id: str, observation: np.ndarray, env: Any) -> int:
        """
        Get the greedy action for an agent based on its observation.

        Args:
        ----
            agent_id: The ID of the agent.
            observation: The observation array for the agent.
            env: The environment instance.

        Returns:
        -------
            The action to take (0-4: right, up, left, down, wait).

        """
        # Get agent information
        agent = env._get_agent(agent_id)
        agent_pos = agent.position
        agent_type = agent.agent_type

        # Get destination position
        destination = env.get_agent_destination_position(agent_id)

        # Calculate the best direction to move
        direction = self._calculate_direction(agent_pos, destination, agent_type, env)

        # Convert direction to action
        action = self._direction_to_action(direction)

        # Check if the action is valid (won't cause collision or go out of bounds)
        if self._is_valid_action(agent_id, action, env):
            return action
        else:
            # If the preferred action is invalid, try alternative actions
            return self._get_fallback_action(agent_id, agent_pos, destination, agent_type, env)

    def _calculate_direction(
        self, current_pos: np.ndarray, destination: np.ndarray, agent_type: AgentType, env: Any
    ) -> np.ndarray:
        """
        Calculate the best direction to move toward the destination.

        Args:
        ----
            current_pos: Current position of the agent.
            destination: Destination position.
            agent_type: Type of the agent (boarding or exiting).
            env: The environment instance.

        Returns:
        -------
            Direction vector [dx, dy] indicating the best move.

        """
        tram_door_pos = np.array([env.config.tram_door_x, env.config.division_y])

        # For boarding agents: go to tram door first, then to destination
        if agent_type == AgentType.BOARDING:
            # If not at tram door yet, go toward tram door
            if not np.array_equal(current_pos, tram_door_pos):
                return self._get_direction_vector(current_pos, tram_door_pos)
            else:
                # At tram door, go toward destination
                return self._get_direction_vector(current_pos, destination)

        # For exiting agents: go to tram door first, then to destination
        else:  # EXITING
            # If still in tram area, go toward tram door
            if current_pos[1] >= env.config.division_y:
                return self._get_direction_vector(current_pos, tram_door_pos)
            else:
                # Out of tram area, go toward destination
                return self._get_direction_vector(current_pos, destination)

    def _get_direction_vector(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Get the direction vector from current position to target position.

        Args:
        ----
            current_pos: Current position.
            target_pos: Target position.

        Returns:
        -------
            Direction vector [dx, dy] with values in {-1, 0, 1}.

        """
        diff = target_pos - current_pos

        # Normalize to get direction (only one step at a time)
        dx = 0
        dy = 0

        if diff[0] > 0:
            dx = 1
        elif diff[0] < 0:
            dx = -1

        if diff[1] > 0:
            dy = 1
        elif diff[1] < 0:
            dy = -1

        return np.array([dx, dy])

    def _direction_to_action(self, direction: np.ndarray) -> int:
        """
        Convert direction vector to action.

        Args:
        ----
            direction: Direction vector [dx, dy].

        Returns:
        -------
            Action code (0-4).

        """
        dx, dy = direction[0], direction[1]

        # Handle single-axis movements
        if dx == 1 and dy == 0:
            return Actions.right.value  # 0
        elif dx == 0 and dy == 1:
            return Actions.up.value  # 1
        elif dx == -1 and dy == 0:
            return Actions.left.value  # 2
        elif dx == 0 and dy == -1:
            return Actions.down.value  # 3

        # Handle diagonal movements by prioritizing the larger component
        elif dx != 0 and dy != 0:
            if abs(dx) > abs(dy):
                # Prioritize horizontal movement
                return Actions.right.value if dx > 0 else Actions.left.value
            else:
                # Prioritize vertical movement
                return Actions.up.value if dy > 0 else Actions.down.value

        # No movement
        else:
            return Actions.wait.value  # 4

    def _is_valid_action(self, agent_id: str, action: int, env: Any) -> bool:
        """
        Check if an action is valid (won't cause collision or go out of bounds).

        Args:
        ----
            agent_id: The ID of the agent.
            action: The action to check.
            env: The environment instance.

        Returns:
        -------
            True if the action is valid, False otherwise.

        """
        if action == Actions.wait.value:
            return True

        # Get current position
        current_pos = env._get_agent_position(agent_id)

        # Calculate new position
        direction = self._action_to_direction(action)
        new_pos = current_pos + direction

        # Use the environment's move validation logic
        return env._is_move_valid(agent_id, current_pos, new_pos)

    def _action_to_direction(self, action: int) -> np.ndarray:
        """Convert action to direction vector."""
        action_map = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
            Actions.wait.value: np.array([0, 0]),
        }
        return action_map[action]

    def _get_fallback_action(
        self,
        agent_id: str,
        current_pos: np.ndarray,
        destination: np.ndarray,
        agent_type: AgentType,
        env: Any,
    ) -> int:
        """
        Get a fallback action when the preferred action is invalid.

        Args:
        ----
            agent_id: The ID of the agent.
            current_pos: Current position of the agent.
            destination: Destination position.
            agent_type: Type of the agent.
            env: The environment instance.

        Returns:
        -------
            A valid action.

        """
        # Try actions in order of preference
        preferred_actions = self._get_preferred_actions(current_pos, destination, agent_type, env)

        for action in preferred_actions:
            if self._is_valid_action(agent_id, action, env):
                return action

        # If no action is valid, wait
        return Actions.wait.value

    def _get_preferred_actions(
        self, current_pos: np.ndarray, destination: np.ndarray, agent_type: AgentType, env: Any
    ) -> list[int]:
        """
        Get a list of preferred actions in order of preference.

        Args:
        ----
            current_pos: Current position.
            destination: Destination position.
            agent_type: Type of the agent.
            env: The environment instance.

        Returns:
        -------
            List of actions in order of preference.

        """
        # Calculate primary direction
        if agent_type == AgentType.BOARDING:
            # Boarding agents: go up first, then toward door
            if current_pos[1] < env.config.division_y:
                # Still in waiting area, prioritize going up
                if current_pos[0] < env.config.tram_door_x:
                    return [
                        Actions.right.value,
                        Actions.up.value,
                        Actions.left.value,
                        Actions.down.value,
                    ]
                elif current_pos[0] > env.config.tram_door_x:
                    return [
                        Actions.left.value,
                        Actions.up.value,
                        Actions.right.value,
                        Actions.down.value,
                    ]
                else:
                    return [
                        Actions.up.value,
                        Actions.right.value,
                        Actions.left.value,
                        Actions.down.value,
                    ]
            else:
                # In tram area, go toward destination
                if current_pos[0] < destination[0]:
                    return [
                        Actions.right.value,
                        Actions.up.value,
                        Actions.down.value,
                        Actions.left.value,
                    ]
                elif current_pos[0] > destination[0]:
                    return [
                        Actions.left.value,
                        Actions.up.value,
                        Actions.down.value,
                        Actions.right.value,
                    ]
                else:
                    return [
                        Actions.up.value,
                        Actions.right.value,
                        Actions.left.value,
                        Actions.down.value,
                    ]
        else:  # EXITING
            # Exiting agents: go down first, then toward door
            if current_pos[1] > env.config.division_y:
                # Still in tram area, prioritize going down
                if current_pos[0] < env.config.tram_door_x:
                    return [
                        Actions.right.value,
                        Actions.down.value,
                        Actions.left.value,
                        Actions.up.value,
                    ]
                elif current_pos[0] > env.config.tram_door_x:
                    return [
                        Actions.left.value,
                        Actions.down.value,
                        Actions.right.value,
                        Actions.up.value,
                    ]
                else:
                    return [
                        Actions.down.value,
                        Actions.right.value,
                        Actions.left.value,
                        Actions.up.value,
                    ]
            else:
                # In waiting area, go toward destination
                if current_pos[0] < destination[0]:
                    return [
                        Actions.right.value,
                        Actions.down.value,
                        Actions.up.value,
                        Actions.left.value,
                    ]
                elif current_pos[0] > destination[0]:
                    return [
                        Actions.left.value,
                        Actions.down.value,
                        Actions.up.value,
                        Actions.right.value,
                    ]
                else:
                    return [
                        Actions.down.value,
                        Actions.right.value,
                        Actions.left.value,
                        Actions.up.value,
                    ]


def create_greedy_policy() -> GreedyPolicy:
    """
    Create a greedy policy instance.

    Returns
    -------
        A GreedyPolicy instance.

    """
    return GreedyPolicy()
