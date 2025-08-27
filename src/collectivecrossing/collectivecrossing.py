"""Multi-agent environment for collective crossing simulation.

This module provides a Gymnasium environment that simulates a collective crossing
scenario where agents need to navigate through a tram area with specific rules
and constraints.
"""

import logging
from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from collectivecrossing.actions import ACTION_TO_DIRECTION
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.types import Agent, AgentType
from collectivecrossing.utils.geometry import TramBoundaries, calculate_tram_boundaries
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Set up logger
logger = logging.getLogger(__name__)


class CollectiveCrossingEnv(MultiAgentEnv):
    """Multi-agent environment simulating collective crossing scenario.

    Geometry:
    - Rectangular domain divided by a horizontal line
    - Upper part: Tram area (configurable length)
    - Lower part: Waiting area for people to board
    - Configurable tram door position and width
    - Tram width equals the size of the upper division
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        config: CollectiveCrossingConfig,
    ):
        """Initialize the collective crossing environment.

        Args:
        ----
            config: Configuration object containing environment parameters.

        """
        super().__init__()

        self._config = config

        # Calculate tram boundaries using the dataclass
        self._tram_boundaries = calculate_tram_boundaries(self.config)

        # Agent tracking - unified structure
        self._agents: dict[str, Agent] = {}  # agent_id -> Agent
        self._step_count = 0

        # Action mapping
        self._action_to_direction = ACTION_TO_DIRECTION

        # Define observation and action spaces
        self._setup_spaces()

        # Rendering
        self._window = None
        self._clock = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self._step_count = 0
        self._agents = {}

        # Initialize boarding agents (start in lower part, away from door)
        for boarding_agent_counter in range(self.config.num_boarding_agents):
            while True:
                pos = np.array(
                    [
                        self.np_random.integers(0, self.config.width),
                        self.np_random.integers(0, self.config.division_y),  # Lower part
                    ]
                )
                # Avoid positions directly under the door initially
                if not self._is_position_occupied(pos) and not (
                    self.tram_door_left <= pos[0] <= self.tram_door_right
                    and pos[1] == self.config.division_y - 1
                ):
                    agent = Agent(
                        id=f"boarding_{boarding_agent_counter}",
                        agent_type=AgentType.BOARDING,
                        position=pos,
                        active=True,
                    )
                    self._agents[agent.id] = agent
                    break

        # Initialize exiting agents (start in upper part, tram area)
        for exiting_agent_counter in range(self.config.num_exiting_agents):
            while True:
                pos = np.array(
                    [
                        self.np_random.integers(self.tram_left, self.tram_right + 1),
                        self.np_random.integers(
                            self.config.division_y, self.config.height
                        ),  # Upper part
                    ]
                )
                if not self._is_position_occupied(pos):
                    agent = Agent(
                        id=f"exiting_{exiting_agent_counter}",
                        agent_type=AgentType.EXITING,
                        position=pos,
                        active=True,
                    )
                    self._agents[agent.id] = agent
                    break

        # Get observations for all agents
        observations = {}
        infos = {}
        for agent in self._agents.values():
            observations[agent.id] = self._get_agent_observation(agent.id)
            infos[agent.id] = {"agent_type": agent.agent_type.value}

        return observations, infos

    def step(  # type: ignore[override]
        self, action_dict: dict[str, int]
    ) -> tuple[dict[Any, Any], dict[Any, Any], dict[Any, Any], dict[Any, Any], dict[Any, Any]]: 
        """Execute one step in the environment.

        Notes:
        -----
        - Process actions for all agents for which there is an action in the action_dict
        - Calculate rewards, check termination, and update observations for **all agents**:
        we iterate over the all agents in the environment (not only the agents for which there is
        an action in the action_dict), because RLlib allows to return obs, rewards, terminateds,
        truncateds, infos for any agent in the environment.
        See [here](https://docs.ray.io/en/releases-2.48.0/rllib/multi-agent-envs.html)
        for more details in RLlib documentation.

        Args:
        ----
            action_dict: A dictionary of agent IDs and actions.

        Returns:
        -------
            A tuple of observations, rewards, terminateds, truncateds, and infos.

        """
        self._step_count += 1

        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        # Process actions for all agents for which there is an action in the action_dict
        for agent_id, action in action_dict.items():
            # check the validity of the action and the agent
            # this function will raise an error if the agent is not active or the action is not
            # valid
            self._check_action_and_agent_validity(agent_id, action)

            # Move agent
            self._move_agent(agent_id, action)

        # Calculate rewards, check termination, and update observations for all agents
        # Note: we need to iterate over the agents in the environment, not the action_dict
        # because RLlib allows to return obs, rewards, terminateds, truncateds, infos for any
        # agent in the environment.

        # TODO: remove this later?
        for agent_id in self._agents.keys():
            if self._agents[agent_id].active and self._is_agent_terminated(agent_id):
                self._agents[agent_id].deactivate()

        for agent_id in self._agents.keys():
            if self._agents[agent_id].active:
                # Calculate reward
                reward = self._calculate_reward(agent_id)
                rewards[agent_id] = reward

                # Check if agent is done
                terminateds[agent_id] = self._is_agent_terminated(agent_id)

                truncateds[agent_id] = self._is_truncated(self._step_count, self.config.max_steps)
                infos[agent_id] = {
                    "agent_type": self._agents[agent_id].agent_type.value,
                    "in_tram_area": self._is_in_tram_area(self._get_agent_position(agent_id)),
                    "at_door": self._is_at_tram_door(self._get_agent_position(agent_id)),
                }
                observations[agent_id] = self._get_agent_observation(agent_id)

        # Check if environment is done
        all_terminated = all(terminateds.values()) if terminateds else False

        terminateds["__all__"] = all_terminated
        truncateds["__all__"] = self._is_truncated(self._step_count, self.config.max_steps)

        return observations, rewards, terminateds, truncateds, infos

    def get_observation_space(self, agent_id: str) -> gym.Space:
        """Get observation space for a specific agent."""
        return self.observation_space

    def get_action_space(self, agent_id: str) -> gym.Space:
        """Get action space for a specific agent."""
        return self.action_space

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:  # type: ignore[override]
        """Render the environment.

        Args:
        ----
            mode: Rendering mode. Supports "rgb_array" and "human".

        Returns:
        -------
            Rendered image as numpy array for "rgb_array" mode, None for "human" mode.

        Raises:
        ------
            NotImplementedError: If the rendering mode is not supported.

        """
        if mode == "rgb_array":
            return self._render_matplotlib()
        elif mode == "human":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            self._draw_matplotlib(ax)
            fig.tight_layout()
            plt.show()
            return None
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def _is_move_valid(self, agent_id: str, current_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """Check if the move is valid.

        A move is valid if:
        - the new position is valid
        - the new position is not occupied by an **active** agent
        - the new position does not cross the tram wall

        Args:
        ----
            agent_id: The ID of the agent.
            current_pos: The current position of the agent.
            new_pos: The new position of the agent.

        Returns:
        -------
            True if the move is valid, False otherwise.

        """
        return (
            self._is_valid_position(new_pos)
            and not self._is_position_occupied(new_pos, exclude_agent=agent_id)
            and not self._would_cross_tram_wall(current_pos, new_pos)
        )

    def _calculate_new_position(self, agent_id: str, action: int) -> np.ndarray:
        """Calculate the new position of the agent."""
        direction = self._action_to_direction[action]
        current_pos = self._get_agent_position(agent_id)
        new_pos = current_pos + direction
        return new_pos

    def _move_agent(self, agent_id: str, action: int) -> None:
        """Move the agent based on the action only if the agent is active and the move is valid.

        Note:
        ----
        - This function has a side effect: the agent is moved to the new position,
        i.e. _boarding_agents or _exiting_agents is updated.

        Args:
        ----
            agent_id: The ID of the agent.
            action: The action to move the agent.

        Returns:
        -------
            None

        """
        # do nothing if the agent is not active
        if not self._agents[agent_id].active:
            return
        # update the position of the agent
        current_pos = self._get_agent_position(agent_id)
        # Calculate new position
        new_pos = self._calculate_new_position(agent_id, action)

        # Check if move is valid
        if self._is_move_valid(agent_id, current_pos, new_pos):
            # Update position
            self._agents[agent_id].update_position(new_pos)

    @staticmethod
    def _is_truncated(current_step_count: int, max_steps: int) -> bool:
        """Return True if the episode is truncated, False otherwise.

        The logic is that the episode is truncated if the step count is greater than the max steps.

        Args:
        ----
            current_step_count: The current step count.
            max_steps: The maximum number of steps.

        Returns:
        -------
            True if the episode is truncated, False otherwise.

        """
        return current_step_count >= max_steps

    def close(self) -> None:
        """Close the environment."""
        pass

    @property
    def config(self) -> CollectiveCrossingConfig:
        """Get the environment configuration."""
        return self._config

    @property
    def tram_boundaries(self) -> TramBoundaries:
        """Get the tram boundaries."""
        return self._tram_boundaries

    @property
    def tram_door_left(self) -> int:
        """Get the left boundary of the tram door."""
        return self._tram_boundaries.tram_door_left

    @property
    def tram_door_right(self) -> int:
        """Get the right boundary of the tram door."""
        return self._tram_boundaries.tram_door_right

    @property
    def tram_left(self) -> int:
        """Get the left boundary of the tram area."""
        return self._tram_boundaries.tram_left

    @property
    def tram_right(self) -> int:
        """Get the right boundary of the tram area."""
        return self._tram_boundaries.tram_right

    @property
    def action_space(self) -> gym.Space:  # type: ignore[override]
        """Get the action space for all agents."""
        return self._action_space

    @property
    def observation_space(self) -> gym.Space:  # type: ignore[override]
        """Get the observation space for all agents."""
        return self._observation_space

    def _setup_spaces(self) -> None:
        """Set up observation and action spaces for all agents."""
        # All agents have the same action space (5 actions including wait)
        self._action_space = spaces.Discrete(5)

        # Observation space includes agent position, tram info, and other agents
        # For simplicity, we'll use a flattened representation
        obs_size = 2 + 6 + len(self._agents) * 2  # agent_pos + tram_info + all_other_agents
        self._observation_space = spaces.Box(
            low=0,
            high=max(self.config.width, self.config.height) - 1,
            shape=(obs_size,),
            dtype=np.int32,
        )

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for a specific agent."""
        agent_pos = self._get_agent_position(agent_id)

        # Start with agent's own position and tram door information
        tram_door_info = np.array(
            [
                self.config.tram_door_x,  # Door center X
                self.config.division_y,  # Division line Y
                self.tram_door_left,  # Door left boundary
                self.tram_door_right,  # Door right boundary
            ]
        )
        # TODO: add active status of the agent
        obs = np.concatenate([agent_pos, tram_door_info])

        # Add positions of all other agents
        # TODO: is this for all agents or only the active ones?
        for other_id in self._agents.keys():
            if other_id != agent_id:
                other_pos = self._get_agent_position(other_id)
                obs = np.concatenate([obs, other_pos])
            else:
                # Use a placeholder for self (will be masked out)
                obs = np.concatenate([obs, np.array([-1, -1])])

        return obs.astype(np.int32)

    def _get_agent_position(self, agent_id: str) -> np.ndarray:
        """Get current position of an agent."""
        if agent_id in self._agents:
            return self._agents[agent_id].position
        else:
            raise ValueError(f"Unknown agent ID: {agent_id}")

    def _get_agent(self, agent_id: str) -> Agent:
        """Get the Agent object for a given agent ID."""
        if agent_id in self._agents:
            return self._agents[agent_id]
        else:
            raise ValueError(f"Unknown agent ID: {agent_id}")

    def _get_agents_by_type(self, agent_type: AgentType) -> list[Agent]:
        """Get all agents of a specific type."""
        return [agent for agent in self._agents.values() if agent.agent_type == agent_type]

    def _get_boarding_agents(self) -> list[Agent]:
        """Get all boarding agents."""
        return self._get_agents_by_type(AgentType.BOARDING)

    def _get_exiting_agents(self) -> list[Agent]:
        """Get all exiting agents."""
        return self._get_agents_by_type(AgentType.EXITING)

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within the grid bounds."""
        return 0 <= pos[0] < self.config.width and 0 <= pos[1] < self.config.height

    def _is_position_occupied(self, pos: np.ndarray, exclude_agent: str | None = None) -> bool:
        """Check if a position is occupied by another active agent."""
        for agent_id, agent in self._agents.items():
            if agent_id != exclude_agent and agent.active and np.array_equal(agent.position, pos):
                return True
        return False

    def _is_in_tram_area(self, pos: np.ndarray) -> bool:
        """Check if a position is in the tram area (upper part within tram boundaries)."""
        return pos[1] >= self.config.division_y and self.tram_left <= pos[0] <= self.tram_right

    def _is_at_tram_door(self, pos: np.ndarray) -> bool:
        """Check if a position is at the tram door."""
        return (
            pos[1] == self.config.division_y
            and self.tram_door_left <= pos[0] <= self.tram_door_right
        )

    def _would_cross_tram_wall(self, current_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """Check if a move would cross a tram wall."""
        # Check if moving across the division line (y = division_y)
        if (current_pos[1] < self.config.division_y and new_pos[1] >= self.config.division_y) or (
            current_pos[1] >= self.config.division_y and new_pos[1] < self.config.division_y
        ):
            # Only allow crossing at the door
            if not (self.tram_door_left <= new_pos[0] <= self.tram_door_right):
                return True

        # Check if moving across tram side walls (x = tram_left or x = tram_right)
        # Only check if the agent is in the tram area (y >= division_y)
        if current_pos[1] >= self.config.division_y or new_pos[1] >= self.config.division_y:
            # Check left wall crossing (from inside tram to outside)
            if self.tram_left <= current_pos[0] <= self.tram_right and new_pos[0] < self.tram_left:
                return True
            # Check right wall crossing (from inside tram to outside)
            if self.tram_left <= current_pos[0] <= self.tram_right and new_pos[0] > self.tram_right:
                return True

        return False

    def _is_in_exiting_destination_area(self, pos: np.ndarray) -> bool:
        """Check if a position is in the exiting destination area."""
        return pos[1] == self.config.exiting_destination_area_y

    def _is_in_boarding_destination_area(self, pos: np.ndarray) -> bool:
        """Check if a position is in the boarding destination area."""
        return pos[1] == self.config.boarding_destination_area_y

    def _calculate_reward(self, agent_id: str) -> float:
        """Calculate reward for an agent.

        The reward is calculated based on the agent's position and type:
        - Boarding agents get positive reward for reaching tram door and boarding destination area
        - Exiting agents get positive reward for reaching exiting destination area
        - Boarding agents get negative reward for moving towards the door
        - Exiting agents get negative reward for moving towards the exit

        Args:
        ----
            agent_id: The ID of the agent.

        Returns:
        -------
            The reward for the agent.

        """
        agent_pos = self._get_agent_position(agent_id)
        agent_type = self._agents[agent_id].agent_type

        if agent_type == AgentType.BOARDING:
            # Boarding agents get positive reward for reaching tram door and boarding destination
            # area
            if self._is_in_boarding_destination_area(agent_pos):
                return 15.0  # Successfully reached boarding destination area
            elif self._is_at_tram_door(agent_pos):
                return 10.0  # Successfully reached tram door
            elif self._is_in_tram_area(agent_pos):
                return 5.0  # Good progress - in tram area
            else:
                # Small reward for moving towards the door
                distance_to_door = abs(agent_pos[0] - self.config.tram_door_x) + (
                    self.config.division_y - agent_pos[1]
                )
                return -distance_to_door * 0.1
        else:  # EXITING
            # Exiting agents get positive reward for reaching exiting destination area
            if self._is_in_exiting_destination_area(agent_pos):
                return 10.0  # Successfully reached exiting destination area
            elif not self._is_in_tram_area(agent_pos):
                return 5.0  # Good progress - exited tram area
            else:
                # Small reward for moving towards exit
                distance_to_exit = abs(agent_pos[0] - self.config.tram_door_x) + (
                    agent_pos[1] - self.config.division_y
                )
                return distance_to_exit * 0.1

    def _is_agent_terminated(self, agent_id: str) -> bool:
        """Check if an agent has completed its goal.

        Args:
        ----
            agent_id: The ID of the agent.

        Returns:
        -------
            True if the agent is terminated, otherwise False.

        """
        agent_pos = self._get_agent_position(agent_id)
        agent_type = self._agents[agent_id].agent_type

        if agent_type == AgentType.BOARDING:
            # Boarding agents are done when they reach the boarding destination area
            return self._is_in_boarding_destination_area(agent_pos)
        else:  # EXITING
            # Exiting agents are done when they reach the exiting destination area
            return self._is_in_exiting_destination_area(agent_pos)

    def _check_action_and_agent_validity(self, agent_id: str, action: int) -> None:
        """Check if the action and agent are valid.

        Args:
        ----
            agent_id: The ID of the agent.
            action: The action to check.

        Raises:
        ------
            ValueError: If the agent ID is not among the agents, the action is not in the
            _action_to_direction, or the agent is not active in the environment.

        """
        # check if the agent is among the agents
        if agent_id not in self._agents.keys():
            raise ValueError(
                f"Unknown agent ID: {agent_id} in action_dict. The action_dict keys must be a "
                f"subset of the agents. Current agents: {self._agents.keys()}"
            )
        # check if the agent is active in the environment
        if not self._agents[agent_id].active:
            raise ValueError(f"Agent {agent_id} exists but it is not active in the environment.")
        # check if the action is valid
        if action not in self._action_to_direction:
            raise ValueError(
                f"Invalid action: {action} for agent {agent_id}. Valid actions are: "
                f"{list(self._action_to_direction.keys())}"
            )

    def _render_matplotlib(self):  # type: ignore[no-untyped-def]
        """Return an RGB array via Agg without touching pyplot (safe for animations)."""
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        # Make a Figure that is NOT connected to any GUI backend
        fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvas(fig)  # Agg canvas
        ax = fig.add_subplot(1, 1, 1)

        # Draw everything
        self._draw_matplotlib(ax)

        # Avoid pyplot tight_layout; use OO API:
        fig.set_tight_layout(True)

        # Render to buffer
        canvas.draw()
        width, height = canvas.get_width_height()
        buf = canvas.buffer_rgba()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        return arr[..., :3]  # RGB

    def _draw_matplotlib(self, ax):  # type: ignore[no-untyped-def]
        """Draw the environment using matplotlib."""
        from collectivecrossing.rendering import draw_matplotlib

        draw_matplotlib(self, ax)
