"""
Multi-agent environment for collective crossing simulation.

This module provides a Gymnasium environment that simulates a collective crossing
scenario where agents need to navigate through a tram area with specific rules
and constraints.
"""

import logging

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from collectivecrossing.actions import ACTION_TO_DIRECTION
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.observations import get_observation_function
from collectivecrossing.rewards import get_reward_function
from collectivecrossing.terminateds import get_terminated_function
from collectivecrossing.truncateds import get_truncated_function
from collectivecrossing.types import Agent, AgentType
from collectivecrossing.utils.geometry import TramBoundaries, calculate_tram_boundaries

# Set up logger
logger = logging.getLogger(__name__)


class CollectiveCrossingEnv(MultiAgentEnv):
    """
    Multi-agent environment simulating collective crossing scenario.

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
        """
        Initialize the collective crossing environment.

        Args:
        ----
            config: Configuration object containing environment parameters.

        """
        self._config = config

        # Calculate tram boundaries using the dataclass
        self._tram_boundaries = calculate_tram_boundaries(self.config)

        # Agent tracking - unified structure
        self._agents: dict[str, Agent] = {}  # agent_id -> Agent
        self._step_count = 0

        # Action mapping
        self._action_to_direction = ACTION_TO_DIRECTION

        # Initialize observation function
        self._observation_function = get_observation_function(self.config.observation_config)

        # Initialize reward function
        self._reward_function = get_reward_function(self.config.reward_config)

        # Initialize termination function
        self._terminated_function = get_terminated_function(self.config.terminated_config)

        # Initialize truncation function
        self._truncated_function = get_truncated_function(self.config.truncated_config)

        self._agents = self._create_dummy_agents()

        # Define observation and action spaces
        self._setup_spaces()

        super().__init__()

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
                if (
                    self._is_valid_position(pos)
                    and not self._is_position_occupied(pos)
                    and not (
                        self.tram_door_left <= pos[0] <= self.tram_door_right
                        and pos[1] == self.config.division_y - 1
                    )
                ):
                    agent = Agent(
                        id=f"boarding_{boarding_agent_counter}",
                        agent_type=AgentType.BOARDING,
                        position=pos,
                        active=True,
                        terminated=False,
                        truncated=False,
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
                if self._is_valid_position(pos) and not self._is_position_occupied(pos):
                    agent = Agent(
                        id=f"exiting_{exiting_agent_counter}",
                        agent_type=AgentType.EXITING,
                        position=pos,
                        active=True,
                        terminated=False,
                        truncated=False,
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

    def step(
        self, action_dict: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        """
        Execute one step in the environment.

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
        self._agents_truncated_or_terminated_this_step = set()
        # Process actions for all agents for which there is an action in the action_dict
        for agent_id, action in action_dict.items():
            # check the validity of the action and the agent
            # this function will raise an error if the agent is not active or the action is not
            # valid
            self._check_action_and_agent_validity(agent_id, action)
            self._move_agent(agent_id, action)

        # Calculate rewards, check termination, and update observations for all agents
        # Note: we need to iterate over the agents in the environment, not the action_dict
        # because RLlib allows to return obs, rewards, terminateds, truncateds, infos for any
        # agent in the environment.

        # Deactivate active agents that have reached their destination.
        for agent_id in self._agents.keys():
            if self._agents[agent_id].active and self.has_agent_reached_destination(agent_id):
                self._agents[agent_id].deactivate()

        for agent_id in self._agents.keys():
            reward: float | None = self._calculate_reward(agent_id)
            if reward is not None:
                rewards[agent_id] = reward

        for agent_id in self._agents.keys():
            terminated: bool | None = self._calculate_terminated(agent_id)
            if terminated is not None:
                terminateds[agent_id] = terminated

        for agent_id in self._agents.keys():
            truncated: bool | None = self._calculate_truncated(agent_id)
            if truncated is not None:
                truncateds[agent_id] = truncated

        for agent_id in self._agents.keys():
            terminated = terminateds.get(agent_id, None)
            if terminated is not None:
                if terminated and not self._agents[agent_id].terminated:
                    self._agents[agent_id].terminate()
                    self._agents_truncated_or_terminated_this_step.add(agent_id)

        for agent_id in self._agents.keys():
            truncated = truncateds.get(agent_id, None)
            if truncated is not None:
                if truncated and not self._agents[agent_id].truncated:
                    self._agents[agent_id].truncate()
                    self._agents_truncated_or_terminated_this_step.add(agent_id)

        for agent_id in set(self.agents) | self._agents_truncated_or_terminated_this_step:
            # Note that we only return observations for agents which are not terminated or
            # truncated, hence we iterate over self.agents instead of self._agents.keys().
            observations[agent_id] = self._get_agent_observation(agent_id)

            infos[agent_id] = {
                "agent_type": self._agents[agent_id].agent_type.value,
                "in_tram_area": self.is_in_tram_area(agent_id),
                "at_door": self.is_at_tram_door(agent_id),
                "active": self._agents[agent_id].active,
                "at_destination": self.has_agent_reached_destination(agent_id),
            }
        # Check if environment is done
        all_terminated = all(terminateds.values()) if terminateds else False
        all_truncated = all(truncateds.values()) if truncateds else False
        terminateds["__all__"] = all_terminated
        truncateds["__all__"] = all_truncated

        return observations, rewards, terminateds, truncateds, infos

    def get_observation_space(self, agent_id: str) -> gym.Space:
        """Get observation space for a specific agent."""
        return self.observation_space

    def get_action_space(self, agent_id: str) -> gym.Space:
        """Get action space for a specific agent."""
        return self.action_space

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """
        Render the environment.

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

    def _create_dummy_agents(self) -> dict[str, Agent]:
        """
        Create dummy agents. Later, this is used to set up the observation and action spaces.

        This function is exactly like reset, but without the position. The ids and types are
        correct.

        TODO: this is a very dirty hack, bc it has common code with reset. We should refactor this.


        Returns
        -------
            A dictionary of dummy agents. Dummy agents are agents with correct ids and types,
            but with positions are set to None.

        """
        agents: dict[str, Agent] = {}
        for boarding_agent_counter in range(self.config.num_boarding_agents):
            pos = np.array([None, None])
            agent = Agent(
                id=f"boarding_{boarding_agent_counter}",
                agent_type=AgentType.BOARDING,
                position=pos,
                active=True,
                terminated=False,
                truncated=False,
            )
            agents[agent.id] = agent

        # Initialize exiting agents (start in upper part, tram area)
        for exiting_agent_counter in range(self.config.num_exiting_agents):
            pos = np.array([None, None])
            agent = Agent(
                id=f"exiting_{exiting_agent_counter}",
                agent_type=AgentType.EXITING,
                position=pos,
                active=True,
                terminated=False,
                truncated=False,
            )
            agents[agent.id] = agent

        return agents

    def _is_move_valid(self, agent_id: str, current_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """
        Check if the move is valid.

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
            and not self._would_hit_tram_wall(current_pos, new_pos)
        )

    def _calculate_new_position(self, agent_id: str, action: int) -> np.ndarray:
        """Calculate the new position of the agent."""
        direction = self._action_to_direction[action]
        current_pos = self._get_agent_position(agent_id)
        new_pos = current_pos + direction
        return new_pos

    def _move_agent(self, agent_id: str, action: int) -> None:
        """
        Move the agent based on the action only if the agent is active and the move is valid.

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
    def action_spaces(self) -> gym.Space:
        """Get the action space for all agents."""
        return self._action_spaces

    @property
    def observation_spaces(self) -> gym.Space:
        """Get the observation space for all agents."""
        return self._observation_spaces

    def _setup_spaces(self) -> None:
        """Set up observation and action spaces for all agents."""
        # All agents have the same action space (5 actions including wait)
        self._action_spaces = {agent_id: spaces.Discrete(5) for agent_id in self._agents.keys()}

        # RLlib/Gymnasium expect prototype single-agent spaces on the env
        # Use the common action space for all agents as the prototype
        any_agent_id = next(iter(self._agents.keys())) if self._agents else None
        if any_agent_id is not None:
            self.action_space = self._action_spaces[any_agent_id]

        self._observation_spaces = {}
        # Observation space includes agent position, tram info, and other agents
        # For simplicity, we'll use a flattened representation
        for agent_id in self._agents.keys():
            self._observation_spaces[agent_id] = (
                self._observation_function.return_agent_observation_space(agent_id, self)
            )

        # Pick any agent's observation space as the prototype observation space
        # This is required by Gymnasium's PassiveEnvChecker
        any_agent_id = next(iter(self._agents.keys())) if self._agents else None
        if any_agent_id is not None:
            self.observation_space = self._observation_spaces[any_agent_id]

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for a specific agent."""
        return self._observation_function.get_agent_observation(agent_id, self)

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
        """Check if a position is within the grid bounds and not on a wall."""
        # Check grid bounds
        # Allow y=height if boarding destination equals height
        x = pos[0]
        y = pos[1]
        if not (0 <= x <= self.config.width and 0 <= y <= self.config.height):
            return False

        # Check if position is on a wall
        # Check division line wall (except door area)
        if y == self.config.division_y:
            # If at division line, check if it's in the door area
            # Make door boundaries exclusive - agents cannot occupy
            # tram_door_left or tram_door_right
            if not (self.tram_door_left < x < self.tram_door_right):
                return False  # On division line wall, not in door area

        # Check tram side walls
        if y >= self.config.division_y:
            # In tram area, check if position is outside tram boundaries
            # Note: the agent cannot be >at< the tram left or right boundary
            if not (self.tram_right > x > self.tram_left):
                return False  # Outside tram boundaries

        return True

    def _is_position_occupied(self, pos: np.ndarray, exclude_agent: str | None = None) -> bool:
        """Check if a position is occupied by another active agent."""
        for agent_id, agent in self._agents.items():
            if agent_id != exclude_agent and agent.active and np.array_equal(agent.position, pos):
                return True
        return False

    def is_in_boarding_destination_area(self, agent_id: str) -> bool:
        """Check if a position is in the boarding destination area."""
        return self._get_agent_position(agent_id)[1] == self.config.boarding_destination_area_y

    def is_in_exiting_destination_area(self, agent_id: str) -> bool:
        """Check if a position is in the exiting destination area."""
        return self._get_agent_position(agent_id)[1] == self.config.exiting_destination_area_y

    def is_in_tram_area(self, agent_id: str) -> bool:
        """Check if a position is in the tram area (upper part within tram boundaries)."""
        pos = self._get_agent_position(agent_id)
        return pos[1] >= self.config.division_y and self.tram_left <= pos[0] <= self.tram_right

    def is_at_tram_door(self, agent_id: str) -> bool:
        """Check if a position is at the tram door (adjacent to the occupied door positions)."""
        pos = self._get_agent_position(agent_id)
        # Agent is at door if they're at the division line and adjacent
        # to the occupied door positions
        return pos[1] == self.config.division_y and (
            pos[0] == self.tram_door_left - 1 or pos[0] == self.tram_door_right + 1
        )

    def _would_hit_tram_wall(self, current_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """Check if a move would hit a tram wall."""
        x_new = new_pos[0]
        y_new = new_pos[1]
        # old position components are not needed for current collision checks
        # Check if moving across the division line (y = division_y)
        if y_new == self.config.division_y:
            # Block movement through the division line EXCEPT at door positions
            # Check if the agent is trying to cross through the door area
            # Make door boundaries exclusive - agents cannot occupy tram_door_left
            # or tram_door_right
            if self.tram_door_left < x_new < self.tram_door_right:
                return False  # Allow movement through door area
            else:
                return True  # Block movement through wall (non-door area)

        # Check if moving across tram side walls (x = tram_left or x = tram_right)
        # Block any movement that goes outside the tram area
        # Left wall crossing - block if moving from inside to outside
        if y_new > self.config.division_y:
            if x_new == self.tram_left or x_new == self.tram_right:
                return True

        return False

    def _calculate_reward(self, agent_id: str) -> float | None:
        """
        Calculate reward for an agent using the configured reward function.

        Args:
        ----
            agent_id: The ID of the agent.

        Returns:
        -------
            The reward for the agent.

        """
        return self._reward_function.calculate_reward(agent_id, self)

    def _calculate_terminated(self, agent_id: str) -> bool | None:
        """
        Calculate termination status for an agent using the configured termination function.

        Args:
        ----
            agent_id: The ID of the agent.

        Returns:
        -------
            True if the agent should be terminated, False otherwise.

        """
        return self._terminated_function.calculate_terminated(agent_id, self)

    def _calculate_truncated(self, agent_id: str) -> bool | None:
        """
        Calculate truncation status for an agent using the configured truncation function.

        Args:
        ----
            agent_id: The ID of the agent.

        Returns:
        -------
            True if the episode should be truncated, False otherwise.

        """
        return self._truncated_function.calculate_truncated(agent_id, self)

    def get_agent_destination_position(self, agent_id: str) -> tuple[int | None, int | None]:
        """
        Get the destination position for an agent.

        That the destination position can have two values:
          - either it is a tuple of two integers (x, y)
          - or it is a tuple of one integer and None (x, None), or (None, y)

        This formulation is used to allow the destination be a seating area or
        exit area, as well as one particular position.

        Args:
        ----
            agent_id: The ID of the agent.

        Returns:
        -------
            The destination position as a numpy array.

        """
        agent_type = self._agents[agent_id].agent_type
        if agent_type == AgentType.BOARDING:
            # Boarding agents go to the boarding destination area
            return (None, self.config.boarding_destination_area_y)
        else:  # EXITING
            # Exiting agents go to the exiting destination area
            return (None, self.config.exiting_destination_area_y)

    def has_agent_reached_destination(self, agent_id: str) -> bool:
        """
        Check if an agent has reached its destination.

        Args:
        ----
            agent_id: The ID of the agent.

        Returns:
        -------
            True if the agent has reached its destination, otherwise False.

        """
        agent_type = self._agents[agent_id].agent_type

        if agent_type == AgentType.BOARDING:
            # Boarding agents are done when they reach the boarding destination area
            return self.is_in_boarding_destination_area(agent_id)
        else:  # EXITING
            # Exiting agents are done when they reach the exiting destination area
            return self.is_in_exiting_destination_area(agent_id)

    def _check_action_and_agent_validity(self, agent_id: str, action: int) -> None:
        """
        Check if the action and agent are valid.

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
        # check if the action is valid
        if action not in self._action_to_direction:
            raise ValueError(
                f"Invalid action: {action} for agent {agent_id}. Valid actions are: "
                f"{list(self._action_to_direction.keys())}"
            )

    def _render_matplotlib(self) -> np.ndarray:
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
        fig.tight_layout()

        # Render to buffer
        canvas.draw()
        width, height = canvas.get_width_height()
        buf = canvas.buffer_rgba()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        return arr[..., :3]  # RGB

    def _draw_matplotlib(self, ax: plt.Axes) -> None:
        """Draw the environment using matplotlib."""
        from collectivecrossing.rendering import draw_matplotlib

        draw_matplotlib(self, ax)

    @property
    def agents(self) -> list[str]:
        """
        Get the agents ids.

        Notes
        -----
        * RLlib expects to have a list of agent ids present at any point in time as an attribute
        of the environment.

        * We keep track of **all** the agents in the _agents attribute, not only agents which
        are present at each timestep. Hence we need to update this attribute at the end of each
        step.

        Returns
        -------
            The list of agent ids which are not terminatd or truncated at any point in time.

        """
        # Find all the agents which are not terminated or truncated.
        agents_ids = [
            agent_id
            for agent_id in self._agents.keys()
            if not self._agents[agent_id].is_terminated and not self._agents[agent_id].is_truncated
        ]
        return agents_ids

    @property
    def possible_agents(self) -> list[str]:
        """
        Get all the possible agents ids (see definition below).

        possible agents refer to agent IDs which might even show up in your episodes, and RLlib
        requires setting the self.possible_agents attribute to a list of all possible agent IDs.

        Returns
        -------
            The list of all the possible agents ids.

        """
        return list(self._agents.keys())
