import gymnasium as gym
import numpy as np
from collectivecrossing.actions import ACTION_TO_DIRECTION
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.types import AgentType
from collectivecrossing.utils.geometry import TramBoundaries, calculate_tram_boundaries
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


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
        super().__init__()

        self._config = config

        # Calculate tram boundaries using the dataclass
        self._tram_boundaries = calculate_tram_boundaries(self.config)

        # Agent tracking
        self._boarding_agents = {}  # agent_id -> position
        self._exiting_agents = {}  # agent_id -> position
        self._agent_types = {}  # agent_id -> AgentType
        self._step_count = 0

        # Action mapping
        self._action_to_direction = ACTION_TO_DIRECTION

        # Initialize agent IDs
        self._init_agent_ids()

        # Define observation and action spaces
        self._setup_spaces()

        # Rendering
        self._window = None
        self._clock = None

    @property
    def config(self):
        return self._config

    @property
    def tram_boundaries(self) -> TramBoundaries:
        return self._tram_boundaries

    @property
    def tram_door_left(self) -> int:
        return self._tram_boundaries.tram_door_left

    @property
    def tram_door_right(self) -> int:
        return self._tram_boundaries.tram_door_right

    @property
    def tram_left(self) -> int:
        return self._tram_boundaries.tram_left

    @property
    def tram_right(self) -> int:
        return self._tram_boundaries.tram_right

    @property
    def all_agent_ids(self):
        return self._boarding_agent_ids + self._exiting_agent_ids

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def boarding_agent_ids(self):
        return self._boarding_agent_ids

    @property
    def exiting_agent_ids(self):
        return self._exiting_agent_ids

    def _init_agent_ids(self):
        """Initialize agent IDs for boarding and exiting agents"""
        self._boarding_agent_ids = [f"boarding_{i}" for i in range(self.config.num_boarding_agents)]
        self._exiting_agent_ids = [f"exiting_{i}" for i in range(self.config.num_exiting_agents)]

        # Set agent types
        for agent_id in self._boarding_agent_ids:
            self._agent_types[agent_id] = AgentType.BOARDING
        for agent_id in self._exiting_agent_ids:
            self._agent_types[agent_id] = AgentType.EXITING

    def _setup_spaces(self):
        """Setup observation and action spaces for all agents"""
        # All agents have the same action space (5 actions including wait)
        self._action_space = spaces.Discrete(5)

        # Observation space includes agent position, tram info, and other agents
        # For simplicity, we'll use a flattened representation
        obs_size = 2 + 6 + len(self.all_agent_ids) * 2  # agent_pos + tram_info + all_other_agents
        self._observation_space = spaces.Box(
            low=0,
            high=max(self.config.width, self.config.height) - 1,
            shape=(obs_size,),
            dtype=np.int32,
        )

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for a specific agent"""
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
        obs = np.concatenate([agent_pos, tram_door_info])

        # Add positions of all other agents
        for other_id in self.all_agent_ids:
            if other_id != agent_id:
                other_pos = self._get_agent_position(other_id)
                obs = np.concatenate([obs, other_pos])
            else:
                # Use a placeholder for self (will be masked out)
                obs = np.concatenate([obs, np.array([-1, -1])])

        return obs.astype(np.int32)

    def _get_agent_position(self, agent_id: str) -> np.ndarray:
        """Get current position of an agent"""
        if agent_id in self._boarding_agents:
            return self._boarding_agents[agent_id]
        elif agent_id in self._exiting_agents:
            return self._exiting_agents[agent_id]
        else:
            raise ValueError(f"Unknown agent ID: {agent_id}")

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within the grid bounds"""
        return 0 <= pos[0] < self.config.width and 0 <= pos[1] < self.config.height

    def _is_position_occupied(self, pos: np.ndarray, exclude_agent: str = None) -> bool:
        """Check if a position is occupied by another agent"""
        for agent_id, agent_pos in self._boarding_agents.items():
            if agent_id != exclude_agent and np.array_equal(agent_pos, pos):
                return True
        for agent_id, agent_pos in self._exiting_agents.items():
            if agent_id != exclude_agent and np.array_equal(agent_pos, pos):
                return True
        return False

    def _is_in_tram_area(self, pos: np.ndarray) -> bool:
        """Check if a position is in the tram area (upper part within tram boundaries)"""
        return pos[1] >= self.config.division_y and self.tram_left <= pos[0] <= self.tram_right

    def _is_at_tram_door(self, pos: np.ndarray) -> bool:
        """Check if a position is at the tram door"""
        return (
            pos[1] == self.config.division_y
            and self.tram_door_left <= pos[0] <= self.tram_door_right
        )

    def _would_cross_tram_wall(self, current_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """Check if a move would cross a tram wall"""
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
        """Check if a position is in the exiting destination area"""
        return pos[1] == self.config.exiting_destination_area_y

    def _is_in_boarding_destination_area(self, pos: np.ndarray) -> bool:
        """Check if a position is in the boarding destination area"""
        return pos[1] == self.config.boarding_destination_area_y

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        self._step_count = 0
        self._boarding_agents = {}
        self._exiting_agents = {}

        # Initialize boarding agents (start in lower part, away from door)
        for _, agent_id in enumerate(self.boarding_agent_ids):
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
                    self._boarding_agents[agent_id] = pos
                    break

        # Initialize exiting agents (start in upper part, tram area)
        for _, agent_id in enumerate(self.exiting_agent_ids):
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
                    self._exiting_agents[agent_id] = pos
                    break

        # Get observations for all agents
        observations = {}
        infos = {}
        for agent_id in self.all_agent_ids:
            observations[agent_id] = self._get_agent_observation(agent_id)
            infos[agent_id] = {"agent_type": self._agent_types[agent_id].value}

        self._agents_to_remove = []
        return observations, infos

    def step(
        self, action_dict: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        """Execute one step in the environment"""
        self._step_count += 1
        max_steps_reached = self._step_count >= self.config.max_steps

        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for agent_id in self._agents_to_remove:
            action_dict.pop(agent_id)
            if agent_id in self._boarding_agents:
                del self._boarding_agents[agent_id]
                self._boarding_agent_ids.remove(agent_id)
            elif agent_id in self._exiting_agents:
                del self._exiting_agents[agent_id]
                self._exiting_agent_ids.remove(agent_id)
        self._agents_to_remove = []

        # Process actions for all agents
        for agent_id, action in action_dict.items():
            if agent_id not in self.all_agent_ids:
                raise ValueError(
                    f"Unknown agent ID: {agent_id} in action_dict. The action_dict keys must be a subset of the all_agent_ids. Current all_agent_ids: {self.all_agent_ids}"
                )
            if action not in self._action_to_direction:
                raise ValueError(
                    f"Invalid action: {action} for agent {agent_id}. Valid actions are: {list(self._action_to_direction.keys())}"
                )

            # Get current position
            current_pos = self._get_agent_position(agent_id)

            # Calculate new position
            direction = self._action_to_direction[action]
            new_pos = current_pos + direction

            # Check if move is valid
            if (
                self._is_valid_position(new_pos)
                and not self._is_position_occupied(new_pos, exclude_agent=agent_id)
                and not self._would_cross_tram_wall(current_pos, new_pos)
            ):
                # Update position
                if agent_id in self._boarding_agents:
                    self._boarding_agents[agent_id] = new_pos
                else:
                    self._exiting_agents[agent_id] = new_pos

            # Calculate reward
            reward = self._calculate_reward(agent_id)
            rewards[agent_id] = reward

            # Check if agent is done
            agent_terminated = self._is_agent_done(agent_id)
            terminateds[agent_id] = agent_terminated
            truncateds[agent_id] = max_steps_reached
            infos[agent_id] = {
                "agent_type": self._agent_types[agent_id].value,
                "in_tram_area": self._is_in_tram_area(self._get_agent_position(agent_id)),
                "at_door": self._is_at_tram_door(self._get_agent_position(agent_id)),
            }
            observations[agent_id] = self._get_agent_observation(agent_id)

            # Remove terminated agents from the environment (after calculating rewards and checking termination)
            if agent_terminated:
                # add this agent to the list of agents which needs to be removed in the next step
                self._agents_to_remove.append(agent_id)

        # Check if environment is done
        all_terminated = all(terminateds.values()) if terminateds else False

        terminateds["__all__"] = all_terminated
        truncateds["__all__"] = max_steps_reached

        return observations, rewards, terminateds, truncateds, infos

    def _calculate_reward(self, agent_id: str) -> float:
        """Calculate reward for an agent"""
        agent_pos = self._get_agent_position(agent_id)
        agent_type = self._agent_types[agent_id]

        if agent_type == AgentType.BOARDING:
            # Boarding agents get positive reward for reaching tram door and boarding destination area
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

    def _is_agent_done(self, agent_id: str) -> bool:
        """Check if an agent has completed its goal"""
        agent_pos = self._get_agent_position(agent_id)
        agent_type = self._agent_types[agent_id]

        if agent_type == AgentType.BOARDING:
            # Boarding agents are done when they reach the boarding destination area
            return self._is_in_boarding_destination_area(agent_pos)
        else:  # EXITING
            # Exiting agents are done when they reach the exiting destination area
            return self._is_in_exiting_destination_area(agent_pos)

    def get_observation_space(self, agent_id: str) -> gym.Space:
        """Get observation space for a specific agent"""
        return self.observation_space

    def get_action_space(self, agent_id: str) -> gym.Space:
        """Get action space for a specific agent"""
        return self.action_space

    def render(self, mode="rgb_array"):
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

    def _render_matplotlib(self):
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

    def _draw_matplotlib(self, ax):
        import matplotlib.patches as patches

        colors = {
            "background": "#f8f9fa",
            "tram_area": "#e3f2fd",
            "waiting_area": "#fff3e0",
            "exiting_destination_area": "#2196f3",  # Light red, similar to boarding agents (#f44336)
            "boarding_destination_area": "#f44336",  # Light blue, similar to exiting agents (#2196f3)
            "tram_wall": "#424242",
            "door": "#4caf50",
            "boarding_agent": "#f44336",
            "exiting_agent": "#2196f3",
        }

        ax.set_facecolor(colors["background"])

        # ----- your original drawing code (rects, circles, texts, legend) -----
        # tram area
        ax.add_patch(
            patches.Rectangle(
                (self.tram_left, self.config.division_y),
                self.tram_right - self.tram_left + 1,
                self.config.height - self.config.division_y,
                facecolor=colors["tram_area"],
                edgecolor="none",
                alpha=0.7,
            )
        )

        # waiting area
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                self.config.width,
                self.config.division_y,
                facecolor=colors["waiting_area"],
                edgecolor="none",
                alpha=0.7,
            )
        )

        # exiting destination area
        if self.config.exiting_destination_area_y < self.config.division_y:
            ax.add_patch(
                patches.Rectangle(
                    (0, self.config.exiting_destination_area_y),
                    self.config.width,
                    1,
                    facecolor=colors["exiting_destination_area"],
                    edgecolor="none",
                    alpha=0.8,
                )
            )

        # boarding destination area (limited to tram geometry)
        if self.config.boarding_destination_area_y >= self.config.division_y:
            ax.add_patch(
                patches.Rectangle(
                    (self.tram_left, self.config.boarding_destination_area_y),
                    self.tram_right - self.tram_left + 1,
                    1,
                    facecolor=colors["boarding_destination_area"],
                    edgecolor="none",
                    alpha=0.8,
                )
            )

        wall_thickness = 0.1
        if self.tram_door_left > self.tram_left:
            ax.add_patch(
                patches.Rectangle(
                    (self.tram_left, self.config.division_y - wall_thickness / 2),
                    self.tram_door_left - self.tram_left,
                    wall_thickness,
                    facecolor=colors["tram_wall"],
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.9,
                )
            )
        if self.tram_door_right < self.tram_right:
            ax.add_patch(
                patches.Rectangle(
                    (self.tram_door_right + 1, self.config.division_y - wall_thickness / 2),
                    self.tram_right - self.tram_door_right,
                    wall_thickness,
                    facecolor=colors["tram_wall"],
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.9,
                )
            )

        for y in range(self.config.division_y, self.config.height):
            ax.add_patch(
                patches.Rectangle(
                    (self.tram_left - wall_thickness / 2, y),
                    wall_thickness,
                    1,
                    facecolor=colors["tram_wall"],
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.9,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (self.tram_right + 1 - wall_thickness / 2, y),
                    wall_thickness,
                    1,
                    facecolor=colors["tram_wall"],
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.9,
                )
            )

        ax.add_patch(
            patches.Rectangle(
                (self.tram_door_left, self.config.division_y),
                self.tram_door_right - self.tram_door_left + 1,
                1,
                facecolor=colors["door"],
                edgecolor="darkgreen",
                linewidth=2,
                alpha=0.8,
            )
        )

        for agent_id, (x, y) in self._boarding_agents.items():
            for r, a in [(0.4, 0.3), (0.3, 0.5), (0.2, 0.8)]:
                ax.add_patch(
                    patches.Circle(
                        (x + 0.5, y + 0.5),
                        r,
                        facecolor=colors["boarding_agent"],
                        edgecolor="darkred",
                        linewidth=1,
                        alpha=a,
                    )
                )
            # defensive split
            label = agent_id.split("_", 1)[-1] if "_" in agent_id else agent_id
            ax.text(
                x + 0.5,
                y + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
            )

        for agent_id, (x, y) in self._exiting_agents.items():
            for r, a in [(0.4, 0.3), (0.3, 0.5), (0.2, 0.8)]:
                ax.add_patch(
                    patches.Circle(
                        (x + 0.5, y + 0.5),
                        r,
                        facecolor=colors["exiting_agent"],
                        edgecolor="darkblue",
                        linewidth=1,
                        alpha=a,
                    )
                )
            label = agent_id.split("_", 1)[-1] if "_" in agent_id else agent_id
            ax.text(
                x + 0.5,
                y + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
            )

        ax.set_xlim(0, self.config.width)
        ax.set_ylim(0, self.config.height)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_title("Collective Crossing Environment", fontsize=14, weight="bold", pad=20)
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#e3f2fd", alpha=0.7, label="Tram Area"),
            Patch(facecolor="#fff3e0", alpha=0.7, label="Waiting Area"),
            Patch(facecolor="#ffcdd2", alpha=0.8, label="Exiting Destination Area"),
            Patch(facecolor="#ffeb3b", alpha=0.8, label="Boarding Destination Area"),
            Patch(facecolor="#4caf50", alpha=0.8, label="Tram Door"),
            patches.Circle((0, 0), 0.1, facecolor="#f44336", label="Boarding Agents"),
            patches.Circle((0, 0), 0.1, facecolor="#2196f3", label="Exiting Agents"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    def close(self):
        """Close the environment"""
        pass
