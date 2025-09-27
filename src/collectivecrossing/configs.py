"""Configuration classes for the collective crossing environment."""

from pydantic import Field, model_validator

from collectivecrossing.observation_configs import DefaultObservationConfig, ObservationConfig
from collectivecrossing.reward_configs import DefaultRewardConfig, RewardConfig
from collectivecrossing.terminated_configs import (
    IndividualAtDestinationTerminatedConfig,
    TerminatedConfig,
)
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig, TruncatedConfig
from collectivecrossing.utils.pydantic import ConfigClass


class CollectiveCrossingConfig(ConfigClass):
    """
    A class that configures the CollectiveCrossing environment.

    Attributes
    ----------
        width: Width of the environment
        height: Height of the environment
        division_y: Y-coordinate of the horizontal division line
        tram_door_left: Left boundary of tram door
        tram_door_right: Right boundary of tram door
        tram_length: Length of the tram (horizontal dimension)
        num_boarding_agents: Number of boarding agents
        num_exiting_agents: Number of exiting agents
        render_mode: Render mode
        max_steps: Maximum number of steps
        exiting_destination_area_y: Y-coordinate of exiting destination area (bottom border)
        boarding_destination_area_y: Y-coordinate of boarding destination area (in tram area)
        reward_config: Configuration for the reward function
        terminated_config: Configuration for the termination function
        truncated_config: Configuration for the truncation function

    """

    width: int = Field(description="Width of the environment", ge=1, le=100)
    height: int = Field(description="Height of the environment", ge=1, le=100)
    division_y: int = Field(
        description="Y-coordinate of the horizontal division line", ge=1, le=100
    )
    tram_door_left: int = Field(description="Left boundary of tram door", ge=0, le=100)
    tram_door_right: int = Field(description="Right boundary of tram door", ge=0, le=100)
    tram_length: int = Field(description="Length of the tram (horizontal dimension)", ge=1, le=100)
    num_boarding_agents: int = Field(description="Number of boarding agents", ge=0, le=100)
    num_exiting_agents: int = Field(description="Number of exiting agents", ge=0, le=100)

    render_mode: str | None = Field(
        description="Render mode",
        default=None,
        examples=["human", "rgb_array", None],
    )

    exiting_destination_area_y: int = Field(
        description="Y-coordinate of exiting destination area (bottom border)"
    )
    boarding_destination_area_y: int = Field(
        description="Y-coordinate of boarding destination area (in tram area)"
    )
    observation_config: ObservationConfig = Field(
        description="Configuration for the observation function",
        default_factory=DefaultObservationConfig,
    )
    reward_config: RewardConfig = Field(
        description="Configuration for the reward function",
        default_factory=DefaultRewardConfig,
    )
    terminated_config: TerminatedConfig = Field(
        description="Configuration for the termination function",
        default_factory=IndividualAtDestinationTerminatedConfig,
    )
    truncated_config: TruncatedConfig = Field(
        description="Configuration for the truncation function",
        default_factory=MaxStepsTruncatedConfig,
    )

    @model_validator(mode="after")
    def validate_config(self) -> "CollectiveCrossingConfig":
        """Validate the configuration after all fields are set."""
        self._validate_tram_parameters()
        self._validate_destination_areas()
        self._validate_environment_bounds()
        self._validate_agent_counts()
        self._validate_render_mode()
        return self

    def _validate_tram_parameters(self) -> None:
        """Validate tram-related parameters."""
        # Validate tram length vs environment width
        if self.tram_length > self.width:
            raise ValueError(
                f"Tram length ({self.tram_length}) cannot exceed grid width ({self.width})"
            )

        # Validate tram door boundaries
        if self.tram_door_left < 0 or self.tram_door_left >= self.tram_length:
            raise ValueError(
                f"Tram door left boundary ({self.tram_door_left}) must be within tram boundaries "
                f"(0 to {self.tram_length - 1})"
            )

        if self.tram_door_right < 0 or self.tram_door_right >= self.tram_length:
            raise ValueError(
                f"Tram door right boundary ({self.tram_door_right}) must be within tram boundaries "
                f"(0 to {self.tram_length - 1})"
            )

        # Validate tram door left <= right
        if self.tram_door_left > self.tram_door_right:
            raise ValueError(
                f"Tram door left boundary ({self.tram_door_left}) cannot be greater than "
                f"right boundary ({self.tram_door_right})"
            )

    def _validate_destination_areas(self) -> None:
        """Validate destination area parameters."""
        # Validate exiting destination area (should be in waiting area)
        if (
            self.exiting_destination_area_y < 0
            or self.exiting_destination_area_y >= self.division_y
        ):
            raise ValueError(
                f"Exiting destination area y-coordinate ({self.exiting_destination_area_y}) must "
                f"be within waiting area (0 to {self.division_y - 1})"
            )

        # Validate boarding destination area (should be in tram area)
        if (
            self.boarding_destination_area_y < self.division_y
            or self.boarding_destination_area_y > self.height
        ):
            raise ValueError(
                f"Boarding destination area y-coordinate ({self.boarding_destination_area_y}) "
                f"must be within tram area ({self.division_y} to {self.height})"
            )

    def _validate_environment_bounds(self) -> None:
        """Validate environment boundary parameters."""
        # Validate division line is within environment height
        if self.division_y >= self.height:
            raise ValueError(
                f"Division line y-coordinate ({self.division_y}) must be less than environment "
                f"height ({self.height})"
            )

        # Validate tram door boundaries are within environment width
        if self.tram_door_left >= self.width:
            raise ValueError(
                f"Tram door left boundary ({self.tram_door_left}) must be less than environment "
                f"width ({self.width})"
            )

        if self.tram_door_right >= self.width:
            raise ValueError(
                f"Tram door right boundary ({self.tram_door_right}) must be less than environment "
                f"width ({self.width})"
            )

    def _validate_agent_counts(self) -> None:
        """Validate agent count parameters."""
        total_agents = self.num_boarding_agents + self.num_exiting_agents

        # Check if total agents exceed reasonable limits
        max_agents = min(self.width * self.height // 4, 50)  # Conservative limit
        if total_agents > max_agents:
            raise ValueError(
                f"Total number of agents ({total_agents}) exceeds reasonable limit ({max_agents}) "
                f"for environment size {self.width}x{self.height}"
            )

        # Check if agents can fit in their respective areas
        waiting_area_size = self.width * self.division_y
        tram_area_size = self.width * (self.height - self.division_y)

        if self.num_exiting_agents > waiting_area_size // 2:
            raise ValueError(
                f"Number of exiting agents ({self.num_exiting_agents}) may be too high for "
                f"waiting area size ({waiting_area_size})"
            )

        if self.num_boarding_agents > tram_area_size // 2:
            raise ValueError(
                f"Number of boarding agents ({self.num_boarding_agents}) may be too high for "
                f"tram area size ({tram_area_size})"
            )

    def _validate_render_mode(self) -> None:
        """Validate render mode parameter."""
        valid_modes = ["human", "rgb_array", None]
        if self.render_mode not in valid_modes:
            raise ValueError(
                f"Invalid render_mode: {self.render_mode}. Valid modes are: {valid_modes}"
            )

    def get_validation_errors(self) -> list[str]:
        """
        Get all validation errors without raising exceptions.

        Note: As long as ConfigClass has frozen=True, this method will not be useful.
        In case of a validation error, the error will be raised at the moment of creating the
        config object.

        Returns
        -------
            List[str]: A list of validation error messages

        """
        errors = []

        try:
            self._validate_tram_parameters()
        except ValueError as e:
            errors.append(f"Tram parameter error: {e}")

        try:
            self._validate_destination_areas()
        except ValueError as e:
            errors.append(f"Destination area error: {e}")

        try:
            self._validate_environment_bounds()
        except ValueError as e:
            errors.append(f"Environment bounds error: {e}")

        try:
            self._validate_agent_counts()
        except ValueError as e:
            errors.append(f"Agent count error: {e}")

        try:
            self._validate_render_mode()
        except ValueError as e:
            errors.append(f"Render mode error: {e}")

        return errors

    def is_valid(self) -> bool:
        """Check if the configuration is valid."""
        return len(self.get_validation_errors()) == 0
