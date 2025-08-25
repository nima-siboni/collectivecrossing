from dataclasses import dataclass

from collectivecrossing.configs import CollectiveCrossingConfig


@dataclass
class TramBoundaries:
    """Dataclass containing tram and tram door boundaries"""

    tram_door_left: int
    tram_door_right: int
    tram_left: int
    tram_right: int


def calculate_tram_boundaries(config: CollectiveCrossingConfig) -> TramBoundaries:
    """
    Calculate tram and tram door boundaries based on configuration.

    Args:
        config: The CollectiveCrossingConfig containing tram parameters

    Returns:
        TramBoundaries: Dataclass containing all tram boundary values
    """
    # Calculate tram door boundaries
    tram_door_left = max(0, config.tram_door_x - config.tram_door_width // 2)
    tram_door_right = min(config.tram_length - 1, config.tram_door_x + config.tram_door_width // 2)

    # Calculate tram boundaries
    tram_left = 0
    tram_right = config.tram_length - 1

    return TramBoundaries(
        tram_door_left=tram_door_left,
        tram_door_right=tram_door_right,
        tram_left=tram_left,
        tram_right=tram_right,
    )
