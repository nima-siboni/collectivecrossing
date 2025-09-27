"""Geometry utilities for the collective crossing environment."""

from dataclasses import dataclass

import numpy as np

from collectivecrossing.configs import CollectiveCrossingConfig


@dataclass
class TramBoundaries:
    """Dataclass containing tram and tram door boundaries."""

    tram_door_left: int
    tram_door_right: int
    tram_left: int
    tram_right: int


def calculate_tram_boundaries(config: CollectiveCrossingConfig) -> TramBoundaries:
    """
    Create tram and tram door boundaries from configuration.

    Args:
    ----
        config: The CollectiveCrossingConfig containing tram parameters

    Returns:
    -------
        TramBoundaries: Dataclass containing all tram boundary values

    """
    # Calculate tram boundaries (centered in environment)
    tram_center = config.width // 2
    tram_left = tram_center - config.tram_length // 2
    tram_right = tram_center + config.tram_length // 2

    # Convert tram door boundaries from relative to absolute coordinates
    tram_door_left = tram_left + config.tram_door_left
    tram_door_right = tram_left + config.tram_door_right

    return TramBoundaries(
        tram_door_left=tram_door_left,
        tram_door_right=tram_door_right,
        tram_left=tram_left,
        tram_right=tram_right,
    )


def calculate_distance(
    pos1: tuple[int | None, int | None], pos2: tuple[int | None, int | None]
) -> float:
    """Calculate the distance between two positions."""
    if pos1[0] is None or pos2[0] is None:
        return abs(pos1[1] - pos2[1])  # type: ignore[operator]
    if pos1[1] is None or pos2[1] is None:
        return abs(pos1[0] - pos2[0])

    return np.linalg.norm(np.array(pos1) - np.array(pos2))
