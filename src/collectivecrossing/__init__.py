"""Collective crossing environment package."""

from gymnasium.envs.registration import register

from .collectivecrossing import CollectiveCrossingEnv

register(
    id="collectivecrossing/CollectiveCrossing-v0",
    entry_point="collectivecrossing:CollectiveCrossingEnv",
)

__all__ = ["CollectiveCrossingEnv"]
