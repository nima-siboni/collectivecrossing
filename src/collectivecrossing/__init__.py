from gymnasium.envs.registration import register

from .envs.collectivecrossing import CollectiveCrossingEnv

register(
    id="collectivecrossing/CollectiveCrossing-v0",
    entry_point="collectivecrossing.envs:CollectiveCrossingEnv",
)

__all__ = ["CollectiveCrossingEnv"]
