from collectivecrossing.envs.collectivecrossing import CollectiveCrossingEnv
from gymnasium.envs.registration import register

register(
    id="collectivecrossing/CollectiveCrossing-v0",
    entry_point="collectivecrossing.envs:CollectiveCrossingEnv",
)

__all__ = ["CollectiveCrossingEnv"]
