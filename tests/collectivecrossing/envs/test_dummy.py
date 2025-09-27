"""Dummy tests for the collective crossing environment."""

from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig


def test_dummy() -> None:
    """Basic test to ensure the package can be imported."""
    # Test basic import
    from collectivecrossing import CollectiveCrossingEnv

    assert CollectiveCrossingEnv is not None

    # Test environment creation
    env = CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=5,
            height=4,
            division_y=2,
            tram_door_left=2,
            tram_door_right=2,
            tram_length=4,
            num_boarding_agents=1,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=3,
            truncated_config=MaxStepsTruncatedConfig(max_steps=100),
            render_mode="human",
        )
    )
    assert env is not None
    assert env.config.width == 5
    assert env.config.height == 4
    assert env.config.division_y == 2
    assert env.config.tram_door_left == 2
    assert env.config.tram_door_right == 2
    assert env.config.tram_length == 4
    assert env.config.num_boarding_agents == 1
    assert env.config.num_exiting_agents == 1
    assert env.config.exiting_destination_area_y == 0
    assert env.config.boarding_destination_area_y == 3
