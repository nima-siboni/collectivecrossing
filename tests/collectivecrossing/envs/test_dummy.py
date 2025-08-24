def test_dummy():
    """Basic test to ensure the package can be imported"""
    # Test basic import
    from collectivecrossing import CollectiveCrossingEnv

    assert CollectiveCrossingEnv is not None

    # Test environment creation
    env = CollectiveCrossingEnv(
        width=5,
        height=4,
        division_y=2,
        tram_door_x=2,
        tram_door_width=1,
        tram_length=4,
        num_boarding_agents=1,
        num_exiting_agents=1,
    )
    assert env is not None
    assert env.width == 5
    assert env.height == 4
