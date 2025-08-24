def test_dummy():
    """Basic test to ensure the package can be imported"""
    from collectivecrossing import CollectiveCrossingEnv

    # Create a simple environment to test basic functionality
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

    # Test that environment can be reset
    observations, infos = env.reset(seed=42)
    assert len(observations) == 2  # 1 boarding + 1 exiting agent
    assert len(infos) == 2

    # Test that environment can take a step
    actions = {agent_id: 4 for agent_id in observations.keys()}  # Wait action
    new_observations, rewards, terminated, truncated, new_infos = env.step(actions)
    assert len(new_observations) == 2

    # Test that rendering works
    rgb_array = env.render()
    assert rgb_array.shape == (400, 500, 3)  # Based on figsize=(10, 8), dpi=50
    assert rgb_array.dtype == "uint8"
