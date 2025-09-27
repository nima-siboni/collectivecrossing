"""Tests for trajectory VCR functionality."""

import json
from pathlib import Path

import numpy as np
import pytest
from _pytest.outcomes import Failed

from collectivecrossing import CollectiveCrossingEnv
from collectivecrossing.configs import CollectiveCrossingConfig
from collectivecrossing.truncated_configs import MaxStepsTruncatedConfig


class TrajectoryVCR:
    """VCR-style recorder for environment trajectories."""

    def __init__(
        self, cassette_dir: str = "tests/fixtures/trajectories", version: str | None = None
    ) -> None:
        """
        Initialize the trajectory VCR recorder.

        Args:
        ----
            cassette_dir: Directory to store trajectory cassettes.
            version: Version identifier for the cassette.

        """
        self.cassette_dir = Path(cassette_dir)
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        self.version = version or "current"

        # Create version-specific subdirectories
        self.golden_dir = self.cassette_dir / "golden"
        self.golden_dir.mkdir(exist_ok=True)
        self.version_dir = self.cassette_dir / self.version
        self.version_dir.mkdir(exist_ok=True)

    def record_trajectory(
        self,
        env: CollectiveCrossingEnv,
        actions_sequence: list[dict[str, int]],
        cassette_name: str,
        golden: bool = False,
    ) -> dict:
        """Record a trajectory by running actions and capturing all states."""
        if golden:
            cassette_path = self.golden_dir / f"{cassette_name}.json"
        else:
            cassette_path = self.version_dir / f"{cassette_name}.json"

        # Reset environment
        observations, infos = env.reset(seed=42)

        trajectory: dict = {
            "config": {
                "width": env.config.width,
                "height": env.config.height,
                "division_y": env.config.division_y,
                "tram_door_left": env.config.tram_door_left,
                "tram_door_right": env.config.tram_door_right,
                "tram_length": env.config.tram_length,
                "num_boarding_agents": env.config.num_boarding_agents,
                "num_exiting_agents": env.config.num_exiting_agents,
                "exiting_destination_area_y": env.config.exiting_destination_area_y,
                "boarding_destination_area_y": env.config.boarding_destination_area_y,
                "render_mode": env.config.render_mode,
                "reward_config": env.config.reward_config.model_dump(),
                "terminated_config": env.config.terminated_config.model_dump(),
                "truncated_config": env.config.truncated_config.model_dump(),
            },
            "initial_observations": {k: v.tolist() for k, v in observations.items()},
            "initial_infos": {
                k: {ik: bool(iv) if isinstance(iv, bool | np.bool_) else iv for ik, iv in v.items()}
                for k, v in infos.items()
            },
            "steps": [],
        }

        # Record each step
        for step_num, actions in enumerate(actions_sequence):
            # Filter out actions for inactive agents
            active_actions = {
                agent_id: action
                for agent_id, action in actions.items()
                if agent_id in env._agents and env._agents[agent_id].active
            }

            # Record current state
            step_data = {
                "step": step_num,
                "actions": actions,  # Keep original actions for record
                "active_actions": active_actions,  # Add filtered actions
                "observations": {k: v.tolist() for k, v in observations.items()},
            }

            # Take step with only active agents
            observations, rewards, terminated, truncated, infos = env.step(active_actions)

            # Store step data (with results from the step)
            step_data["next_observations"] = {k: v.tolist() for k, v in observations.items()}
            step_data["next_rewards"] = {k: float(v) for k, v in rewards.items()}
            step_data["next_terminated"] = {k: bool(v) for k, v in terminated.items()}
            step_data["next_truncated"] = {k: bool(v) for k, v in truncated.items()}
            step_data["next_infos"] = {
                k: {ik: bool(iv) if isinstance(iv, bool | np.bool_) else iv for ik, iv in v.items()}
                for k, v in infos.items()
            }

            trajectory["steps"].append(step_data)

            # Check if episode is done
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        # Save trajectory
        with open(cassette_path, "w") as f:
            json.dump(trajectory, f, indent=2)

        return trajectory

    def replay_trajectory(
        self, env: CollectiveCrossingEnv, cassette_name: str, use_golden: bool = True
    ) -> dict:
        """Replay a recorded trajectory and verify consistency."""
        if use_golden:
            cassette_path = self.golden_dir / f"{cassette_name}.json"
        else:
            cassette_path = self.version_dir / f"{cassette_name}.json"

        if not cassette_path.exists():
            if use_golden:
                pytest.skip(
                    f"Golden cassette {cassette_name} not found. Create golden baseline first."
                )
            else:
                pytest.skip(
                    f"Version cassette {cassette_name} not found. Run in record mode first."
                )

        with open(cassette_path) as f:
            trajectory = json.load(f)

        # Reset environment
        observations, infos = env.reset(seed=42)

        # Verify initial state
        for agent_id, expected_obs in trajectory["initial_observations"].items():
            assert agent_id in observations, f"Agent {agent_id} missing in replay"
            np.testing.assert_array_equal(
                observations[agent_id],
                np.array(expected_obs),
                err_msg=f"Initial observation mismatch for {agent_id}",
            )

        # Replay each step
        for step_data in trajectory["steps"]:
            step_num = step_data["step"]
            actions = step_data["actions"]

            # Verify current observations match
            for agent_id, expected_obs in step_data["observations"].items():
                if agent_id in observations:  # Agent might have been terminated
                    np.testing.assert_array_equal(
                        observations[agent_id],
                        np.array(expected_obs),
                        err_msg=f"Step {step_num} observation mismatch for {agent_id}",
                    )

            # Take step
            observations, rewards, terminated, truncated, infos = env.step(actions)

            # Verify next state matches
            for agent_id, expected_obs in step_data["next_observations"].items():
                if agent_id in observations:  # Agent might have been terminated
                    np.testing.assert_array_equal(
                        observations[agent_id],
                        np.array(expected_obs),
                        err_msg=f"Step {step_num} next observation mismatch for {agent_id}",
                    )

            # Verify rewards match
            for agent_id, expected_reward in step_data["next_rewards"].items():
                if agent_id in rewards:  # Agent might have been terminated
                    assert abs(rewards[agent_id] - expected_reward) < 1e-6, (
                        f"Step {step_num} reward mismatch for {agent_id}"
                    )

            # Verify termination states match
            for agent_id, expected_terminated in step_data["next_terminated"].items():
                if agent_id in terminated:  # Agent might have been terminated
                    assert terminated[agent_id] == expected_terminated, (
                        f"Step {step_num} termination mismatch for {agent_id}"
                    )

        return trajectory

    def create_golden_baseline(
        self, env: CollectiveCrossingEnv, actions_sequence: list[dict[str, int]], cassette_name: str
    ) -> dict:
        """Create a golden baseline trajectory from known good code."""
        return self.record_trajectory(env, actions_sequence, cassette_name, golden=True)

    def compare_with_golden(
        self, env: CollectiveCrossingEnv, cassette_name: str
    ) -> tuple[dict, dict]:
        """Compare current trajectory with golden baseline."""
        golden_path = self.golden_dir / f"{cassette_name}.json"
        current_path = self.version_dir / f"{cassette_name}.json"

        if not golden_path.exists():
            pytest.skip(f"Golden baseline {cassette_name} not found. Create golden baseline first.")

        if not current_path.exists():
            pytest.skip(
                f"Current trajectory {cassette_name} not found. Record current trajectory first."
            )

        # Load both trajectories
        with open(golden_path) as f:
            golden_trajectory = json.load(f)

        with open(current_path) as f:
            current_trajectory = json.load(f)

        # Compare trajectories
        self._compare_trajectories(golden_trajectory, current_trajectory, "golden", "current")

        return golden_trajectory, current_trajectory

    def _compare_trajectories(
        self, trajectory1: dict, trajectory2: dict, name1: str, name2: str
    ) -> None:
        """Compare two trajectories and report differences."""
        # Compare configs
        if trajectory1["config"] != trajectory2["config"]:
            pytest.fail(f"Config mismatch between {name1} and {name2}")

        # Compare initial observations
        for agent_id in trajectory1["initial_observations"]:
            if agent_id not in trajectory2["initial_observations"]:
                pytest.fail(f"Agent {agent_id} missing in {name2} initial observations")

            obs1 = np.array(trajectory1["initial_observations"][agent_id])
            obs2 = np.array(trajectory2["initial_observations"][agent_id])

            if not np.array_equal(obs1, obs2):
                pytest.fail(
                    f"Initial observation mismatch for {agent_id} between {name1} and {name2}"
                )

        # Compare steps
        steps1 = trajectory1["steps"]
        steps2 = trajectory2["steps"]

        if len(steps1) != len(steps2):
            pytest.fail(
                f"Step count mismatch: {name1} has {len(steps1)} steps, {name2} has "
                f"{len(steps2)} steps"
            )

        for step_num, (step1, step2) in enumerate(zip(steps1, steps2, strict=True)):
            # Compare actions
            if step1["actions"] != step2["actions"]:
                pytest.fail(f"Action mismatch at step {step_num} between {name1} and {name2}")

            # Compare next observations
            for agent_id in step1["next_observations"]:
                if agent_id not in step2["next_observations"]:
                    pytest.fail(f"Agent {agent_id} missing in {name2} at step {step_num}")

                obs1 = np.array(step1["next_observations"][agent_id])
                obs2 = np.array(step2["next_observations"][agent_id])

                if not np.array_equal(obs1, obs2):
                    pytest.fail(
                        f"Observation mismatch for {agent_id} at step {step_num} between "
                        f"{name1} and {name2}"
                    )

            # Compare rewards
            for agent_id in step1["next_rewards"]:
                if agent_id not in step2["next_rewards"]:
                    pytest.fail(f"Agent {agent_id} reward missing in {name2} at step {step_num}")

                reward1 = step1["next_rewards"][agent_id]
                reward2 = step2["next_rewards"][agent_id]

                if abs(reward1 - reward2) > 1e-6:
                    pytest.fail(
                        f"Reward mismatch for {agent_id} at step {step_num}: {name1}={reward1}, "
                        f"{name2}={reward2}"
                    )

            # Compare termination states
            for agent_id in step1["next_terminated"]:
                if agent_id not in step2["next_terminated"]:
                    pytest.fail(
                        f"Agent {agent_id} termination missing in {name2} at step {step_num}"
                    )

                term1 = step1["next_terminated"][agent_id]
                term2 = step2["next_terminated"][agent_id]

                if term1 != term2:
                    pytest.fail(
                        f"Termination mismatch for {agent_id} at step {step_num}: "
                        f"{name1}={term1}, {name2}={term2}"
                    )

    def list_golden_baselines(self) -> list[str]:
        """List all available golden baselines."""
        golden_files = list(self.golden_dir.glob("*.json"))
        return [f.stem for f in golden_files]

    def list_version_trajectories(self) -> list[str]:
        """List all available version trajectories."""
        version_files = list(self.version_dir.glob("*.json"))
        return [f.stem for f in version_files]


def create_test_environment() -> CollectiveCrossingEnv:
    """Create a standard test environment."""
    return CollectiveCrossingEnv(
        config=CollectiveCrossingConfig(
            width=10,
            height=6,
            division_y=3,
            tram_door_left=3,  # Relative to tram (tram_left + 3 = 4)
            tram_door_right=5,  # Relative to tram (tram_left + 5 = 6)
            tram_length=8,
            num_boarding_agents=2,
            num_exiting_agents=1,
            exiting_destination_area_y=0,
            boarding_destination_area_y=5,
            truncated_config=MaxStepsTruncatedConfig(max_steps=50),
            render_mode=None,
        )
    )


def generate_deterministic_actions(
    observations: dict[str, np.ndarray], num_steps: int = 20
) -> list[dict[str, int]]:
    """Generate deterministic actions based on agent positions."""
    actions_sequence = []

    for step in range(num_steps):
        actions = {}
        for agent_id in observations.keys():
            # Simple deterministic policy: move towards door for boarding agents
            if agent_id.startswith("boarding"):
                obs = observations[agent_id]
                x, y = obs[:2]

                # Move towards door (positions 3, 5, or 7 are adjacent to occupied door at 4,6)
                if x < 3:  # Move towards left door position
                    actions[agent_id] = 0  # Right
                elif x > 7:  # Move towards right door position
                    actions[agent_id] = 2  # Left
                elif x == 3 or x == 5 or x == 7:  # At door positions, move up
                    if y < 3:  # Division line is at y=3
                        actions[agent_id] = 1  # Up
                    else:
                        actions[agent_id] = 4  # Wait
                else:  # Between door positions (4,6) - move to adjacent door
                    if x < 4:
                        actions[agent_id] = 0  # Right
                    elif x == 4:  # At left occupied door
                        actions[agent_id] = 0  # Right to position 5
                    elif x == 6:  # At right occupied door
                        actions[agent_id] = 2  # Left to position 5
                    else:  # x == 5, at center door position
                        actions[agent_id] = 1  # Up
            else:  # Exiting agents
                obs = observations[agent_id]
                x, y = obs[:2]

                # Move towards exit, but stop after reaching destination
                if y > 0:
                    actions[agent_id] = 3  # Down
                else:
                    actions[agent_id] = 4  # Wait

        actions_sequence.append(actions)

        # Update observations for next step (simplified)
        # In real implementation, this would be done by the environment
        if step < num_steps - 1:
            # This is a simplified update - in practice, the environment handles this
            pass

    return actions_sequence


@pytest.fixture
def vcr() -> TrajectoryVCR:
    """Fixture for the trajectory VCR recorder."""
    return TrajectoryVCR()


def test_record_trajectory(vcr: TrajectoryVCR) -> None:
    """Record a trajectory for future comparison."""
    env = create_test_environment()
    observations, _ = env.reset(seed=42)

    # Generate deterministic actions
    actions_sequence = generate_deterministic_actions(observations, num_steps=15)

    # Record trajectory
    trajectory = vcr.record_trajectory(env, actions_sequence, "test_basic_trajectory")

    # Verify trajectory was recorded
    assert len(trajectory["steps"]) > 0
    assert "config" in trajectory
    assert "initial_observations" in trajectory


def test_replay_trajectory(vcr: TrajectoryVCR) -> None:
    """Replay a recorded trajectory and verify consistency."""
    env = create_test_environment()

    # Replay trajectory
    trajectory = vcr.replay_trajectory(env, "test_basic_trajectory")

    # Verify trajectory was replayed
    assert len(trajectory["steps"]) > 0


def test_trajectory_consistency(vcr: TrajectoryVCR) -> None:
    """Test that the same environment produces consistent trajectories."""
    env1 = create_test_environment()
    env2 = create_test_environment()

    observations1, _ = env1.reset(seed=42)
    observations2, _ = env2.reset(seed=42)

    # Generate same actions
    actions_sequence = generate_deterministic_actions(observations1, num_steps=10)

    # Record trajectory from first environment
    _trajectory1 = vcr.record_trajectory(env1, actions_sequence, "consistency_test")

    # Replay trajectory on second environment (this verifies consistency)
    # The replay_trajectory method will raise assertions if there are any mismatches
    vcr.replay_trajectory(env2, "consistency_test")

    # If we get here, the replay was successful and trajectories are consistent


def test_trajectory_with_random_actions(vcr: TrajectoryVCR) -> None:
    """Test trajectory recording with random but seeded actions."""
    env = create_test_environment()
    observations, _ = env.reset(seed=42)

    # Generate random actions with fixed seed
    np.random.seed(123)
    actions_sequence = []
    for _ in range(10):
        actions = {}
        for agent_id in observations.keys():
            actions[agent_id] = np.random.randint(0, 5)  # Random action
        actions_sequence.append(actions)

    # Record trajectory
    trajectory = vcr.record_trajectory(env, actions_sequence, "random_trajectory")

    # Verify trajectory was recorded
    assert len(trajectory["steps"]) == 10


def test_create_golden_baseline(vcr: TrajectoryVCR) -> None:
    """Create a golden baseline from known good code."""
    env = create_test_environment()
    observations, _ = env.reset(seed=42)

    # Generate deterministic actions
    actions_sequence = generate_deterministic_actions(observations, num_steps=10)

    # Create golden baseline
    trajectory = vcr.create_golden_baseline(env, actions_sequence, "golden_basic_trajectory")

    # Verify golden baseline was created
    assert len(trajectory["steps"]) > 0
    assert "config" in trajectory
    assert "initial_observations" in trajectory

    # Verify golden baseline file exists
    golden_path = vcr.golden_dir / "golden_basic_trajectory.json"
    assert golden_path.exists()


def test_golden_baseline_comparison(vcr: TrajectoryVCR) -> None:
    """
    Compare current trajectory with existing golden baseline.

    This test verifies that the golden baseline comparison mechanism works correctly.
    It uses an existing golden baseline and compares it with a current trajectory
    created from the same environment code.

    NOTE: This test demonstrates the basic golden baseline comparison mechanism,
    but it doesn't actually test for regressions since both golden and current
    trajectories are created with the same environment code. For actual regression
    testing, see test_regression_detection().

    REQUIREMENT: Golden baseline must exist before running this test.
    """
    # Check if golden baseline exists - fail if it doesn't
    golden_path = vcr.golden_dir / "golden_basic_trajectory.json"
    if not golden_path.exists():
        pytest.fail(
            "Golden baseline 'golden_basic_trajectory' not found. "
            "Create golden baseline first using test_create_golden_baseline() or manually."
        )

    # Now create current trajectory with a different environment instance
    # This simulates running the test with potentially modified code
    env_current = create_test_environment()
    observations_current, _ = env_current.reset(seed=42)
    actions_sequence = generate_deterministic_actions(observations_current, num_steps=10)

    # Record current trajectory (same name as golden baseline, but in version_dir)
    vcr.record_trajectory(env_current, actions_sequence, "golden_basic_trajectory")

    # Compare with golden baseline
    golden_traj, current_traj = vcr.compare_with_golden(env_current, "golden_basic_trajectory")

    # Verify trajectories are identical
    assert golden_traj == current_traj


def test_regression_detection(vcr: TrajectoryVCR) -> None:
    """
    Test that golden baselines can detect actual regressions.

    NOTE: This test artificially simulates a regression by modifying the golden baseline file.
    In practice, regressions would be detected when:
    1. Golden baselines are created from known-good code and committed to git
    2. Code changes introduce bugs that change behavior
    3. Current trajectories differ from golden baselines

    This test demonstrates the mechanism but doesn't test real code changes.
    """
    # Create a golden baseline first (only if it doesn't exist)
    golden_path = vcr.golden_dir / "regression_test.json"
    if not golden_path.exists():
        env_golden = create_test_environment()
        observations_golden, _ = env_golden.reset(seed=42)
        actions_sequence = generate_deterministic_actions(observations_golden, num_steps=5)
        vcr.create_golden_baseline(env_golden, actions_sequence, "regression_test")

    # Save the original golden baseline for restoration
    with open(golden_path) as f:
        original_golden = json.load(f)

    # Now simulate a "current" run with the same environment
    # In practice, this would be a different version of the code
    env_current = create_test_environment()
    observations_current, _ = env_current.reset(seed=42)
    actions_sequence = generate_deterministic_actions(observations_current, num_steps=5)

    # Record current trajectory (same name as golden baseline, but in version_dir)
    vcr.record_trajectory(env_current, actions_sequence, "regression_test")

    # This should pass because both environments are identical
    golden_traj, current_traj = vcr.compare_with_golden(env_current, "regression_test")
    assert golden_traj == current_traj

    # Now let's simulate what would happen if there was a bug
    # We'll modify the golden trajectory to simulate a regression
    with open(golden_path) as f:
        modified_golden = json.load(f)

    # Modify the golden trajectory to simulate a bug
    if modified_golden["steps"]:
        # Change a reward value to simulate a bug
        first_step = modified_golden["steps"][0]
        if first_step["next_rewards"]:
            first_agent = list(first_step["next_rewards"].keys())[0]
            first_step["next_rewards"][first_agent] = 999.0  # Obviously wrong value

    # Save the modified golden baseline
    with open(golden_path, "w") as f:
        json.dump(modified_golden, f, indent=2)

    # Now the comparison should fail
    with pytest.raises(Failed):
        vcr.compare_with_golden(env_current, "regression_test")

    # Restore the original golden baseline
    with open(golden_path, "w") as f:
        json.dump(original_golden, f, indent=2)


def test_list_trajectories(vcr: TrajectoryVCR) -> None:
    """Test listing available trajectories."""
    env = create_test_environment()
    observations, _ = env.reset(seed=42)
    actions_sequence = generate_deterministic_actions(observations, num_steps=5)

    # Create golden baseline (only if it doesn't exist)
    golden_path = vcr.golden_dir / "test_list_golden.json"
    if not golden_path.exists():
        vcr.create_golden_baseline(env, actions_sequence, "test_list_golden")

    # Create current trajectory
    vcr.record_trajectory(env, actions_sequence, "test_list_current")

    # List trajectories
    golden_baselines = vcr.list_golden_baselines()
    version_trajectories = vcr.list_version_trajectories()

    # Verify lists contain expected trajectories
    assert "test_list_golden" in golden_baselines
    assert "test_list_current" in version_trajectories


if __name__ == "__main__":
    # Manual test: record a trajectory
    vcr_manual = TrajectoryVCR()
    env = create_test_environment()
    observations, _ = env.reset(seed=42)
    actions_sequence = generate_deterministic_actions(observations, num_steps=20)
    vcr_manual.record_trajectory(env, actions_sequence, "manual_test_trajectory")
    print("Trajectory recorded successfully!")
