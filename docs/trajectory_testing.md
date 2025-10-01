# 🎬 Trajectory Testing with VCR

This document explains the VCR-style trajectory testing system used to ensure consistency during refactoring of the CollectiveCrossing environment.

## 📋 Overview

The trajectory testing system records environment interactions (actions → observations, rewards, terminations) and replays them to verify that refactored code produces identical behavior. This prevents regressions during code changes.

## ⚙️ How It Works

### VCR (Video Cassette Recorder) Concept

The system works like a VCR for environment interactions:

1. **Record Mode**: Capture complete environment state at each step
2. **Replay Mode**: Feed the same actions and verify identical outputs
3. **Comparison**: Detect any behavioral changes during refactoring

### Key Components

- **TrajectoryVCR Class**: Main recorder/replayer
- **Golden Baselines**: Known good trajectories from working code
- **Version-Specific Trajectories**: Track changes across versions
- **JSON Storage**: Trajectories stored as structured data files

## Directory Structure

```
tests/fixtures/trajectories/
├── golden/                    # Golden baselines (known good)
│   ├── golden_basic_trajectory.json
│   └── ...
├── current/                   # Current version trajectories
│   ├── test_basic_trajectory.json
│   └── ...
├── v1.0/                      # Version-specific trajectories
│   └── ...
└── v2.0/
    └── ...
```

## Version Control

### What to Commit

- **`golden/` directory**: Golden baselines should be committed to version control
- **📋 Test files**: All test files should be committed

### What NOT to Commit

- **`current/` directory**: Current trajectories are temporary test artifacts
- **Version-specific directories**: These are generated during testing

The `current/` directory is automatically ignored by `.gitignore`:

```gitignore
# VCR trajectory test artifacts
tests/fixtures/trajectories/current/
```

### Golden Baseline Lifecycle

1. **Create**: Golden baselines are created from known-good code
2. **Commit**: Golden baselines are committed to version control
3. **Test**: Tests compare current behavior against golden baselines
4. **Update**: Golden baselines are updated when behavior intentionally changes

## Usage

### 1. Creating Golden Baselines

Golden baselines are trajectories from known good code that serve as reference points.

```bash
# Create golden baseline from working code
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py::test_create_golden_baseline -v
```

**When to create golden baselines:**
- Before starting major refactoring
- After fixing bugs in working code
- When you have a stable, tested version

**Important**: Tests now preserve existing golden baselines. They will only create new ones if they don't exist, preventing accidental overwrites.

### 2. Comparing Against Golden Baselines

Compare current code behavior against golden baselines to detect regressions.

```bash
# Compare current trajectory with golden baseline
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py::test_golden_baseline_comparison -v
```

**What this catches:**
- Changes in agent behavior
- Reward calculation changes
- Termination condition changes
- Observation space changes

**Test Behavior**: This test requires the golden baseline to exist and will fail with a clear error message if it's missing.

### 3. Version-Specific Testing

Track changes across different versions of your code.

```bash
# Test version-specific trajectories
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py::test_version_specific_trajectories -v
```

### 4. Running All Tests

```bash
# Run all trajectory VCR tests
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py -v
```

## Creating New Versions

### Step 1: Create Version-Specific VCR

```python
from tests.collectivecrossing.envs.test_trajectory_vcr import TrajectoryVCR

# Create VCR for new version
vcr_new = TrajectoryVCR(version="v2.1")
```

### Step 2: Record Trajectories

```python
# Record trajectory for new version
trajectory = vcr_new.record_trajectory(env, actions_sequence, "new_feature_test")
```

### Step 3: Compare with Previous Version

```python
# Compare with previous version
vcr_old = TrajectoryVCR(version="v2.0")
vcr_old._compare_trajectories(old_trajectory, new_trajectory, "v2.0", "v2.1")
```

## Creating Golden Baselines

### Method 1: Using Test Functions

```bash
# Run the golden baseline creation test
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py::test_create_golden_baseline -v
```

### Method 2: Manual Creation

```python
from tests.collectivecrossing.envs.test_trajectory_vcr import TrajectoryVCR, create_test_environment, generate_deterministic_actions

# Create VCR
vcr = TrajectoryVCR()

# Create environment
env = create_test_environment()
observations, _ = env.reset(seed=42)

# Generate actions
actions_sequence = generate_deterministic_actions(observations, num_steps=20)

# Create golden baseline
trajectory = vcr.create_golden_baseline(env, actions_sequence, "my_golden_baseline")
```

### Method 3: Command Line Script

```bash
# Run the manual script
uv run python tests/collectivecrossing/envs/test_trajectory_vcr.py
```

## Trajectory Data Structure

Each trajectory is stored as a JSON file with the following structure:

```json
{
  "config": {
    "width": 10,
    "height": 6,
    "division_y": 3,
    "tram_door_left": 4,
    "tram_door_right": 5,
    "tram_length": 8,
    "num_boarding_agents": 2,
    "num_exiting_agents": 1,
    "exiting_destination_area_y": 0,
    "boarding_destination_area_y": 5
  },
  "initial_observations": {
    "boarding_0": [2, 1, 5, 3, 4, 5, ...],
    "boarding_1": [7, 2, 5, 3, 4, 5, ...],
    "exiting_0": [4, 4, 5, 3, 4, 5, ...]
  },
  "initial_infos": {
    "boarding_0": {"agent_type": "boarding"},
    "boarding_1": {"agent_type": "boarding"},
    "exiting_0": {"agent_type": "exiting"}
  },
  "steps": [
    {
      "step": 0,
      "actions": {
        "boarding_0": 0,
        "boarding_1": 2,
        "exiting_0": 3
      },
      "observations": {
        "boarding_0": [2, 1, 5, 3, 4, 5, ...],
        "boarding_1": [7, 2, 5, 3, 4, 5, ...],
        "exiting_0": [4, 4, 5, 3, 4, 5, ...]
      },
      "next_observations": {
        "boarding_0": [3, 1, 5, 3, 4, 5, ...],
        "boarding_1": [6, 2, 5, 3, 4, 5, ...],
        "exiting_0": [4, 3, 5, 3, 4, 5, ...]
      },
      "next_rewards": {
        "boarding_0": -0.3,
        "boarding_1": -0.4,
        "exiting_0": 0.1
      },
      "next_terminated": {
        "boarding_0": false,
        "boarding_1": false,
        "exiting_0": false,
        "__all__": false
      },
      "next_truncated": {
        "boarding_0": false,
        "boarding_1": false,
        "exiting_0": false,
        "__all__": false
      },
      "next_infos": {
        "boarding_0": {"agent_type": "boarding", "in_tram_area": false, "at_door": false},
        "boarding_1": {"agent_type": "boarding", "in_tram_area": false, "at_door": false},
        "exiting_0": {"agent_type": "exiting", "in_tram_area": true, "at_door": false}
      }
    }
  ]
}
```

## Best Practices

### 1. 🏆 When to Create Golden Baselines

- **Before major refactoring**: Create baselines from stable code
- **After bug fixes**: Update baselines to reflect correct behavior
- **Before releases**: Ensure baselines represent intended behavior

### 2. 🎯 Test Coverage

- **Multiple scenarios**: Create baselines for different environment configurations
- **Edge cases**: Include trajectories that test boundary conditions
- **Common paths**: Focus on typical agent behaviors

### 3. 🔧 Maintenance

- **Regular updates**: Update golden baselines when behavior intentionally changes
- **Version control**: Commit trajectory files to track changes over time
- **Documentation**: Document why baselines were updated

### 4. 🐛 Debugging

When tests fail, the system provides detailed information:

- **Step-by-step comparison**: Shows exactly where trajectories diverge
- **Agent-specific details**: Identifies which agents behave differently
- **State differences**: Shows observation, reward, and termination differences

## 🔧 Troubleshooting

### Understanding Test Skipping

The VCR testing system is designed to **skip tests** when required golden baseline files are missing. This is intentional behavior to prevent false failures when baseline data isn't available.

**Why tests are skipped:**
- **Golden baselines missing**: Tests require specific golden baseline files to compare against
- **No comparison data**: Without baselines, tests can't verify consistency
- **Prevents false failures**: Skipping is better than failing due to missing data

**Common skipped tests:**
- `test_replay_trajectory`: Requires `test_basic_trajectory.json` in golden directory
- `test_trajectory_consistency`: Requires `consistency_test.json` in golden directory

**How to identify what's missing:**
```bash
# Run tests with verbose output to see skip reasons
uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py -v -rs

# Check what golden baselines exist
ls tests/fixtures/trajectories/golden/

# Check what current trajectories exist
ls tests/fixtures/trajectories/current/
```

### 🚨 Common Issues

1. **Missing Golden Baseline**
   ```
   pytest.skip: Golden baseline test_name not found. Create golden baseline first.
   ```
   **Solution**: Run the golden baseline creation test first.

2. **Tests Being Skipped**
   ```
   pytest.skip: Golden cassette test_basic_trajectory not found. Create golden baseline first.
   pytest.skip: Golden cassette consistency_test not found. Create golden baseline first.
   ```
   ** Solution**: These tests require specific golden baseline files. You can resolve this by:
   
   **Option A: Create golden baselines automatically**
   ```bash
   # Create all required golden baselines
   uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py::test_create_golden_baseline -v
   ```
   
   **Option B: Copy existing current trajectories to golden baselines**
   ```bash
   # Copy specific missing files
   cp tests/fixtures/trajectories/current/test_basic_trajectory.json tests/fixtures/trajectories/golden/
   cp tests/fixtures/trajectories/current/consistency_test.json tests/fixtures/trajectories/golden/
   ```
   
   **Option C: Check what golden baselines exist**
   ```bash
   # List existing golden baselines
   ls tests/fixtures/trajectories/golden/
   
   # List current trajectories that can be copied
   ls tests/fixtures/trajectories/current/
   ```

3. **⚙️ Config Mismatch**
   ```
   pytest.fail: Config mismatch between golden and current
   ```
   **Solution**: Ensure environment configuration matches between recording and replay.

4. **👁️ Observation Mismatch**
   ```
   pytest.fail: Observation mismatch for agent_id at step N
   ```
   **Solution**: Check for changes in environment logic that affect agent behavior.

5. **🔄 Golden Baseline Modified**
   ```
   git status shows modified golden baseline files
   ```
   **💡 Solution**: Tests now preserve golden baselines. If you see modifications, it means:
   - The test detected a regression (intentional behavior)
   - You need to update golden baselines for intentional changes
   - Restore golden baselines with `git restore tests/fixtures/trajectories/golden/`

### 🛠️ Debugging Commands

```bash
# 📋 List available golden baselines
python -c "from tests.collectivecrossing.envs.test_trajectory_vcr import TrajectoryVCR; vcr = TrajectoryVCR(); print('Golden:', vcr.list_golden_baselines())"

# 📋 List current version trajectories
python -c "from tests.collectivecrossing.envs.test_trajectory_vcr import TrajectoryVCR; vcr = TrajectoryVCR(); print('Current:', vcr.list_version_trajectories())"

# 🔍 Inspect trajectory file
cat tests/fixtures/trajectories/golden/golden_basic_trajectory.json | jq '.steps[0]'
```

## 🔄 Integration with CI/CD

The trajectory testing system integrates with the GitHub Actions workflow:

```yaml
# .github/workflows/test.yml
- name: Run trajectory tests
  run: |
    uv run pytest tests/collectivecrossing/envs/test_trajectory_vcr.py -v
```

This ensures that:
- ✅ Trajectory consistency is checked on every commit
- 🚨 Regressions are caught before merging
- 📝 Behavioral changes are documented and reviewed

## 🚀 Advanced Usage

### 🎯 Custom Action Sequences

```python
def custom_action_sequence(observations, num_steps):
    """Generate custom deterministic actions"""
    actions_sequence = []
    for step in range(num_steps):
        actions = {}
        for agent_id in observations.keys():
            # Custom logic here
            actions[agent_id] = custom_policy(observations[agent_id])
        actions_sequence.append(actions)
    return actions_sequence

# Use custom actions
trajectory = vcr.record_trajectory(env, custom_action_sequence(observations, 20), "custom_test")
```

### 📋 Multiple Environment Configurations

```python
def test_multiple_configs():
    configs = [
        {"width": 10, "height": 6, "num_boarding_agents": 2},
        {"width": 15, "height": 8, "num_boarding_agents": 4},
        {"width": 8, "height": 4, "num_boarding_agents": 1}
    ]
    
    for i, config in enumerate(configs):
        env = create_test_environment_with_config(config)
        trajectory = vcr.create_golden_baseline(env, actions, f"config_{i}")
```

This trajectory testing system provides robust regression testing for the CollectiveCrossing environment, ensuring that refactoring doesn't introduce behavioral changes. 🎉
