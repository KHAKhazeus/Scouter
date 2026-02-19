---
name: Traditional RL Integration
overview: Replace the broken custom ScoutRllibEnv with a minimal PettingZoo AEC wrapper that restructures observations, then use RLlib's built-in PettingZooEnv + ActionMaskingTorchRLModule for PPO self-play training.
todos:
  - id: configurable-env
    content: Add num_rounds and reward_mode params to ScoutEnv.__init__(), replace hardcoded NUM_ROUNDS with self._num_rounds
    status: pending
  - id: flat-obs-wrapper
    content: "Write FlatObsWrapper(BaseWrapper) in rllib_wrapper.py that restructures ScoutEnv obs into {observations: flat_188d, action_mask: 256d}"
    status: pending
  - id: action-mask-rlmodule
    content: Write ScoutActionMaskRLModule inheriting from DefaultPPOTorchRLModule (avoids deprecation warning) with mask extraction + logit masking
    status: pending
  - id: register-helper
    content: "Write register_scout_env() that does register_env('scout_v0', lambda _: PettingZooEnv(FlatObsWrapper(ScoutEnv(...))))"
    status: pending
  - id: train-script
    content: Create scripts/train_ppo.py with PPO self-play config, shared policy, configurable hyperparams
    status: pending
  - id: tests
    content: Add tests for FlatObsWrapper, single-round mode, score_diff reward, and verify training loop runs
    status: pending
  - id: smoke-test
    content: "Run GPU smoke test: 100 training iterations on V100"
    status: pending
isProject: false
---

# Traditional RL via RLlib PettingZooEnv + Action Masking

## Key Insight: PettingZooEnv Is Usable -- With a Thin Obs Wrapper

The `register_env("scout_v0", lambda _: PettingZooEnv(ScoutEnv()))` pattern **does work**, but not directly. Here's why and what's needed:

### What PettingZooEnv does (and does NOT do)

From `[pettingzoo_env.py](scouter/../.venv/lib/python3.12/site-packages/ray/rllib/env/wrappers/pettingzoo_env.py)`:

- **Does**: Correctly handles AEC turn-taking (including Scout's tempo rule where scouting keeps the turn), dead-agent stepping, and the MultiAgentEnv protocol.
- **Does NOT**: Transform or restructure observations. It passes `env.observe()` output through unchanged.

### The Gap

RLlib's action masking convention (both old and new API) requires the observation space to be a `Dict` with exactly two keys:

- `"observations"` -- a flat Box (the actual observation)
- `"action_mask"` -- a Box(0, 1) mask

ScoutEnv currently returns a Dict with 12+ keys (`hand`, `active_set`, `scout_chips`, ..., `action_mask`). This does not match.

### Solution: Thin AEC Wrapper

A ~40-line PettingZoo `BaseWrapper` subclass that sits **between** ScoutEnv and PettingZooEnv:

```
ScoutEnv (12-key Dict obs)
    |
FlatObsWrapper (PettingZoo AEC wrapper)
    - Flattens all obs keys (except action_mask) -> 188-d float32 "observations"
    - Keeps "action_mask" as 256-d int8
    - New obs space: Dict({"observations": Box(188,), "action_mask": Box(256,)})
    |
PettingZooEnv (RLlib's built-in AEC -> MultiAgentEnv adapter)
    - Handles turn-taking, dead steps, etc.
    |
register_env("scout_v0", lambda _: PettingZooEnv(FlatObsWrapper(ScoutEnv())))
```

### Model: No Custom Model Needed

Ray 2.54.0 (installed) ships a built-in `ActionMaskingTorchRLModule` at `[ray.rllib.examples.rl_modules.classes.action_masking_rlm](scouter/../.venv/lib/python3.12/site-packages/ray/rllib/examples/rl_modules/classes/action_masking_rlm.py)` that:

1. Extracts `"action_mask"` from `batch[OBS]`
2. Feeds `"observations"` through `DefaultPPOTorchRLModule`'s encoder (MLP)
3. Applies `logits += clamp(log(action_mask), min=-inf)` to mask invalid actions
4. Returns masked logits for the Categorical action distribution

Note: this built-in class emits a deprecation warning (imports old `PPOTorchRLModule`). We should write our own slim version inheriting from `DefaultPPOTorchRLModule` to avoid that (~50 lines, same logic).

### Data Flow: Input -> Model -> Output

```
Observation Dict from ScoutEnv.observe()
  {hand: (14,3), hand_size, active_set: (14,3), ..., action_mask: (256,)}
          |
    FlatObsWrapper.observe()
          |
  {"observations": float32[188], "action_mask": int8[256]}
          |
    PettingZooEnv passes through
          |
    ActionMaskingRLModule._preprocess_batch()
      -> extracts action_mask tensor
      -> passes observations (188-d) to PPO encoder
          |
    PPO Encoder (MLP):
      Linear(188, 256) -> ReLU -> Linear(256, 256) -> ReLU
          |
    +-- Pi Head: Linear(256, 256) -> raw logits
    |     |
    |   logits[mask==0] = -inf   (via log(action_mask) clamping)
    |     |
    |   Categorical(softmax(masked_logits)) -> sample action int
    |
    +-- Value Head: Linear(256, 1) -> V(s) scalar (for GAE)
          |
    action int in [0, 256)
          |
    is_show_action(a)? -> decode_show(a) -> Show(start, end)
    is_scout_action(a)? -> decode_scout(a) -> Scout(side, flip, insert_pos)
          |
    env.step(action)
```

## Changes to Make

### 1. ScoutEnv: add `num_rounds` + `reward_mode` params

File: `[scouter/env/scout_env.py](scouter/env/scout_env.py)`

- Add `num_rounds: int = 2` and `reward_mode: str = "raw"` params to `__init__()`.
- `reward_mode="score_diff"` computes `my_score - opp_score` in `_end_round()`.
- Replace hardcoded `NUM_ROUNDS` references with `self._num_rounds`.
- Backward-compatible: existing `ScoutEnv()` calls work unchanged.

### 2. Rewrite `rllib_wrapper.py`

File: `[scouter/rl/rllib_wrapper.py](scouter/rl/rllib_wrapper.py)`

Replace the entire broken `ScoutRllibEnv` class with three clean components:

**a) `FlatObsWrapper(pettingzoo.utils.BaseWrapper)`** (~40 lines)

- Overrides `observation_space()` to return `Dict({"observations": Box(flat_size,), "action_mask": Box(256,)})`.
- Overrides `observe()` to flatten the dict and restructure.
- All AEC protocol methods (`step`, `reset`, `last`, etc.) inherited from BaseWrapper.

**b) `ScoutActionMaskRLModule(DefaultPPOTorchRLModule)`** (~50 lines)

- Clean re-implementation of the action masking pattern from Ray's example, but inheriting from the non-deprecated `DefaultPPOTorchRLModule`.
- Overrides `_forward`, `_forward_train`, `compute_values` to extract mask and apply it.

**c) `register_scout_env()` helper** (~5 lines)

```python
def register_scout_env(num_rounds=2, reward_mode="raw"):
    from ray.tune.registry import register_env
    register_env("scout_v0", lambda _: PettingZooEnv(
        FlatObsWrapper(ScoutEnv(num_rounds=num_rounds, reward_mode=reward_mode))
    ))
```

### 3. Training script

File: `scripts/train_ppo.py` (new)

A working PPO self-play training script (~60 lines):

- Calls `register_scout_env()`
- Configures PPO with `ScoutActionMaskRLModule`, shared policy for both agents
- Runs with `tune.run()` or `algo.train()` loop
- Configurable: hidden sizes, lr, num workers, GPU

### 4. Tests + smoke test

- Add test for `FlatObsWrapper` (correct obs shape, mask passes through).
- Add test for `num_rounds=1` mode and `reward_mode="score_diff"`.
- GPU smoke test: 100 training steps on V100.

