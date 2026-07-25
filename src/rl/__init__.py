"""
Reinforcement learning support, for training assembled architectures on control tasks instead of on image
classification. The entry point is `src/rl_train.py`, the sibling of `src/stitch_train.py`.

The algorithm itself comes from Stable-Baselines3; this package supplies the glue:
  - `envs`: turning a "benchmark" plus an environment id into vectorized environments.
  - `models`: exposing a model built by `assembly.model_from_config()` as an SB3 features extractor.
  - `policies`: turning the `policy` sub-config into SB3 `policy_kwargs`.
  - `algo`: constructing and running the algorithm.
  - `callbacks`: reporting metrics and checkpoints through this repo's logging stack.
"""
