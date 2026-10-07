import pytest

from rl.envs import BENCHMARKS, get_benchmark, make_train_and_eval_envs, make_vec_envs
from tests.rl.configs import smoke_config, validated


def obs_shape(obs_mode, **train_overrides):
    config = validated(train_config={"obs_mode": obs_mode, **train_overrides})
    venv = make_vec_envs(config, n_envs=1)
    try:
        return venv.observation_space.shape, venv.observation_space.dtype
    finally:
        venv.close()


def test_symbolic_obs_is_the_transposed_grid():
    # MiniGrid gives (7, 7, 3); SB3 expects channels first, so the vec env must present (3, 7, 7).
    shape, dtype = obs_shape("symbolic")
    assert shape == (3, 7, 7)
    assert dtype == "uint8"


def test_rgb_obs_is_rendered_at_tile_size():
    shape, _ = obs_shape("rgb", obs_kwargs={"tile_size": 8})
    assert shape == (3, 56, 56), "A 7x7 view at 8 pixels per tile is 56x56, transposed to channels first."


def test_rgb_tile_size_is_configurable():
    shape, _ = obs_shape("rgb", obs_kwargs={"tile_size": 4})
    assert shape == (3, 28, 28)


def test_flat_obs_is_one_dimensional():
    shape, _ = obs_shape("flat")
    assert len(shape) == 1, "Flat observations must not be transposed or otherwise treated as images."


def test_minigrid_success_is_a_positive_return():
    # MiniGrid pays out a positive, time-discounted reward on success and exactly zero on failure.
    benchmark = get_benchmark("minigrid")
    assert benchmark.is_success({"r": 0.7})
    assert not benchmark.is_success({"r": 0.0})


def test_unknown_benchmark_is_rejected():
    with pytest.raises(ValueError, match="Unrecognized benchmark"):
        get_benchmark("not-a-benchmark")


def test_all_benchmarks_declare_their_default_obs_mode():
    for name, benchmark in BENCHMARKS.items():
        assert benchmark.default_obs_mode in benchmark.supported_obs_modes, f"{name} has an unsupported default."


def test_eval_env_is_seeded_differently_from_training():
    # Otherwise we would be evaluating on exactly the layouts just trained on.
    config = validated()
    train_env, eval_env = make_train_and_eval_envs(config)
    try:
        assert eval_env is not None
        assert config["train_config"]["eval_seed_offset"] != 0
        assert train_env.observation_space.shape == eval_env.observation_space.shape, \
            "The eval env must carry the same observation wrappers as the training env."
    finally:
        train_env.close()
        eval_env.close()


@pytest.mark.parametrize("reseed", [True, False])
def test_every_evaluation_sees_the_same_layouts(monkeypatch, reseed):
    # Without reseeding, each evaluation would continue the eval env's random stream and see new layouts, so
    # differences between checkpoints would be partly due to the layouts rather than the policy.
    import numpy as np
    import stable_baselines3.common.evaluation as sb3_eval
    from rl.logging import RLLog

    first_obs = []

    def fake_evaluate_policy(model, env, **kwargs):
        first_obs.append(env.reset())
        return [1.0], [1]

    monkeypatch.setattr(sb3_eval, "evaluate_policy", fake_evaluate_policy)
    config = validated(train_config={"env": "MiniGrid-DoorKey-8x8-v0"})
    train_cfg = config["train_config"]
    eval_seed = train_cfg["seed"] + train_cfg["eval_seed_offset"]
    log = RLLog(get_benchmark("minigrid"), expected_steps=1, eval_seed=eval_seed if reseed else None)
    eval_env = make_vec_envs(config, n_envs=1, seed=eval_seed, is_eval=True)
    try:
        log.evaluate(None, eval_env)
        log.evaluate(None, eval_env)
    finally:
        eval_env.close()
    assert np.array_equal(first_obs[0], first_obs[1]) == reseed


def test_eval_env_is_omitted_when_evaluation_is_off():
    config = validated(eval_checkpoints=False)
    train_env, eval_env = make_train_and_eval_envs(config)
    try:
        assert eval_env is None
    finally:
        train_env.close()


def test_normalizing_image_observations_is_rejected():
    # VecNormalize would both destroy the uint8 dtype and fight the policy's own /255 scaling.
    from rl_train import validate_config
    with pytest.raises(RuntimeError, match="normalize_obs is only supported"):
        validate_config(smoke_config(train_config={"obs_mode": "symbolic", "normalize_obs": True}),
                        print_config=False)


def test_unknown_environment_is_reported_clearly():
    from rl_train import validate_config
    with pytest.raises(RuntimeError, match="not registered"):
        validate_config(smoke_config(train_config={"env": "MiniGrid-NoSuchEnv-v0"}), print_config=False)


def test_count_bonus_decays_with_visits():
    import gymnasium as gym
    import minigrid  # noqa: F401
    from rl.envs import MiniGridCountBonus
    env = MiniGridCountBonus(gym.make("MiniGrid-Empty-5x5-v0"), coef=0.1)
    env.reset(seed=0)
    left, right = 0, 1
    _, first, *_ = env.step(left)
    _, _, *_ = env.step(right)  # Back to the starting direction, which was never counted.
    _, second, *_ = env.step(left)
    assert first == pytest.approx(0.1)
    assert second == pytest.approx(0.1 / 2 ** 0.5)


def test_count_bonus_is_training_only_and_not_logged():
    # The bonus must reach PPO, but the Monitor's returns (hence success rate) and the eval env stay extrinsic.
    config = validated(train_config={"count_bonus": 0.1})
    train_env, eval_env = make_train_and_eval_envs(config)
    try:
        train_env.reset()
        _, rewards, _, _ = train_env.step([0] * train_env.num_envs)
        assert all(r > 0 for r in rewards), "Every training step should carry a positive bonus."
        monitor = train_env.envs[0].env.env  # Bonus -> ImgObsWrapper -> Monitor.
        assert sum(monitor.rewards) == 0, "The Monitor sits inside the bonus, so it should see no bonus."
        eval_env.reset()
        _, rewards, _, _ = eval_env.step([0])
        assert all(r == 0 for r in rewards), "The eval env should not carry the bonus."
    finally:
        train_env.close()
        eval_env.close()
