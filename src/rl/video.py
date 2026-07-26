"""
Recording a policy to a video.

Used both at the end of a training run and by `rl_render.py` to replay a saved checkpoint, so that a video of the
final policy is something you get by default rather than something you have to remember to ask for.

The frames come from Gymnasium itself -- MiniGrid draws them, and `render_mode="rgb_array"` hands them over as
arrays -- and Stable-Baselines3's VecVideoRecorder encodes them, which is the same path RL Baselines3 Zoo uses.
"""
import logging
from pathlib import Path

import numpy as np

import rl.envs as envs
from rl.envs import get_benchmark


def unique_path(video_dir, name_prefix):
    """ A path which does not exist yet, so a new recording never overwrites an earlier one. """
    path = Path(video_dir) / f"{name_prefix}.mp4"
    index = 0
    while path.is_file():
        index += 1
        path = Path(video_dir) / f"{name_prefix}-{index}.mp4"
    return path


def rollout(sb3_model, venv, benchmark, episodes, deterministic=True):
    """
    Step the environment until `episodes` episodes have finished.

    Returns:
        list: One dict per episode, with its return, length, and whether it succeeded.
    """
    obs = venv.reset()
    results = []
    ep_reward, ep_length = 0.0, 0
    while len(results) < episodes:
        action, _ = sb3_model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, _ = venv.step(action)
        ep_reward += float(rewards[0])
        ep_length += 1
        if dones[0]:
            results.append({"reward": ep_reward, "length": ep_length,
                            "success": bool(benchmark.is_success({"r": ep_reward}))})
            ep_reward, ep_length = 0.0, 0
    return results


def report(results, per_episode=True):
    """ Log how each episode went, and the summary over all of them. """
    if per_episode:
        for i, ep in enumerate(results, start=1):
            outcome = "solved" if ep["success"] else "FAILED"
            logging.info(f"  Episode {i}: {outcome} in {ep['length']} steps, return {ep['reward']:.3f}")
    logging.info(f"Over {len(results)} episodes: "
                 f"success rate {np.mean([e['success'] for e in results]):.2f}, "
                 f"mean return {np.mean([e['reward'] for e in results]):.3f}, "
                 f"mean length {np.mean([e['length'] for e in results]):.1f}")


def record_policy(sb3_model, config, video_dir, episodes=5, fps=8, deterministic=True, seed=None, name_prefix=None,
                  per_episode_report=True):
    """
    Record episodes of the given policy to an mp4.

    Args:
        sb3_model: An already-constructed algorithm whose policy should be recorded.
        config: The full config, used to rebuild a single environment with rendering turned on.
        video_dir: Directory to write the video into. Created if needed.
        episodes: How many complete episodes to record.
        fps: Frames per second of the resulting video.
        deterministic: Whether to take the policy's most likely action rather than sampling.
        seed: (Optional) Environment seed. Defaults to the run's evaluation seed, so the layouts are ones the agent
            was scored on rather than ones it trained on.
        name_prefix: (Optional) Base name for the video file. Defaults to the environment id.
        per_episode_report: Whether to log a line per episode as well as the summary.
    Returns:
        Path: The video that was written, or None if nothing was recorded.
    """
    from stable_baselines3.common.vec_env import VecVideoRecorder

    train_cfg = config["train_config"]
    if seed is None:
        seed = train_cfg["seed"] + train_cfg["eval_seed_offset"]
    video_dir = Path(video_dir)
    name_prefix = name_prefix or train_cfg["env"]
    final_path = unique_path(video_dir, name_prefix)

    # One environment, so the video follows a single agent. The policy is unaffected by how many environments it is
    # asked to act on, so the model built for training can be recorded as-is.
    venv = envs.make_vec_envs(config, n_envs=1, seed=seed, render_mode="rgb_array")
    try:
        # `video_length` is a step budget, and we cannot know one in advance: an episode runs until the agent solves
        # it or times out, and a BabyAI level does not even fix its own step limit until reset(), when the limit is
        # derived from the mission it just generated. So give the recorder an effectively unlimited budget and let
        # the rollout loop decide when to stop; closing the recorder is what writes the file.
        venv = VecVideoRecorder(venv, str(video_dir), record_video_trigger=lambda step: step == 0,
                                video_length=10 ** 9, name_prefix=name_prefix)
        # Read from the environment's metadata at construction, so it has to be overridden afterwards.
        venv.frames_per_sec = fps

        logging.info(f"Recording {episodes} episodes of {train_cfg['env']} "
                     f"({'deterministic' if deterministic else 'stochastic'} policy, seed {seed}).")
        results = rollout(sb3_model, venv, get_benchmark(train_cfg["benchmark"]), episodes, deterministic)
        report(results, per_episode=per_episode_report)
        recorded_path = Path(venv.video_path)
    finally:
        venv.close()  # Closing is what flushes the video to disk.

    if not recorded_path.is_file():
        logging.warning(f"No video was written to {video_dir}.")
        return None
    # The recorder names the file after the step budget it was given, which is a sentinel here. Rename it to the
    # unique name chosen above, so the video is identifiable and no earlier recording is overwritten.
    recorded_path.replace(final_path)
    logging.info(f"Wrote {final_path}")
    return final_path


def record_after_training(config, sb3_model, result_file):
    """
    Record the final policy at the end of a training run.

    Deliberately best-effort: training has already finished and its results are already on disk by the time this
    runs, so a problem here (a missing encoder, a headless machine) must not take the run down with it.

    Returns:
        Path: The video that was written, or None if recording was disabled or failed.
    """
    train_cfg = config["train_config"]
    if not config.get("record_video", True):
        return None

    video_dir = Path(config.get("save_dir") or Path(result_file).parent) / "video"
    try:
        return record_policy(sb3_model, config, video_dir,
                             episodes=train_cfg.get("video_episodes", 5),
                             fps=train_cfg.get("video_fps", 8),
                             per_episode_report=False)
    except Exception as e:
        logging.warning(f"Could not record a video of the final policy ({type(e).__name__}: {e}). Training results "
                        f"are unaffected; use rl_render.py to record one later.")
        return None
