"""
Tests for recording a trained policy to a video.
"""
import os

import pytest

import rl_render
from rl_train import main as train_main

CONFIG = "tests/rl-ppo-conv4-minigrid.yml"


def trained_run(tmp_path):
    """ Produce a real (if tiny) run directory, including a checkpoint with a policy in it. """
    os.environ["WANDB_MODE"] = "disabled"
    assert train_main(["-c", CONFIG, "--st", "--env", "MiniGrid-Empty-5x5-v0", "-o", str(tmp_path)]) == 0
    return tmp_path


def test_training_records_a_video_by_default(tmp_path):
    # A video of the final policy should be something you get without asking.
    run_dir = trained_run(tmp_path)
    videos = list((run_dir / "video").glob("*.mp4"))
    assert len(videos) == 1, f"Expected training to leave one video, got {videos}"
    assert videos[0].stat().st_size > 0


def test_training_video_can_be_turned_off(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    assert train_main(["-c", CONFIG, "--st", "--env", "MiniGrid-Empty-5x5-v0", "-o", str(tmp_path),
                       "--no-video"]) == 0
    assert not (tmp_path / "video").exists()
    assert (tmp_path / "result.pkl").is_file(), "Turning off video must not affect the results."


def test_a_broken_recorder_does_not_lose_the_run(monkeypatch, tmp_path):
    # Training has finished and its results are already on disk by then, so recording is best-effort.
    import rl.video

    def boom(*args, **kwargs):
        raise RuntimeError("no encoder available")

    monkeypatch.setattr(rl.video, "record_policy", boom)
    os.environ["WANDB_MODE"] = "disabled"
    assert train_main(["-c", CONFIG, "--st", "--env", "MiniGrid-Empty-5x5-v0", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "result.pkl").is_file()
    assert (tmp_path / "checkpoint.pth").is_file()


def test_records_a_video_from_a_run_directory(tmp_path):
    run_dir = trained_run(tmp_path)
    before = set((run_dir / "video").glob("*.mp4"))  # Training already left one behind.
    assert rl_render.main([str(run_dir), "-n", "2", "--fps", "4"]) == 0

    added = set((run_dir / "video").glob("*.mp4")) - before
    assert len(added) == 1, f"Expected one new video, got {added}"
    assert next(iter(added)).stat().st_size > 0


def test_rerendering_does_not_overwrite_earlier_videos(tmp_path):
    # Recording the same checkpoint again should keep both videos, so a render is never silently lost.
    run_dir = trained_run(tmp_path)  # Leaves MiniGrid-Empty-5x5-v0.mp4.
    for _ in range(3):
        rl_render.main([str(run_dir), "-n", "1"])

    videos = sorted(p.name for p in (run_dir / "video").glob("*.mp4"))
    assert videos == ["MiniGrid-Empty-5x5-v0-1.mp4", "MiniGrid-Empty-5x5-v0-2.mp4",
                      "MiniGrid-Empty-5x5-v0-3.mp4", "MiniGrid-Empty-5x5-v0.mp4"], \
        "Each recording should get its own file, suffixed to avoid a collision."


def test_records_a_specific_checkpoint(tmp_path):
    # Rendering an earlier checkpoint is how you watch a policy improve over training.
    run_dir = trained_run(tmp_path)
    checkpoint = next(run_dir.glob("model-*.pth"))
    assert rl_render.main([str(run_dir), "--checkpoint", checkpoint.name, "-n", "1"]) == 0
    assert (run_dir / "video" / f"{checkpoint.stem}.mp4").is_file()


def test_output_location_is_configurable(tmp_path):
    run_dir = trained_run(tmp_path / "run")
    out = tmp_path / "elsewhere"
    assert rl_render.main([str(run_dir), "-o", str(out), "-n", "1"]) == 0
    assert list(out.glob("*.mp4"))


def test_rejects_a_checkpoint_with_no_policy(tmp_path):
    # A checkpoint from stitch_train.py holds only a trunk; say so rather than failing on a missing key.
    import torch

    run_dir = trained_run(tmp_path)
    supervised = run_dir / "supervised.pth"
    torch.save({"model": {}, "epoch": 1}, supervised)
    with pytest.raises(SystemExit):
        rl_render.main([str(run_dir), "--checkpoint", supervised.name])


def test_reports_per_episode_outcomes(tmp_path):
    # The episode summary is the point of the tool as much as the video is.
    from rl.envs import get_benchmark

    from rl.video import report

    results = [{"reward": 0.9, "length": 4, "success": True}, {"reward": 0.0, "length": 64, "success": False}]
    report(results)  # Should not raise on a mix of solved and failed episodes.
    report(results, per_episode=False)
    assert get_benchmark("minigrid").is_success({"r": 0.9})
