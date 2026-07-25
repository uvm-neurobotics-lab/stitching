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


def test_records_a_video_from_a_run_directory(tmp_path):
    run_dir = trained_run(tmp_path)
    assert rl_render.main([str(run_dir), "-n", "2", "--fps", "4"]) == 0

    videos = list((run_dir / "video").glob("*.mp4"))
    assert len(videos) == 1, f"Expected exactly one video, got {videos}"
    assert videos[0].stat().st_size > 0


def test_rerendering_replaces_rather_than_accumulates(tmp_path):
    # The recorder names files after the step budget it was given, so without intervention every run would leave a
    # differently-named file behind.
    run_dir = trained_run(tmp_path)
    rl_render.main([str(run_dir), "-n", "1"])
    rl_render.main([str(run_dir), "-n", "1"])
    assert len(list((run_dir / "video").glob("*.mp4"))) == 1


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

    results = [{"reward": 0.9, "length": 4, "success": True}, {"reward": 0.0, "length": 64, "success": False}]
    rl_render.report(results)  # Should not raise on a mix of solved and failed episodes.
    assert get_benchmark("minigrid").is_success({"r": 0.9})
