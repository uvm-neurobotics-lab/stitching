"""
Reporting a reinforcement learning run through this repo's logging stack.

`RLLog` supplies the evaluation and checkpointing that `BaseLog` leaves abstract, and `RLCallback` drives it from
inside SB3's training loop. Everything is indexed by environment timestep; there are no epochs.

Metric names are deliberately RL-native. The `Train/` and `Eval/` prefixes mark a real distinction: training
episodes come from the stochastic behavior policy in the middle of learning, while evaluation episodes come from a
separately seeded environment under a deterministic policy.
"""
import datetime
from time import time

import numpy as np
import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback

import utils.distributed as dist
from utils import make_pretty
from utils.logging import BaseLog

# How SB3's own logger names the statistics PPO records during an update, and what we call them.
SB3_METRIC_NAMES = {
    "train/loss": "Loss",
    "train/policy_gradient_loss": "Policy Loss",
    "train/value_loss": "Value Loss",
    "train/approx_kl": "Approx KL",
    "train/clip_fraction": "Clip Fraction",
    "train/explained_variance": "Explained Variance",
    "train/learning_rate": "LR",
}


class RLLog(BaseLog):
    """
    The RL counterpart to `StandardLog`.

    `eval_freq` and `save_freq` are in environment timesteps, and must be multiples of the rollout size so that
    `BaseLog.decide_save_and_eval()`'s modulo test lands on them; `algo.check_algo_config()` rounds them for you.
    """

    def __init__(self, benchmark, expected_steps, eval_episodes=20, deterministic_eval=True,
                 metrics_to_print=tuple(), print_freq=1, save_freq=0, eval_freq=0, save_dir=None, model_name="",
                 use_wandb=False, checkpoint_initial_model=True, print_delimiter="\t"):
        super().__init__(metrics_to_print, eval_freq, save_freq, save_dir, model_name, use_wandb,
                         checkpoint_initial_model, False, print_delimiter)
        self.benchmark = benchmark
        self.expected_steps = expected_steps
        self.eval_episodes = eval_episodes
        self.deterministic_eval = deterministic_eval
        self.print_freq = print_freq
        self.records_seen = 0
        if self.use_wandb:
            self.define_wandb_metric("Loss", "last")
            self.define_wandb_metric("Eval/Reward", "max")
            self.define_wandb_metric("Eval/Success Rate", "max")
            self.define_wandb_metric("Train/Reward", "max")
            self.define_wandb_metric("Time/Total", "last")
            self.define_wandb_metric("GPU Mem", "max")
            self.define_wandb_metric("Proc Mem", "max")

    def evaluate(self, sb3_model, eval_env):
        """ Run complete episodes on the evaluation environment and summarize them. """
        from stable_baselines3.common.evaluation import evaluate_policy

        start = time()
        rewards, lengths = evaluate_policy(sb3_model, eval_env, n_eval_episodes=self.eval_episodes,
                                           deterministic=self.deterministic_eval, return_episode_rewards=True,
                                           warn=False)
        successes = [self.benchmark.is_success({"r": r}) for r in rewards]
        return {
            "Eval/Reward": float(np.mean(rewards)),
            "Eval/Episode Length": float(np.mean(lengths)),
            "Eval/Success Rate": float(np.mean(successes)),
            "Time/Eval Total": time() - start,
        }

    def maybe_save_and_eval(self, it, sb3_model, eval_env, config, should_eval=None, should_save=None):
        should_save, should_eval = self.decide_save_and_eval(it, should_eval, should_save)
        metrics = {}

        if should_eval and eval_env is not None:
            metrics.update(self.evaluate(sb3_model, eval_env))
            self.info(f"    Step {it} Eval: Reward: {metrics['Eval/Reward']:.3f}"
                      f"\tSuccess Rate: {metrics['Eval/Success Rate']:.3f}"
                      f"\tEpisode Length: {metrics['Eval/Episode Length']:.1f}"
                      f"\t({self.eval_episodes} episodes in {metrics['Time/Eval Total']:.1f}s)")

        if should_save and dist.is_main_process():
            self.write_checkpoint(self.build_checkpoint(it, sb3_model, config), it)

        if self.start_time is not None:
            metrics["Time/Total"] = time() - self.start_time

        self.record(metrics, it)

    def build_checkpoint(self, it, sb3_model, config):
        """
        Assemble the checkpoint.

        "model" holds the trunk on its own, in the same layout `stitch_train.py` writes, so a trunk trained here can
        be loaded by a supervised config's `ckp_path` and vice versa. "policy" holds the whole SB3 policy, including
        the actor and critic heads, which is what an exact resume needs.
        """
        from rl.models import trunk_state_dict

        return {
            "model": trunk_state_dict(sb3_model),
            "policy": sb3_model.policy.state_dict(),
            "optimizer": sb3_model.policy.optimizer.state_dict(),
            "step": it,
            "config": make_pretty(config),
        }

    def record_rollout(self, it, sb3_model, rollout_time):
        """ Record the training metrics available at the end of a rollout. """
        metrics = {}
        ep_info = list(sb3_model.ep_info_buffer or [])
        if ep_info:
            metrics["Train/Reward"] = float(np.mean([e["r"] for e in ep_info]))
            metrics["Train/Episode Length"] = float(np.mean([e["l"] for e in ep_info]))
            metrics["Train/Success Rate"] = float(np.mean([self.benchmark.is_success(e) for e in ep_info]))

        if rollout_time and rollout_time > 0:
            metrics["Time/Step"] = rollout_time
            metrics["Time/FPS"] = sb3_model.n_steps * sb3_model.n_envs / rollout_time
        if torch.cuda.is_available():
            metrics["GPU Mem"] = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            torch.cuda.reset_peak_memory_stats()
        metrics["Proc Mem"] = psutil.Process().memory_info().rss / 1024.0 / 1024.0

        self.record(metrics, it)
        self.maybe_print(it, metrics)

    def record_update(self, it, sb3_model):
        """
        Record the statistics of the PPO update which just finished.

        PPO writes these to its own logger inside `train()`, which runs *after* the rollout ends, so they are only
        available at the start of the next rollout. `num_timesteps` has not advanced by then, so they land on the
        same row as the rollout they came from.
        """
        values = sb3_model.logger.name_to_value
        metrics = {ours: float(values[theirs]) for theirs, ours in SB3_METRIC_NAMES.items() if theirs in values}
        # SB3 logs the entropy *loss*, which is the negative mean entropy. Report the entropy itself: it is the
        # quantity worth watching, since a collapse toward zero means the policy has stopped exploring.
        if "train/entropy_loss" in values:
            metrics["Entropy"] = -float(values["train/entropy_loss"])
        self.record(metrics, it)

    def maybe_print(self, it, metrics):
        """ Print a progress line periodically. """
        self.records_seen += 1
        if self.print_freq <= 0 or (self.records_seen % self.print_freq and it != 0):
            return
        msg = [f"Step {it}/{self.expected_steps}"]
        if self.start_time is not None and it > 0:
            eta = (time() - self.start_time) / it * (self.expected_steps - it)
            msg.append(f"ETA: {datetime.timedelta(seconds=int(eta))}")
        for key in (self.metrics_to_print or
                    ["Train/Reward", "Train/Success Rate", "Loss", "Entropy", "Time/FPS", "Proc Mem"]):
            if key in self.smoothed_metrics:
                msg.append(f"{key}: {self.smoothed_metrics[key]}")
        self.info(self.delimiter.join(msg))


class RLCallback(BaseCallback):
    """
    Drives `RLLog` from inside SB3's training loop.

    SB3 interleaves the hooks as: `on_rollout_start`, N x `on_step`, `on_rollout_end`, then the algorithm's update,
    then around again. Evaluation and checkpointing happen at rollout boundaries, where the policy is in a
    consistent state; `on_step` is left free for early termination.
    """

    def __init__(self, log, config, eval_env, record_freq):
        super().__init__()
        self.log = log
        self.config = config
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.last_record_step = 0
        self.rollout_start_time = None
        self.pending_update_step = None

    def _on_training_start(self):
        self.log.begin(self.model, self.eval_env, self.config)

    def _on_rollout_start(self):
        # Collect the previous update's statistics, which only became available after the last rollout ended.
        if self.pending_update_step is not None:
            self.log.record_update(self.pending_update_step, self.model)
            self.pending_update_step = None
        self.rollout_start_time = time()

    def _on_step(self):
        return True  # Reserved for early termination; returning False would stop training.

    def _on_rollout_end(self):
        it = self.model.num_timesteps
        if it - self.last_record_step >= self.record_freq:
            self.last_record_step = it
            self.pending_update_step = it
            rollout_time = time() - self.rollout_start_time if self.rollout_start_time else None
            self.log.record_rollout(it, self.model, rollout_time)
        self.log.maybe_save_and_eval(it, self.model, self.eval_env, self.config)

    def _on_training_end(self):
        # The final update's statistics have no following rollout to be picked up by.
        if self.pending_update_step is not None:
            self.log.record_update(self.pending_update_step, self.model)
            self.pending_update_step = None
