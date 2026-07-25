# What is Possible with Neural Network Stitching?

In this work, we seek to push the boundaries of neural network stitching to see what it can and can't do.

# Setup

You can install necessary dependencies using the provided environment file:
```shell
conda env create -f environment.yml
conda activate stitch
```
However, many users will need to install PyTorch manually, based on their specific system configuration. In that case,
 1. Create an environment (using your preferred Python version): `conda create -n stitch python=3.11`
 1. Activate: `conda activate stitch`
 1. [Install PyTorch **and** Torchvision](https://pytorch.org/get-started/locally/) first.
 1. Manually install the rest of the packages listed in [environment.yml](environment.yml). Install `conda` packages
    before `pip` packages.
    - _Note:_ The `wandb` package comes from the `conda-forge` channel: `conda install wandb -c conda-forge`

For convenience, you may consider setting up a symlink to the folder that contains your datasets. Otherwise you must
specify the `--data-path` when you run. See the section on [Datasets](#datasets).

## Datasets

By default, we will look for datasets in a `data/` folder in the root of the repository. Before you begin, you should
create this folder or a symlink to your actual dataset folder. For instance:
```shell
cd stitching
ln -s ~/datasets ./data
```

For the basic stitching experiments from the original paper (_"Bridging Large Gaps in Neural Network Representations
with Model Stitching"_), you will only need ImageNet-1k. This should be located at `data/imagenet/`.

For model merging, do the following to set up datasets that are commonly used by model merging papers:
1. Follow the prerequisite steps described at the top of [src/download-merge-vision-datasets.sh](src/download-merge-vision-datasets.sh).
1. Activate your Python environment if you haven't already.
1. Run: `src/download-merge-vision-datasets.sh`
1. Run: `python src/configure_merge_vision_datasets.py`

To use geospatial vision datasets, do the following:
1. Follow the prerequisite steps described at the top of [src/download-geospatial-datasets.sh](src/download-geospatial-datasets.sh).
1. Activate your Python environment if you haven't already.
1. Run: `src/download-geospatial-datasets.sh`
1. Run: `python src/configure_geospatial_datasets.py`


# Organization

- The executable for a single stitching job is [`src/stitch_train.py`](src/stitch_train.py).
  - Example configs can be found in [`tests/`](tests).
- The same architectures can be trained with reinforcement learning instead, using
  [`src/rl_train.py`](src/rl_train.py). See [Run an RL Job](#run-an-rl-job) below.
- There is also a script which tests stitching across different layer depths and different architecture combinations:
  [`src/launch_scaling_experiments.py`](src/launch_scaling_experiments.py).
  - This script requires access to a Slurm cluster as it will launch an array of jobs with different configurations.
  - Example configs can be found in [`across-scales/`](across-scales).
  - Once all experimental results are generated, we use [`src/across-scales.ipynb`](src/across-scales.ipynb) to
    post-process the results and generate all our plots.

# Run a Stitching Job

To test model stitching, you can run [`src/stitch_train.py`](src/stitch_train.py).
  - This will run a single stitching job. See examples at the top of the file.
  - It will generate a pickled dataframe (`result.pkl`) which logs each training step, and (optionally) model checkpoints. 
  - See [`tests/`](tests) for a list of example configs that can be executed with `stitch_train.py`. This will give you
    a sense for the wide range of possible configurations.
    - You can even train models from scratch instead of stitching pre-trained models (see
      [`tests/train-resnet18.yml`](tests/train-resnet18.yml)).

A single stitching job consists of the following steps:
 1. Load a configured set of subnets using [`utils.subgraphs.create_sub_network()`](src/utils/subgraphs.py).
 1. Construct a network with configured stitching modules in between each subnet.
 1. Train the stitching module(s) for a configured number of epochs using a configured optimizer.
 1. Write the training trajectory to a dataframe on disk (`result.pkl`).

We recommend you create a subfolder `experiments/<my-experiment-name>` for each experiment. Copy the config here and
edit as needed. Then, run from this folder (e.g., `python ../../src/stitch_train.py -c ./config.yml`). This means the
results and all checkpoints will be neatly packaged together with the config that was used to generate them.

You can also run on a Slurm cluster, by customizing one of our example `*.sbatch` files. From the experiment folder,
run `sbatch /<full-path-to>/stitching/nvtrain.sbatch stitchup /<full-path-to>/stitching/src/stitch_train.py --config config.yml`.

# Run an RL Job

[`src/rl_train.py`](src/rl_train.py) trains the same architectures with reinforcement learning, on MiniGrid and
BabyAI environments. It is the sibling of `stitch_train.py`: the same config format, the same model construction, and
the same result files. Three things differ.

- **An environment replaces the dataset.** Instead of `dataset`, the config names a `benchmark` and an `env`, e.g.
  `--benchmark minigrid --env BabyAI-GoToRedBallNoDists-v0`. `obs_mode` chooses how observations are presented:
  `symbolic` (the native 7x7x3 grid codes), `rgb` (the agent's view rendered to pixels, the mode where a pretrained
  image backbone is meaningful), `rgb_full`, or `flat`.
- **The model config is split in two.** `trunk` is the feature extractor this repo stitches, written exactly as
  `model` is for supervised training -- and in fact `model` and `assembly` are accepted as aliases, so a supervised
  config can be handed to `rl_train.py` unchanged. `policy` describes the actor-critic heads that Stable-Baselines3
  builds on top of the resulting feature vector.
- **Progress is measured in environment timesteps, not epochs.** `total_timesteps` replaces `epochs`, and
  `eval_freq`, `save_freq`, and `record_freq` are all counted in timesteps. The resulting `result.pkl` is indexed by
  `Step`, with no `Epoch` column; read it with `utils.postprocess.last_step_only()` rather than `last_epoch_only()`.

Try it with:

```bash
WANDB_MODE=disabled python src/rl_train.py -c tests/rl-ppo-conv4-minigrid.yml --st
```

Two example configs are provided: [`tests/rl-ppo-conv4-minigrid.yml`](tests/rl-ppo-conv4-minigrid.yml) is the
reference setup, a small ConvNet trained from scratch on symbolic observations, and
[`tests/rl-ppo-mlp-minigrid-flat.yml`](tests/rl-ppo-mlp-minigrid-flat.yml) reproduces RL Zoo's tuned MiniGrid
configuration, which is the one to compare against when checking the algorithm itself.

## Watch a Trained Policy

[`src/rl_render.py`](src/rl_render.py) replays a saved policy and records it to an mp4, reporting how each episode
went. Point it at a run directory:

```bash
python src/rl_render.py experiments/poc/go-to-red-ball-nodists -n 8
```

The frames come from Gymnasium itself (MiniGrid draws them, and `render_mode="rgb_array"` hands them over as
arrays); Stable-Baselines3's `VecVideoRecorder` encodes them, which is the same path RL Baselines3 Zoo uses. It
needs `moviepy`, which is in `environment.yml`.

Pass `--checkpoint` to watch an earlier stage of training, which is the easiest way to see a policy improve:

```bash
python src/rl_render.py experiments/poc/go-to-red-ball-nodists --checkpoint model-20480.pth -n 8
```

By default the environment is seeded with the run's *evaluation* seed, so you are watching layouts the agent was
scored on rather than ones it trained on. Use `--seed` to pick your own, and `--stochastic` to sample from the
policy the way it behaved while training instead of taking its most likely action.

## Notes

Note that RL parallelizes by stepping many environments at once (`--n-envs`), **not** by DDP. Do not launch
`rl_train.py` under `torchrun` with more than one process; it would start several identical runs competing to write
the same output files, and the script refuses to run if it detects one.

# Run a Sweep Over Stitching Gaps and Adapters

Each config in [`across-scales/`](across-scales) defines all the jobs for a single pair of architectures. The two given
architectures (`src_stages` and `dest_stages`) are stitched in a number of different ways. A Slurm job is launched for
each different way. See examples at the top of [`src/launch_scaling_experiments.py`](src/launch_scaling_experiments.py).

# Citation

If you use this work, please cite as:
```
@inproceedings{traft2025bridging,
  title={Bridging Large Gaps in Neural Network Representations with Model Stitching},
  author={Traft, Neil and Cheney, Nick},
  booktitle={Proceedings of UniReps: the Third Edition of the Workshop on Unifying Representations in Neural Models},
  year={2025},
  organization={PMLR}
}
```