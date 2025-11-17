# Bridging Large Gaps in Neural Network Representations with Model Stitching

This is the code repository for the paper published in the UniReps Workshop @ NeurIPS 2025: **Neil Traft and Nick
Cheney. "Bridging Large Gaps in Neural Network Representations with Model Stitching".** (link forthcoming)

**What is possible with neural network stitching?** In this work, we empirically investigate whether very structurally
different layers can be stitched together, and suggest some new techniques for doing so.

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
specify the `--data-path` when you run. For instance:
```shell
cd stitching
ln -s ~/datasets ./data
```

# Organization

- The executable for a single stitching job is [`src/stitch_train.py`](src/stitch_train.py).
  - Example configs can be found in [`tests/`](tests).
- The script to reproduce our experiments by launching an array of stitching jobs is
  [`src/launch_scaling_experiments.py`](src/launch_scaling_experiments.py) (but this requires access to a Slurm cluster).
  - Example configs can be found in [`across-scales/`](across-scales).
  - Unfortunately it would take quite a lot of Slurm jobs to fully reproduce our results. But hopefully this code at
    least gives some clarity as to how it works.
- Once all experimental results were generated, we used [`src/across-scales.ipynb`](src/across-scales.ipynb) to
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