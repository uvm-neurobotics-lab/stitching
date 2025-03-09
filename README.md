# Neural Network Stitching

What is possible with neural network stitching?

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

Configuration and output of all experiments will live in the `experiments/` folder.

For now, each experiment will consist of the stitching of two networks. For initial experiments, instead of stitching
two separate networks we will first knock out some layer(s) of a single network and replace them with new stitching
layer(s). Organization will be as follows:
 - `experiments/`
   - `<project name>/`
     - `<experiment name>/`
       - `config.yml`
       - `traj.pkl`

Where `config.yml` is the experiment configuration and `traj.pkl` is a pickled Pandas dataframe describing the
stitch training trajectory.

# Experimental Procedure

 1. Load a configured set of subnets using [`utils.subgraphs.create_sub_network()`](src/utils/subgraphs.py).
 1. Construct a network with configured stitching modules in between each subnet.
 1. Train the stitching module(s) for a configured number of epochs using a configured optimizer.
 1. Write the training trajectory to a dataframe on disk.
