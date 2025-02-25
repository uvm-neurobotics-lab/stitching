# Neural Network Stitching

What is possible with neural network stitching?

# Setup

Install necessary dependencies using the provided environment file:
```
conda create -f environment.yml
conda activate stitch
```
Or manually install the packages listed there.

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
