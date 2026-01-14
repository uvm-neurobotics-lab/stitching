#!/bin/bash
###
# A script to download datasets for vision model merging.
#
# These are some of the datasets used by the Task Vectors paper, and subsequent papers, to evaluate model merging. The
# rest of the datasets used by this line of work can be acquired directly through Torchvision, but these few require
# manual setup. This script automates most of the setup, followed by `configure_merge_vision_datasets.py`.
# Task Vectors: https://github.com/mlfoundations/task_vectors/
# Dataset issues: https://github.com/mlfoundations/task_vectors/issues/1
#
# TO RUN:
#
# Before running, you must do the following:
#  - Ensure the `kaggle` CLI can run with your Kaggle credentials.
#      - One option is to provide your KAGGLE_USERNAME and KAGGLE_KEY in the terminal or in the ./data/.env file (see
#        .env.example).
#  - Manually download this file and store it in ./data: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
#  - Manually download SUN397 from this torrent: https://hyper.ai/en/datasets/5367
#    (This can be done either before or after running this script.)
#
# After running, run the following to complete setup:
#  - python src/configure_merge_vision_datasets.py
###

# Exit immediately if any command has a non-zero return code.
set -e
set -o pipefail

cd data

if [ -f .env ]; then
  source .env
fi
if [[ -z "$(which kaggle)" ]]; then
  pip install kaggle
fi


# stanford cars dataset (ref: https://github.com/pytorch/vision/issues/7545#issuecomment-2282674373)
mkdir -p stanford_cars
if [ -z "$(ls -A stanford_cars)" ]; then  # Only proceed if directory is empty.
  echo """
==============================================================================
Downloading Stanford Cars
==============================================================================
"""
  kaggle datasets download --unzip rickyyyyyyy/torchvision-stanford-cars
else
  echo """
==============================================================================
Skipping Stanford Cars - Already present
==============================================================================
"""
fi

# FER2013 Dataset
mkdir -p fer2013
if [ -z "$(ls -A fer2013)" ]; then  # Only proceed if directory is empty.
  echo """
==============================================================================
Downloading FER2013
==============================================================================
"""
  cd fer2013
  kaggle datasets download --unzip msambare/fer2013
  # NOTE: Use the code below if you prefer to use the original dataset from the Kaggle competition. This is the format
  # supported by Torchvision. But it's honestly kind of a mess. And you need to join the competition on Kaggle and
  # accept the terms before you can download: https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data
#  kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
#  unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip
#  rm challenges-in-representation-learning-facial-expression-recognition-challenge.zip
  cd ..
else
  echo """
==============================================================================
Skipping FER2013 - Already present
==============================================================================
"""
fi

# resisc45
# NOTE: Manually download this before running: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
mkdir -p resisc45
if [ -z "$(ls -A resisc45)" ]; then  # Only proceed if directory is empty.

  if [ -f "./NWPU-RESISC45.rar" ]; then
    echo """
==============================================================================
Unpacking RESISC45
==============================================================================
"""
  cd resisc45
  mv ../NWPU-RESISC45.rar ./
  if [[ "$(uname -s)" == "Darwin" ]]; then
    brew install sevenzip
  else
    # NOTE: Assuming this is a server where we don't have global install privileges.
    conda install p7zip
  fi
  7z x NWPU-RESISC45.rar
  wget -O resisc45-train.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-train.txt"
  wget -O resisc45-val.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-val.txt"
  wget -O resisc45-test.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-test.txt"
  rm -rf NWPU-RESISC45.rar
  cd ..
else
  echo """
==============================================================================
WARNING: RESISC45 file not found
==============================================================================
"""
  fi

else
    echo """
==============================================================================
Skipping RESISC45 - Already present
==============================================================================
"""
fi

# eurosat
mkdir -p eurosat
if [ -z "$(ls -A eurosat)" ]; then  # Only proceed if directory is empty.
  echo """
==============================================================================
Downloading EuroSAT
==============================================================================
"""
  cd eurosat
  wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSAT.zip
  unzip EuroSAT.zip
  rm -rf EuroSAT.zip
  cd ..
else
  echo """
==============================================================================
Skipping EuroSAT - Already present
==============================================================================
"""
fi

# sun397
# NOTE: SUN397 is no longer available from the original host. Instead, manually download and extract it using the
# torrent at this link: https://hyper.ai/en/datasets/5367
#mkdir sun397
#if [ -z "$(ls -A sun397)" ]; then  # Only proceed if directory is empty.
#  echo """
#==============================================================================
#Skipping SUN397
#==============================================================================
#"""
#  cd sun397
#  wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
#  unzip Partitions.zip
#  tar -xvzf SUN397.tar.gz
#  rm -rf SUN397.tar.gz
#else
#  echo """
#==============================================================================
#Skipping SUN397 - Already present
#==============================================================================
#"""
#fi
