#!/bin/bash
###
# A script to download datasets for vision model merging.
#
# NOTE: Before running, manually download this file: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
#
# These are some of the datasets used by the Task Vectors paper, and subsequent papers, to evaluate model merging. The
# rest of the datasets used by this line of work can be acquired directly through Torchvision, but these few require
# manual setup. These scripts automate most of the setup.
# Task Vectors: https://github.com/mlfoundations/task_vectors/
# Dataset issues: https://github.com/mlfoundations/task_vectors/issues/1
###

# Exit immediately if any command has a non-zero return code.
set -e
set -o pipefail

pip install kaggle
cd data
export KAGGLE_USERNAME=neiltraft
export KAGGLE_KEY=550514a731a19c14bdcf685830db9801

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
Skipping Stanford Cars
==============================================================================
"""
fi

# resisc45
mkdir -p resisc45
if [ -z "$(ls -A resisc45)" ]; then  # Only proceed if directory is empty.
  echo """
==============================================================================
Downloading RESISC45
==============================================================================
"""
  cd resisc45
  # NOTE: Manually download this before running: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
  mv ~/Downloads/NWPU-RESISC45.rar ./
  if [[ "$(uname -s)" == "Darwin" ]]; then
    brew install unar
  else
    sudo apt -y install unar
  fi
  unar NWPU-RESISC45.rar
  wget -O resisc45-train.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-train.txt"
  wget -O resisc45-val.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-val.txt"
  wget -O resisc45-test.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-test.txt"
  rm -rf NWPU-RESISC45.rar
  cd ..
else
  echo """
==============================================================================
Skipping RESISC45
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
Skipping EuroSAT
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
#Skipping SUN397
#==============================================================================
#"""
#fi
