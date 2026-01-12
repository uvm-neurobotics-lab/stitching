#!/bin/bash
###
# A script to download datasets for satellite imagery tasks.
#
# TO RUN:
#
# Before running, you must do the following:
#  - Ensure the `kaggle` CLI can run with your Kaggle credentials.
#      - One option is to provide your KAGGLE_USERNAME and KAGGLE_KEY in the terminal or in the ./data/.env file (see
#        .env.example).
#  - Ensure the `aws` CLI is installed and logged in.
#      - Installation: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
#      - Login: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html
#
# After running, run the following to complete setup:
#  - python src/configure_geospatial_datasets.py
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
if [[ -z "$(which aws)" ]]; then
  echo "Error: AWS command-line app not found. Please install before running."
  exit 1
fi


# Aerial Image Dataset
# https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets
if [ ! -d AID ] || [ -z "$(ls -A AID)" ]; then  # Only proceed if directory is empty.
  echo """
==============================================================================
Downloading AID
==============================================================================
"""
  kaggle datasets download --unzip jiayuanchengala/aid-scene-classification-datasets
else
  echo """
==============================================================================
Skipping AID - Already present
==============================================================================
"""
fi

# UC Merced Land Use Dataset
# http://weegee.vision.ucmerced.edu/datasets/landuse.html
if [ ! -d UCMerced_LandUse ] || [ -z "$(ls -A UCMerced_LandUse)" ]; then  # Only proceed if directory is empty.
  echo """
==============================================================================
Downloading UCM Land Use
==============================================================================
"""
  wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
  unzip UCMerced_LandUse.zip
  rm UCMerced_LandUse.zip
else
  echo """
==============================================================================
Skipping UCM Land Use - Already present
==============================================================================
"""
fi

# Functional Map of the World (FMoW) Dataset
# https://github.com/fMoW/dataset?tab=readme-ov-file
mkdir -p fmow
if [ -z "$(ls -A fmow)" ]; then  # Only proceed if directory is empty.
  echo """
==============================================================================
Downloading FMoW
==============================================================================
"""
  aws s3 cp s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/ ./data/fmow/ --recursive
else
  echo """
==============================================================================
Skipping FMoW - Already present
==============================================================================
"""
fi
