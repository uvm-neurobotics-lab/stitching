#!/bin/bash
###
# A script to download datasets for satellite imagery tasks.
#
# TO RUN:
#
# Before running, you must do the following:
#  - Set your KAGGLE_USERNAME and KAGGLE_KEY in the terminal or in the ./data/.env file (see .env.example).
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
if [[ -z "$KAGGLE_KEY" ]]; then
  echo "Error: KAGGLE_KEY is not set or is empty."
  echo "Please create a .env file in the data/ folder with your Kaggle API key. See .env.example for an example."
  exit 1
fi

pip install kaggle


# Aerial Image Dataset
# https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets
if [ -z "$(ls -A AID)" ]; then  # Only proceed if directory is empty.
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
if [ -z "$(ls -A UCMerced_LandUse)" ]; then  # Only proceed if directory is empty.
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
