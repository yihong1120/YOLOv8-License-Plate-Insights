#!/bin/bash
# A shell script to set up an environment for a Car License Plate recognition project.

# Define the .zip filename.
filename="Car-License-Plate.zip"

# Remove the .zip extension to get the folder name.
foldername="${filename%.*}"

# Install the essential packages
! pip install -r requirements.txt

# Execute the Python scripts for preparing the dataset and training the model.
python3 xml2txt.py
python3 dataset_preparation.py
python3 data_augmentation.py
python3 train.py
python3 inference.py