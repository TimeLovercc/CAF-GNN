#!/bin/bash

# Install Anaconda3
# cd ~
# wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
# bash Anaconda3-2023.07-2-Linux-x86_64.sh
# This need to type in mannualy, yes, enter, yes. Then reopen

# Make sure the correct conda, make sure they are in .bashrc
export PATH=~/anaconda3/bin:$PATH
# export PATH=/home/zzg5107/anaconda3/envs/pytorch/bin/:$PATH

# Create a new conda environment named pytorch with Python 3.9
conda create -y -n pytorch python=3.9

# Initialize conda for bash (Note: This may need to be run manually in some cases)
conda init bash
source ~/.bashrc

# Activate the newly created pytorch environment
conda activate pytorch

# Install PyTorch 2.0.0 with CUDA 11.8 support
pip install https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp39-cp39-linux_x86_64.whl
echo "Setup completed. PyTorch environment is ready!"
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install pytorch_lightning==2.0.4
pip install wandb
