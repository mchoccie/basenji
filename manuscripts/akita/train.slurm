#!/bin/bash
#SBATCH --job-name=akita_train
#SBATCH --output=akita_train.out
#SBATCH --error=akita_train.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --time=48:00:00

# Load modules (ensure correct CUDA version)
# module load cuda/11.6  # Adjust if needed
# module load nvidia/cudnn/8.4

# Load Conda properly (check for correct path)
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust if needed
conda activate basenji  # Activate Conda environment

# Create output directory if missing
mkdir -p ./data/1m/train_out/

# Debugging: Print GPU availability
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# Run training script
python /home/017448899/basenji/bin/akita_train.py -k -o ./data/1m/train_out/ ./data/1m/params_tutorial.json ./data/1m

