#!/bin/bash
#SBATCH --job-name=jupyter_gpu      # Job name
#SBATCH --partition=gpu             # Partition name (adjust for your cluster)
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           # Request 4 CPU cores
#SBATCH --mem=16G                   # Request 16GB RAM
#SBATCH --time=04:00:00             # Time limit (4 hours)
#SBATCH --output=jupyter_gpu.out    # Output file
#SBATCH --error=jupyter_gpu.err     # Error file

# Load required modules
module load cuda/11.5               # Load CUDA module (adjust version as needed)

# Activate your Conda environment
source ~/miniconda3/bin/activate basenji

# Get the node's IP address
ipnport=$(shuf -i8000-9999 -n1)     # Random port number
ipnip=$(hostname -I | awk '{print $1}')

# Start Jupyter Notebook
echo "Starting Jupyter Notebook on $ipnip:$ipnport"
jupyter notebook --no-browser --port=$ipnport --ip=$ipnip