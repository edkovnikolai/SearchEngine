#!/bin/bash

#SBATCH --job-name=EmbeddingsCreation         # Job name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Total number of tasks
#SBATCH --cpus-per-task=30        # Number of CPUs per task
#SBATCH --mem=120GB                # Memory per node
#SBATCH --time=1-00:00:00        # Wall time (DD-HH:MM:SS)
#SBATCH --gpus-per-task=2         # 2 v100 gpus per task
#SBATCH --output=slurm-create_embeddings.out    # Where to store output
#SBATCH --error=slurm-create_embeddings.err     # Where to store error of the job

# activating venv
source /home/nikolai/SearchEngine/venv/bin/activate

/home/nikolai/SearchEngine/venv/bin/python /home/nikolai/SearchEngine/create_embeddings.py