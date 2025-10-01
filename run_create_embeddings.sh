#!/bin/bash

#SBATCH --job-name=EmbeddingsCreation         # Job name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Total number of tasks
#SBATCH --cpus-per-task=28        # Number of CPUs per task
#SBATCH --mem=64GB                # Memory per node
#SBATCH --time=1-00:00:00        # Wall time (DD-HH:MM:SS)

/home/nikolai/SearchEngine/venv/bin/python /home/nikolai/SearchEngine/create_embeddings.py