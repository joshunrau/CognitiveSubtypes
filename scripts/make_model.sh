#!/bin/bash
#SBATCH --account=def-mlepage
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1.5G
#SBATCH --time=1:00:00

source venv/bin/activate
fit_model "$1"