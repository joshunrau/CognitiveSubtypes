#!/bin/bash
#SBATCH --account=def-mlepage
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1.5G
#SBATCH --time=1:00:00

source venv/bin/activate
python src/main.py
