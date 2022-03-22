#!/bin/bash
#SBATCH --account=def-mlepage
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1:00:00

source venv/bin/activate
python src/main.py
