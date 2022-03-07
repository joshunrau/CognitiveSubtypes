#!/bin/bash
#SBATCH --account=def-mlepage
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1.5G
#SBATCH --time=1:00:00

venv/bin/python -c "from data.dataset import Dataset; Dataset.make().write_csv()"