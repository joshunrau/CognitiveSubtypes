#!/bin/bash
#SBATCH --account=def-mlepage
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1.5G
#SBATCH --time=1:00:00

cd "$1"

if [ $? -eq 0 ]; then
    echo "Set working directory: ${1}"
else
    "Failed to set working directory: ${1}"
    exit 1
fi

command_exists() {
    command -v "$1" >/dev/null 2>&1;
}

if ! command_exists virtualenv; then
    echo ERROR: virtualenv not found && exit 1
fi

if command_exists module; then
    module load python/3.9 scipy-stack/2022a
fi

rm -rf venv
virtualenv --no-download venv
source venv/bin/activate
python -m pip install --require-virtualenv --upgrade pip
python -m pip install --require-virtualenv -r requirements.txt
pip install .