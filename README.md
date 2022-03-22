# CognitiveSubtypes
Cognitive Subtypes and Associations with Brain Structure in the UK Biobank: A Machine Learning Approach

## Reproducing Analysis

Step 1: Create a new virtual environment and install dependencies from requirements.txt: 

        make venv

Step 2: Construct a useable dataset from raw data Biobank data:

        export CURRENT_CSV=/Users/joshua/Developer/CognitiveSubtypes/data/raw/current.csv
        export DATA_DIR=/Users/joshua/Developer/CognitiveSubtypes/data
        make data

