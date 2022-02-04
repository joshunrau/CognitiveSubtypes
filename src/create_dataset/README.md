# Subsetting Biobank Data

## About

This module contains the code used to subset tabular data from the UK Biobank. The files config.py and dataset.py contain the Config and Dataset classes respectively. To make things easy to understand for people who do not know how to code, the Config class is designed to encapsulate all variables that can be manipulated when subsetting the dataset, whereas the Dataset class – which inherits the attributes (i.e., variables) of the Config class – provides methods to create and manipulate the dataset. After modifying the Config class, you can import the create_dataset module and use these methods to create your dataset.

## Example

This example assumes that the create_dataset directory is located in your working directory.

If you are using Compute Canada, load the required modules:
  module load python/3.9 scipy-stack



