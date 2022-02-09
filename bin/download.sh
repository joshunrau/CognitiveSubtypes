#!/bin/bash

PATH_TABULAR_DATA=/home/junrau/projects/def-mlepage/UKBB/current_civet.csv
PATH_DOWNLOAD_DIR=/Users/joshua/Developer/CognitiveSubtypes/data

if [ ! -d $PATH_DOWNLOAD_DIR ]; then
    mkdir $PATH_DOWNLOAD_DIR
fi

autossh cc -d $PATH_TABULAR_DATA $PATH_DOWNLOAD_DIR