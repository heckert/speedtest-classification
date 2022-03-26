# adapted from
# https://github.com/drivendata/cookiecutter-data-science/blob/master/%7B%7B%20cookiecutter.repo_name%20%7D%7D/Makefile
PROJECT_DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# extract project name from full project directory
PROJECT_NAME=$(shell basename $(PROJECT_DIR))
# https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

environment:
# create conda env from environemt.yml file
	@echo "Creating conda environment for $(PROJECT_NAME)..."
	@conda env create -n $(PROJECT_NAME) -f environment.yaml

test:
	@$(CONDA_ACTIVATE) $(PROJECT_NAME); pytest

# load raw data from the web
raw_data:
# remove possible prior contents
	@rm -f ./data/raw/*
# load zipped data from url and store it in `data/raw`
	wget -N https://www.netztest.at/RMBTStatisticServer/export/netztest-opendata.zip -P ./data/raw/
# unzip and remove the zip
	unzip ./data/raw/netztest-opendata.zip -d ./data/raw/
	rm ./data/raw/netztest-opendata.zip

# select relevant rows and columns for modelling
input_data:
	@$(CONDA_ACTIVATE) $(PROJECT_NAME); python -m src --prepareInputData

	