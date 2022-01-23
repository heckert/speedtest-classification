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
	@conda env create -n $(PROJECT_NAME) -f environment.yml

test:
	@$(CONDA_ACTIVATE) $(PROJECT_NAME); pytest