# speedtest-classification
Exploring multiclass classification with sklearn custom transform pipelines on internet speedtest data.  

> NOTE: _Currently still work in progress_

## How to run locally
    make environment
will create a conda environment with the packages listed in `environment.yaml`.  

    make raw_data
will load the dataset from the web.

    make input_data
will select relevant instances and inputs for further modelling / feature generation.

    make tests
will run the tests.  

## Model training and evaluation
is handelled in `notebooks/05-training.ipynb`

---
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
