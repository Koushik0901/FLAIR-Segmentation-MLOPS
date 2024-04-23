# ***Brain MRI FLAIR Segmentation*** 
***A Complete MLOPS project on Brain MRI FLAIR Segmentation. A MobileNet v3 based segmentation project to perform instance segmentation on FLAIR (Fluid-Attenuated Inversion Recovery) abnormality in brain MRI images. This model is trained using [Brain MRI segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) from kaggle and is deployed using Streamlit.***

## **Demo**
 ### Try it yourself [here](https://koushik0901-flair-segmentation-mlops-inferenceui-1so37o.streamlit.app/)
 <p align="center"> <img src="https://github.com/Koushik0901/FLAIR-Segmentation-MLOPS/blob/master/images/predictions.png" width="700" height="500"  /> </p>

## **Project Organization**
------------

    ├── LICENSE
    ├── README.md               <- Documentation to get more information about the project.
    ├── data
    │   |
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── saved_models            <- Trained and serialized models.
    │
    ├── inference               <- Code for inference.
    │
    ├── report                  <- metrics and logs from training.
    ├── images                  <- Generated graphics and figures to be used in reporting
    │
    ├── full-requirements.txt   <- The requirements file for reproducing the analysis environment.
    |
    ├── requirements.txt        <- The requirements file for reproducing the deployment environment.
    │
    ├── dvc.yaml                <- dvc.yaml is used to define dvc pipelines.
    │
    ├── config.yaml             <- Contains all the parameters for training.
    │
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module.
    │   │
    |   ├── utils.py            <- Script that contains utility functions.
    |   |
    |   ├── earl_stopping.py    <- Script that contains early stopping callback.
    |   |
    │   ├── data                <- Scripts to download or generate data.
    │   │   └── preprocess_data.py
    |   |   |
    │   │   └── dataset.py
    │   │
    │   ├── models              <- Scripts to train models and optimize graphs.
    │       ├── train.py
    │       └── optimize-graph.py
    │
    ├── tests                   <- Unit test code.
    │   ├── __init__.py         <- Makes tests a Python module.
    │   ├── config_test.py      <- tests the config.yaml file.
    |
    └── Procfile                <- Procfile is used for deployment in Heroku.
    │
    └── setup.sh                <- setup.sh  is used for deployment in Heroku.
    │
    └── runtime.txt             <- runtime.txt is used to specify the python runtime version in Heroku.
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io.
## **Running on native machine**
### *dependencies*
* python3
### *pip packages*
```bash
pip install -r requirements.txt
```
## **Steps to train your own model**
 ### *Scripts*
 `src/train.py` - is used to train the model \
 `inference/engine.py` - is used to perform inference \
 `inference/ui.py` - is used to build the streamlit web application
