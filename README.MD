# TUZA Take Home Assesment
This repository contains the work submitted for the Data Science Take Home Assessment for TUZA. This README details how to setup your machine and how to use the models developed for the project.

## Setting Up the Virtual Environment
Follow these steps to set up the virtual environment with the required Python version and libraries in your sourrce directory:

### 1. Create the Virtual Environment
```bash
python3.12 -m venv .venv
```
### 2. Activate the Virtual Environment
```bash
source .venv/bin/activate
```
### 3. Upgrade pip and Install Dependencies
The library dependancies are found in [requirements.txt](https://github.com/lucarino2000/TUZA/blob/main/requirements.txt).
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## The Project

### Testing
Running [test_models.py](https://github.com/lucarino2000/TUZA/blob/main/test_models.py) trains a variety of discriminative models on the dataset in order to compare their effectiveness before selecting the optimal model.

### The Models
To view how the models were trained see [predictive_model.py](https://github.com/lucarino2000/TUZA/blob/main/predictive_model.py) and [clustering_model.py](https://github.com/lucarino2000/TUZA/blob/main/clustering_model.py) respectively, when run they will save the test set classification as a .csv file with their original row index from [data.csv](https://github.com/lucarino2000/TUZA/blob/main/data.csv), [predictive_model.py](https://github.com/lucarino2000/TUZA/blob/main/predictive_model.py) will also save an augmented version of the original dataset (containing the infered labels) as 'data_labelled.csv' as well as classify a real world example from [floom.csv](https://github.com/lucarino2000/TUZA/blob/main/floom.csv).

Running each file will give the infered outputs on the test set from the models trained on the dataset derived from [data.csv](https://github.com/lucarino2000/TUZA/blob/main/data.csv), see [create_dataset.py](https://github.com/lucarino2000/TUZA/blob/main/create_dataset.py).

### The Selected Model
Running [predictive_model.py](https://github.com/lucarino2000/TUZA/blob/main/predictive_model.py) will output the scores of the selected model on both the validation and the test set. It will also plot the distribution of test set errors in order to visualise how classification is performed and then once the window is closed it will display a plot for all the data showing the classification labels based on provider. Finally, once previous plot window is closed, it will classify a real-world example datapoint from the company data of Floom Creative. 

### Helper functions and files
[nn.py](https://github.com/lucarino2000/TUZA/blob/main/nn.py) contains the PyTorch implementation for a simple neural network to be tested as a potential model for the task, when trained it will save its best weights as 'best_model.pth' in your directory.

[visa_mcc.json](https://github.com/lucarino2000/TUZA/blob/main/visa_mcc.json) a file containing the labelling of MCC Codes to business types, this was used in conjuction with other research to derive an MCC Code to MCC Risk variable conversion.
