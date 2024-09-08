# Predicting-Life-Satisfaction-Using-Machine-Learning
Welcome to our research repository! This repository contains a Jupyter Notebook illustrating the computational analysis and visualization done as part of our research article published in Heliyon titled "Predicting life satisfaction using machine learning and explainable AI" ([Link](https://www.sciencedirect.com/science/article/pii/S2405844024071895)). Here, you will find the steps and code snippets used to process the data, build, and validate models, alongside visualizing the results.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- **Python:** Our analyses are conducted using Python. You need to have Python installed to run the Jupyter Notebook. Download it [here](https://www.python.org/downloads/).
- **Jupyter Notebook:** This is where all the magic happens! You can install it using pip (Python’s package installer). If you haven’t installed pip, you can get it [here](https://pip.pypa.io/en/stable/installation/).

### Libraries
To run the notebook, the following libraries are required. You can install them using pip:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn missingno imbalanced-learn scikit-plot xgboost lightgbm

```
## Overview of Libraries
- **[NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/)**: Employed for data manipulation and analysis.
- **[Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)**: Utilized for creating static, animated, and interactive visualizations in Python.
- **[Scikit-learn](https://scikit-learn.org/stable/)**: Applied for various machine learning models and data preprocessing.
- **[Missingno](https://github.com/ResidentMario/missingno)**: Beneficial for visualizing missing data.
- **[Imbalanced-learn](https://imbalanced-learn.org/stable/)**: Valuable for addressing imbalanced datasets through resampling techniques.
- **[Scikit-plot](https://scikit-plot.readthedocs.io/en/stable/)**: Handy for visualizing machine learning results and metrics.
- **[XGBoost](https://xgboost.readthedocs.io/en/latest/)**: Implemented for the XGBoost algorithm, an optimized gradient-boosting machine learning library.
- **[LightGBM](https://lightgbm.readthedocs.io/en/latest/)**: Incorporated for utilizing the LightGBM algorithm, a gradient boosting framework that uses tree-based learning algorithms.

### Clone the Repository
To get started, clone this repository to your local machine using the following command:
```bash
git clone https://github.com/alifelham/Predicting-Life-Satisfaction-Using-Machine-Learning.git
```
Navigate to the cloned directory:

```bash
cd [REPO DIRECTORY]
```

### Launch Jupyter Notebook 
Run the following command to start Jupyter Notebook:
```bash
jupyter notebook
```
Navigate through the Jupyter Notebook interface in your browser to open the notebook.

### How to run
- To use a custom dataset similar to the SHILD dataset, upload it in the same folder as then ipynb file.
- There are several models included in the ipynb file itself, the user can choose to run any of them for testing purposes.
- After uploading our pretrained model our obtained results can be reproduced.
  
## Dataset
* SHILD (Survey of Health Impairment and Living Conditions in Denmark): 
Get the dataset from **[here](https://doi.org/10.5061/dryad.qd2nj)**

### License of the Dataset 
Licensed under a **[CC0 1.0 Universal (CC0 1.0) Public Domain Dedication license.](https://creativecommons.org/publicdomain/zero/1.0/)**

## Notebook Structure
Our notebook is structured as follows:

1. Data Importing and Preprocessing: Loading the dataset and performing initial preprocessing.
2. Exploratory Data Analysis: Visualizations and summaries to understand the data.
3. Model Building: Implementing machine learning models (Random Forest, Logistic Regression, etc.) and tuning their parameters.
4. Model Evaluation: Evaluating model performance using various metrics.
5. Results Visualization: Graphically representing the results obtained.
6. Explainable AI: Used to interpret the results of the models.


## Contact Details of the Authors:
Alif Elham Khan: alif.khan1@northsouth.edu

Mohammad Junayed Hasan: mohammad.hasan5@northsouth.edu

Humayra Anjum: humayra.anjum@northsouth.edu

Dr. Nabeel Mohammad: nabeel.mohammed@northsouth.edu

Dr. Sifat Momen: sifat.momen@northsouth.edu

## Acknowledgement
A big thanks to our peers and reviewers for their valuable inputs. We would like to acknowledge our supervisors, Dr. Sifat Momen and Dr. Nabeel Mohammed for their patience and guidance during this research.
