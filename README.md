# Kaggle Binary Classification with a Bank Churn Dataset

## Overview

This project aims to predict customer churn in a banking context. The dataset used is from the Kaggle Playground Series - Season 4 Episode 1. The primary goal is to create a model that achieves an accuracy of at least 70%. 

### Problem Statement

The task is to predict whether a customer will continue with their bank account or close it (i.e., churn). This model could be critical for banks to understand customer behavior and implement retention strategies.

### Dataset Source

The dataset can be found on Kaggle: [Playground Series - Season 4 Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1).

## Methodology

### Data Pre-processing

#### Libraries Import

```python
# Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline
```

#### Data Loading

```python
# Loading the training dataset
train = pd.read_csv("data/train.csv")
```

#### Initial Data Exploration

```python
# Displaying basic information about the dataset
train.info()
```

**Output:**
The dataset includes various columns like 'CustomerId', 'Surname', 'CreditScore', etc., with a mix of numerical and categorical data types.

### Exploratory Data Analysis (EDA)



### Model Building



### Model Evaluation



### Final Model Selection



## Results



## Challenges and Learnings



## Future Work

1. Explore on one-hot-encoding method🔥.
