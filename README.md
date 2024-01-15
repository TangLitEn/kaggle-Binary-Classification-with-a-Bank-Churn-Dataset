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
Based on the provided data visualizations, here's how you can expand the GitHub README documentation for the Exploratory Data Analysis and Visualization section:

---

## Exploratory Data Analysis (EDA) and Visualization

### Data Insights

Through the exploratory data analysis, several insights have been gathered:

1. The average credit score in the dataset is relatively high.
2. The customer age distribution is skewed towards younger individuals.
3. The average tenure, which refers to the duration of holding the bank account, is about 5 years.
4. A significant portion of customers maintain some balance in their accounts.
5. A majority of customers possess a credit card.

*Note: Tenure refers to the amount of time you are given to repay your loan as per [Capital.com.sg](https://capital.com.sg).*

### Data Correlation Analysis

A heatmap was utilized to explore the correlation between different numerical features and the target

variable 'Exited', which indicates whether a customer has churned:

```python
numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary", "Exited"]

# Selecting the numerical columns for correlation analysis
train_numerical = train.loc[:, numerical_features]
corr_matrix = train_numerical.corr()

# Creating a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)

# Adding titles and labels
plt.title('Correlation Heatmap')
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.show()
```

*The heatmap provides a visual representation of the strength and direction of the relationship between the variables.*

### Insights from Correlation Heatmap

Upon reviewing the heatmap:

- **Positive Correlation**: Age and Balance show a positive correlation with customer churn.
- **Negative Correlation**: The number of products used by a customer and their activity level are negatively correlated with churn.
- **Little Correlation**: Surprisingly, CreditScore, Tenure, and EstimatedSalary show little to no correlation with churn.

### Geographic Analysis

To understand the geographic distribution of customer churn:

```python
# Count plot for customer distribution by geography
sns.countplot(data=train, x="Geography", hue="Exited")
```

![Geographic Distribution](/path/to/geographic_distribution

.png)

The count plot illustrates the number of customers in each geographical region and highlights the churn rate within these regions. The analysis shows that:

- France has the lowest churn ratio.
- Spain has a slightly higher churn ratio than France.
- Germany shows a significantly higher churn ratio, which could suggest cultural or economic factors at play.

```python
# Analysis by country
France = train[train.Geography == "France"]
France_Exited = France[France.Exited == 1]
print("France Exited ratio: ", len(France_Exited)/len(France))

Spain = train[train.Geography == "Spain"]
Spain_Exited = Spain[Spain.Exited == 1]
print("Spain Exited ratio: ", len(Spain_Exited)/len(Spain))

Germany = train[train.Geography == "Germany"]
Germany_Exited = Germany[Germany.Exited == 1]
print("Germany Exited ratio: ", len(Germany_Exited)/len(Germany))
```

### Gender Analysis

Similar to geographic analysis, customer churn was also examined based on gender:

```python
# Count plot for customer distribution by gender
sns.countplot(data=train, x="Gender", hue="Exited")
```

The count plot differentiates the churn rate between male and female customers, indicating:

- A higher churn rate for female customers as compared to male customers.

### Model Building



### Model Evaluation



### Final Model Selection



## Results



## Challenges and Learnings



## Future Work

1. Explore on one-hot-encoding method🔥.
