# Kaggle Binary Classification with a Bank Churn Dataset

# Overview

This project aims to predict customer churn in a banking context. The dataset used is from the Kaggle Playground Series - Season 4 Episode 1.

My personal goal is to create a model that achieves an accuracy of at least 70%. 

## Problem Statement

The task is to predict whether a customer will continue with their bank account or close it (i.e., churn). This model could be critical for banks to understand customer behavior and implement retention strategies.

## Dataset Source

The dataset can be found on Kaggle: [Playground Series - Season 4 Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1).

# Data Pre-processing

## Libraries Import

```python
# Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline
```

## Data Loading

```python
# Loading the training dataset
train = pd.read_csv("data/train.csv")
```

## Initial Data Exploration

```python
# Displaying basic information about the dataset
train.info()
```

**Output:**
The dataset includes various columns like 'CustomerId', 'Surname', 'CreditScore', etc., with a mix of numerical and categorical data types.

# Exploratory Data Analysis (EDA)

Our analysis began with deriving insights from the dataset. Here's what we found:

- The average credit score is relatively high, indicating a customer base with generally good creditworthiness.
- The customer age distribution is skewed towards younger individuals.
- The average tenure with the bank is about 5 years, suggesting a mix of new and long-standing customers.
- A significant portion of customers have some balance in their accounts.
- The majority of customers possess a credit card.

These insights help us understand the demographics and financial behaviors of the customers, which are crucial for predicting churn.

# Data Correlation

To understand how different attributes relate to customer churn, we conducted a correlation analysis. The following steps were taken:

1. We identified numerical features relevant to churn.
2. We excluded non-numerical data for this analysis.
3. We employed a heatmap to visualize the correlation matrix.

## Code Snippet for Correlation Analysis

```python
numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary", "Exited"]
train_numerical = train.loc[:, numerical_features]
corr_matrix = train_numerical.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)

# Add titles and labels as needed
plt.title('Correlation Heatmap')
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.show()
```

![image](https://github.com/TangLitEn/kaggle-Binary-Classification-with-a-Bank-Churn-Dataset/assets/65808174/27e1359e-f2f7-4153-af33-2cffaf7de8fa)

## Correlation Findings

- **Positive Correlation:** Age and account balance showed a positive correlation with churn, suggesting older customers and those with higher balances are more likely to churn.
- **Negative Correlation:** The number of products a customer uses and their activity level showed a negative correlation with

churn, indicating that customers engaged with more bank products and those who are active are less likely to leave.
- **Little to No Correlation:** Surprisingly, credit score, tenure, and estimated salary showed little to no correlation with churn.

## Geographic Analysis

We explored the impact of geography on churn by visualizing the distribution of churn across different regions:

### Code Snippet for Geographic Analysis

```python
# Count plot for customer distribution by geography
sns.countplot(data=train, x="Geography", hue="Exited")
plt.show()
```

![image](https://github.com/TangLitEn/kaggle-Binary-Classification-with-a-Bank-Churn-Dataset/assets/65808174/7032719e-8762-42ed-b550-5df9c80acc81)

### Geographic Findings

- Customers from Germany have a noticeably higher churn rate compared to those from France and Spain.
- This difference might suggest that cultural factors or market competition in Germany could be influencing customer churn.

## Gender Analysis

Another aspect we analyzed was the effect of gender on churn:

### Code Snippet for Gender Analysis

```python
# Count plot for customer distribution by gender
sns.countplot(data=train, x="Gender", hue="Exited")
plt.show()
```
![image](https://github.com/TangLitEn/kaggle-Binary-Classification-with-a-Bank-Churn-Dataset/assets/65808174/e2a85ea2-34d0-410f-bf90-8fc20a552290)

#### Gender Findings

- The churn rate among female customers was higher than that of male customers.
- This pattern might indicate potential areas for improving customer retention strategies for different demographic groups.

## Churn by Geography and Gender

We further drilled down into the data to examine the churn rate within specific geographies and genders:

### Code Snippets for Churn Analysis by Geography and Gender

```python
# Churn rate in France
France = train[train.Geography == "France"]
France_Exited = France[France.Exited == 1]
print("France Exited ratio: ", len(France_Exited)/len(France))

# Churn rate in Spain
Spain = train[train.Geography == "Spain"]
Spain_Exited = Spain[Spain.Exited == 1]
print("Spain Exited ratio: ", len(Spain_Exited)/len(Spain))

# Churn rate in Germany
Germany = train[train.Geography == "Germany"]
Germany_Exited = Germany[Germany.Exited == 1]
print("Germany Exited ratio: ", len(Germany_Exited)/len(Germany

```

### Churn Analysis Findings by Geography

- France had a churn ratio of approximately 16.5%, Spain 17.2%, and Germany a significant 37.9%.
- This discrepancy emphasizes the importance of regional strategies in customer retention efforts.

```python
# Churn rate by Gender
Male = train[train.Gender == "Male"]
Male_Exited = Male[Male.Exited == 1]
print("Male Exited ratio: ", len(Male_Exited)/len(Male))

Female = train[train.Gender == "Female"]
Female_Exited = Female[Female.Exited == 1]
print("Female Exited ratio: ", len(Female_Exited)/len(Female))
```

### Churn Analysis Findings by Gender

- The churn ratio for male customers was around 15.9%, whereas for female customers it was higher, at approximately 27.9%.
- This indicates gender-specific factors may influence the likelihood of churn and should be considered in customer engagement strategies.

## Conclusions from EDA

The exploratory data analysis provided valuable insights into factors influencing customer churn. Our findings highlight the importance of age, balance, product usage, and activity level, as well as geographic and gender differences. This understanding can be instrumental in developing targeted interventions to improve customer retention.

### Model Building



### Model Evaluation



### Final Model Selection



## Results



## Challenges and Learnings



## Future Work

1. Explore on one-hot-encoding method🔥.
