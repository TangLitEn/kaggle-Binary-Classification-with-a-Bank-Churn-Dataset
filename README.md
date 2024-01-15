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

# Data Encoding

## Encoding Non-numerical Data

Machine learning models generally require numerical input, so it was necessary to encode non-numerical features. We converted categorical variables into numerical values using a basic encoding scheme.

### Geography and Gender Encoding

We mapped the 'Geography' and 'Gender' categorical variables to integers as follows:

- **Geography**: France to 0, Spain to 1, Germany to 2
- **Gender**: Male to 1, Female to 0

Although this is not the optimal way to encode categorical data due to the introduction of artificial ordinal relationships, it serves as a straightforward initial approach.

### Code Snippet for Encoding

```python
def Gender_to_Num(row):
    if row.Gender == "Male": return 1
    elif row.Gender == "Female": return 0
    else: return 2 # Fallback case

train["Gender_Numerical"] = train.apply(Gender_to_Num, axis=1)

def Geography_to_Num(row):
    if row.Geography == "France": return 0
    elif row.Geography == "Spain": return 1
    elif row.Geography == "Germany": return 2
    else: return 3 # Fallback case

train["Geography_Numerical"] = train.apply(Geography_to_Num, axis=1)
```

### Considerations for Encoding

It's important to note that using numerical encoding for categorical data might imply an ordinal relationship where none exists. For example, assigning Germany a higher number than Spain does not mean it's 'greater' in any meaningful way for our model. In future iterations, one-hot encoding or similar techniques that avoid implying an order should be considered to potentially improve model performance.

# Model Selection and Training

## Model Selection

For this project, I decided to use the Random Forest model due to its robust performance on similar datasets in past experiences. Random Forest is an ensemble learning method that can handle both classification and regression tasks well. It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.

## Feature Consideration

Before training the model, we carefully selected features based on their impact and correlation with the target variable, 'Exited'. The following features were excluded due to their little impact or correlation with the churn outcome:

- `id`: A unique identifier that has no predictive power.
- `CreditScore`: Surprisingly showed little correlation with churn.
- `Tenure`: Had little impact on the target variable.
- `EstimatedSalary`: Did not show significant correlation with churn.

The final features used for the model are as follows:

```python
FOREST_features = ['Geography_Numerical', 'Gender_Numerical', 'Age', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
FOREST_output = ['Exited']
```

## Splitting the Dataset

The dataset was split into training and testing sets to evaluate the performance of the model. We used 80% of the data for training and reserved 20% for testing.

### Code Snippet for Splitting the Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[FOREST_features], train[FOREST_output], test_size=0.2, random_state=42)
```

## Random Forest Training

The Random Forest Classifier from `sklearn` was used for training the model. We specified the number of estimators and the maximum depth for the trees.

### Code Snippet for Training the Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

FOREST_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
FOREST_model.fit(X_train, y_train)
```

The `n_estimators` parameter dictates the number of trees in the forest, and the `max_depth` parameter controls the maximum depth of the trees. We chose a `random_state` to ensure the reproducibility of our results.

# Model Evaluation

Evaluating the performance of the machine learning model is crucial to understanding its effectiveness and where it may need improvement. For this project, we used a confusion matrix to visualize the performance of our Random Forest model and calculated the F1 score as a metric.

## Confusion Matrix

A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows easy identification of confusion between classes i.e., how often the model confused two classes.

### Code Snippet for Confusion Matrix

```python
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_test["Exited"], y_test["Predicted Exiting"])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.show()
```

The confusion matrix provides insights into the number of correct and incorrect predictions made by the model, distinguished by each class.

## F1 Score

The F1 score is a measure of a test's accuracy and considers both the precision and the recall of the test to compute the score. The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.

### Code Snippet for F1 Score Calculation

```python
f1 = metrics.f1_score(y_test["Exited"], y_test["Predicted Exiting"], average="weighted")
print(f1)
```

By using the `average="weighted"` parameter, we ensure that the F1 score takes into account the imbalance in the class distribution.

## Evaluation Summary

![image](https://github.com/TangLitEn/kaggle-Binary-Classification-with-a-Bank-Churn-Dataset/assets/65808174/c36b7426-888b-408a-9da1-71e4113442df)

**F1 Score: 0.830002870867119**

# Challenges and Learnings

"While the dataset from the Kaggle Playground Series may present itself as straightforward, the true essence and excitement of data analysis lie in uncovering the narratives woven into the fabric of the data. Each dataset tells a story, and it is through rigorous exploration and diverse analytical methods that we decode these hidden tales. The journey of dissecting the dataset, interpreting its patterns, and predicting outcomes is not just about applying algorithms—it's a quest to understand the underlying phenomena that the data encapsulates. As we peel back the layers, we're granted insights into the behaviors and trends that would otherwise remain obscured. This project is not just a technical exercise; it's an investigative adventure into the heart of data storytelling." - ChatGPT4

^ Good writing by ChatGPT🤣

# Future Work and Improvements

1. Explore on one-hot-encoding method🔥.
