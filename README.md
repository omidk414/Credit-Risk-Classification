# Loan Risk Prediction Model

## Overview

The purpose of this analysis was to develop and evaluate a machine learning model for predicting loan risk. The goal was to create a tool that could accurately classify loans as either healthy or high-risk, thereby assisting financial institutions in making informed lending decisions and managing their risk exposure.
This project implements a supervised machine learning model to predict loan risk using logistic regression. The model classifies loans as either healthy (0) or high-risk (1) based on various financial features.

## Table of Contents
1. [Features](#features)
2. [Example Code](#example-code)
3. [Model Performance](#model-performance)
4. [Results Summary](#results-summary)
5. [Analysis](#analysis)

## Features

- Utilizes logistic regression for binary classification
- Handles imbalanced dataset of loan applications
- Provides comprehensive performance metrics
- Includes confusion matrix visualization

## Example Code

```python
# Generate a confusion matrix for the model
# A value of 0 in the "loan_status" column means that the loan is healthy.
# A value of 1 means that the loan has a high risk of defaulting.
confusion_matrix_model = confusion_matrix(y_test, predictions)
confusion_matrix_model_df = pd.DataFrame(
    confusion_matrix_model,
    index=['Actual Healthy (0)', 'Actual High-Risk (1)'],
    columns=['Predicted Healthy (0)', 'Predicted High-Risk (1)']
)
print(confusion_matrix_model_df)
```

Output:
![confusion_matrix](https://github.com/omidk414/credit-risk-classification/blob/main/Credit_Risk/images/confusion_matrix.png)

## Model Performance
![model_performance](https://github.com/omidk414/credit-risk-classification/blob/main/Credit_Risk/images/model_performance.png)

## Results Summary

The logistic regression model demonstrates exceptional performance in predicting loan risk:

- The overall accuracy of 0.99 indicates that the model correctly classifies 99% of all loans.
- For healthy loans, the model achieves perfect precision (1.00) and near-perfect recall (0.99).
- For high-risk loans, the model maintains high recall (0.94) while achieving good precision (0.84).
- The balanced F1-scores for both classes (1.00 for healthy loans and 0.89 for high-risk loans) demonstrate the model's ability to handle the class imbalance effectively.

## Analysis

Based on the performance, I strongly recommend the use of this logistic regression model for loan risk prediction. The model's high accuracy, precision, and recall make it a reliable tool for risk management in lending decisions. The ability to maintain high recall for high-risk loans (0.94) is significant, as it minimizes the chance of approving potentially defaulting loans. The model's performance across all metrics, especially in the significant class imbalance from the dataset, further proves its reliability and effectiveness in real-world applications.
