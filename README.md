# ChurnPrediction
Using various ML and feature engineering techniques and evaluating them to find the best performing model for a client to leave or not.
# Customer Churn Prediction

This project aims to predict customer churn for a service provider using machine learning techniques. The dataset used contains customer demographic and account data, and the goal is to classify customers as either "Churned" (Exited) or "Retained" based on various features such as tenure, balance, credit score, and more.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
Customer churn refers to when a customer stops using a company's services. In this project, we use machine learning algorithms to predict whether a customer will churn (exit) based on several features, allowing businesses to take proactive actions to retain customers.

## Dataset
The dataset used for this project is the **Churn_Modelling.csv** file. It contains information about:
- Customer demographics (age, gender, geography)
- Account details (credit score, balance, number of products)
- Whether the customer has exited (churned) or not (`Exited` column)

## Project Overview
The project is divided into the following stages:
1. **Data Preprocessing**: Data cleaning, handling missing values, and one-hot encoding for categorical variables.
2. **Feature Engineering**: Creating new features such as `BalanceSalaryRatio` and `TenureByAge` to enrich the dataset.
3. **Exploratory Data Analysis**: Visualizing the distribution of churn across different features using pie charts, box plots, and histograms.
4. **Model Building**: Implementing various machine learning algorithms, including:
   - Logistic Regression
   - Random Forest
   - LightGBM
   - Sequential Neural Network
   - Decision Tree
   - AdaBoost
   - CatBoost
5. **Model Evaluation**: Calculating and comparing the accuracy, precision, recall, F1-score, and confusion matrix for each model.
6. **Conclusion**: Determining the best model based on the evaluation metrics.

## Requirements
- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- CatBoost
- LightGBM
- Google Colab (for execution in Colab)

- ## Feature Engineering
We engineered additional features to enhance model performance:

- **BalanceSalaryRatio**: Ratio of balance to estimated salary.
- **TenureByAge**: Ratio of tenure to age, which helps understand customer loyalty relative to their age.

## Machine Learning Models
The following machine learning algorithms were used:

- **Logistic Regression**: A baseline linear model for binary classification.
- **Random Forest**: An ensemble learning method based on decision trees.
- **LightGBM**: A gradient boosting model optimized for speed and efficiency.
- **Sequential Neural Network**: A basic neural network using TensorFlow/Keras.
- **Decision Tree**: A tree-based model.
- **AdaBoost**: An ensemble learning method using weak learners (decision trees).
- **CatBoost**: A gradient boosting model particularly suited for categorical features.

## Evaluation
We evaluated the models based on:

- **Accuracy**: Overall correctness of predictions.
- **Precision**: Correctly predicted positive cases out of all positive predictions.
- **Recall**: Correctly predicted positive cases out of actual positive cases.
- **F1-Score**: Harmonic mean of precision and recall, useful for imbalanced datasets.
- **Confusion Matrix**: A matrix displaying true positives, true negatives, false positives, and false negatives.

### Model Performance Summary:
The **LightGBM** model was identified as the best-performing model for predicting customer churn, with the highest average score across all metrics (accuracy, precision, recall, and F1-score).

## Conclusion
In this project, we identified that **LightGBM** performed the best for predicting customer churn. The model achieved a balanced performance across various metrics and handled the imbalanced nature of the dataset effectively. This information can help businesses take preventive actions to retain customers and reduce churn.
