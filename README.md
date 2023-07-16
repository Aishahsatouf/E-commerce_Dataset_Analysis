# Brazilian E-commerce Dataset Analysis

## Introduction

This repository contains Python code to analyze the Brazilian E-commerce dataset, which includes information from various sources related to customer orders, product details, reviews, and payment information. The code aims to perform data preprocessing, exploratory data analysis, and train a machine learning model to predict product prices based on various features.

## Dependencies

To run the code in this repository, you'll need the following Python libraries:

- pandas (as pd): For data manipulation and analysis
- numpy (as np): For numerical computing and array operations
- scikit-learn: For machine learning algorithms and evaluation metrics
- matplotlib: For creating visualizations
- datetime: For handling date and time operations
- seaborn: For creating advanced visualizations
- plotly.express: For interactive data visualizations

## Data Loading

The code reads multiple CSV files containing different datasets related to the Brazilian E-commerce business, such as order items, orders, payments, products, customers, sellers, reviews, and product category translations. The data is merged to create a consolidated dataset.

## Data Preprocessing

1. Outliers Removal: The code removes outliers from the 'price' column of the dataset using the quantile method.
2. Date Manipulation: The code extracts the year and month from the 'order_purchase_timestamp' column and adds them as new columns.
3. Monthly Demand Calculation: The code calculates the sum of product quantity ordered per month ('demand') and merges it back to the original dataset.

## Price and Freight Analysis

The code performs a comparison of prices, freight values, and review scores for different product categories. It visualizes the relationships using a 3D scatter plot.

## Time Series Analysis

The code analyzes the variation of orders throughout the year using a histogram and a time series plot.

## Price-Quantity Relation

The code creates a scatter plot of the logarithmic values of demand and price to explore their exponential relationship.

## Correlation Heatmap

The code generates a correlation heatmap to identify the most important features that can be used to train a machine learning model to predict product prices.

## Linear Regression Model

1. Data Cleaning: The code drops irrelevant columns, converts categorical variables to numerical values using label encoding, and removes any remaining null values from the dataset.
2. Model Training: The code splits the data into training and testing sets, and it uses the Random Forest Regressor algorithm to train a machine learning model to predict product prices based on the selected features.
3. Model Evaluation: The code evaluates the model's performance using Mean Absolute Error (MAE) and R-squared (R2) metrics.

## Conclusion

This repository provides an analysis of the Brazilian E-commerce dataset, exploring relationships between product prices, demand, and various features. It also trains a machine learning model to predict product prices based on selected attributes. The code is well-documented, covering data loading, preprocessing, analysis, and model evaluation steps for comprehensive understanding and reproducibility.
