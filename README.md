# 🧾 S&P 500 Stock Price Movement Prediction

# 1. Project Title & Overview

This project develops a predictive model for forecasting the direction of the S&P 500 Index (using the ^GSPC ticker) based on historical closing prices. Utilizing a machine learning approach, the system aims to predict whether the stock price will move "Up" (1) or "Down" (0) on the next trading day. This capability provides quantitative guidance for swing trading strategies, risk management, and market timing decisions.

# 2. Problem Statement

The stock market is inherently volatile and noisy. The challenge is to filter this noise and build a model that can identify persistent, non-random patterns in historical price data sufficient to predict the binary direction of future price movement with accuracy exceeding a simple coin flip (50%). The real-world issue is the need for a data-driven system to augment human judgment in financial trading, providing an objective probabilistic forecast for short-term market direction.

# 3. Objectives

The key goals for this project were:

Data Acquisition: Programmatically fetch comprehensive historical stock data for the S&P 500 index.

Feature Engineering: Create a robust set of technical features based on price history (e.g., rolling means, look-back periods) to capture momentum and volatility.

Target Creation: Define a binary target variable (Target) indicating future price movement.

Model Training & Optimization: Train a Random Forest Classifier and optimize its parameters to maximize predictive accuracy.

Backtesting Simulation: Implement a time series-aware backtesting process to rigorously evaluate the model's performance on unseen, chronological data.

# 4. Dataset Description

Attribute

Details

Source

Yahoo Finance (via yfinance library)

Ticker

^GSPC (S&P 500 Index)

Samples

Historical data up to the present day (period=max)

Features

Initial: Open, High, Low, Close, Volume, Dividends, Stock Splits

Feature Engineering

Engineered features include: Rolling means of closing price over multiple look-back periods (e.g., 2, 5, 60, 250, 1000 days), and the ratio of current price to these rolling means.

Target Variable

A binary Target column (1 for price increase tomorrow, 0 for decrease/no change), created by shifting the next day's price relative to the current day's price.

# 5. Methodology / Approach

Data Preprocessing

Date Indexing: The raw data was filtered to include only necessary columns (Close, Volume).

Target Definition: The Target was defined as 1 if the next day's closing price was greater than the current day's closing price.

Backfill: All NaN values (created by rolling windows and the look-ahead Target) were removed, ensuring the model only trains on complete data.

Model Used: Random Forest Classifier

A Random Forest Classifier was selected for its robustness against overfitting, ability to handle complex feature interactions, and suitability for binary classification tasks.

Training, Testing, and Evaluation Strategy (Backtesting)

Due to the sequential nature of time series data, a standard train-test split is inappropriate. A rigorous Backtesting function was implemented:

Initial Training Period: The model was initially trained on the first 2500 rows of data.

Iterative Training: The model was re-trained every 250 steps (approximately yearly), simulating a real-world scenario where the model is periodically updated with new market data.

Prediction: At each step, the model predicts the movement for the next 250 days based on the current knowledge.

Evaluation: Predictions were compiled chronologically, and the final performance metrics were calculated across the entire backtested period.

# 6. Results & Evaluation

Performance Metrics

The notebook demonstrates the final overall accuracy achieved by the backtested model:

Accuracy: 57.9%

Prediction Counts: The model predicts "Down" (0.0) much more frequently (4690 times) than "Up" (1.0) (829 times), suggesting a bias toward stability or a prevailing downward trend during the backtest period.

Interpretation

An accuracy of 57.9% is significantly better than random chance (50%). While not high enough for flawless trading, it demonstrates a quantifiable edge, indicating that the engineered features successfully capture patterns in historical price data that are predictive of short-term direction. The low number of "Up" predictions warrants further investigation into the model's sensitivity to positive movements.

# 7. Technologies Used

Category

Technology / Library

Language

Python 3.x

Data Acquisition

yfinance

Data Manipulation

Pandas, NumPy

Modeling

Scikit-learn (RandomForestClassifier)

Streamlit Interface

(Planned for Future Improvement)

# 8. How to Run the Project

Prerequisites

Ensure you have a Python 3 environment installed.

# Install the necessary libraries
pip install pandas numpy yfinance scikit-learn


Execution Guide

Save the notebook content as Stock_Price_Prediction.ipynb.

Ensure internet connectivity to fetch data from Yahoo Finance.

Open the file in a Jupyter environment (Jupyter Lab or VS Code).

Execute all cells sequentially. The backtesting process is computationally intensive and may take several minutes to run, after which the final predictions and accuracy score will be printed.

# 9. Conclusion

The Random Forest model successfully demonstrated a statistically significant ability to predict the next day's S&P 500 movement with an accuracy of nearly 58%. The project validates the feasibility of using rolling technical indicators as primary features in a time series classification context. This model provides a foundational component for an automated trading signal generation system.
