# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 05:07:09 2025

@author: hongf
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # ✅ Import restored

# ✅ Load the dataset correctly
def load_data():
    data = load_iris(as_frame=True)
    df = data.frame  # This is correct if using as_frame=True
    df['target'] = data.target
    return df

# ✅ Preprocess the data with correct column names and transformation
def preprocess(df):
    # These are the actual column names in the iris dataset
    features = df[['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']]
    X = features
    y = df['target']

    # Correct: Use fit_transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train and evaluate the model
def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    return acc

# ✅ Main block
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    train_model(X_train, X_test, y_train, y_test)
