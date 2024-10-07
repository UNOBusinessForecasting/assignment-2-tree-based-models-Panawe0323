import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Loading data
train = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Preparing target and features
y = train['meal']  # Dependent variable
X = train.drop(columns=['id', 'DateTime', 'meal'])  # Drop irrelevant columns

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining and training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
modelFit = model.fit(X_train, y_train)

# Preparing the test data
X_test = test.drop(columns=['id', 'DateTime', 'meal'])

# Make predictions on test data
pred = modelFit.predict(X_test)

pred = pred.astype(int)

print(pred[:10])  # Print first 10 predictions to check
print(f"Total number of predictions: {len(pred)}")  # Ensure 1000 predictions are made
