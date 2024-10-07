import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load data, then separate x and y variables
train = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

train.head()

y = train['meal']
X = train.drop(columns=['id', 'DateTime', 'meal'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

boosted_tree_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
boosted_tree_model.fit(X_train, y_train)

# Decision Tree evaluation
y_pred_dt = decision_tree_model.predict(X_val)
print(f"Decision Tree Accuracy: {accuracy_score(y_val, y_pred_dt)}")

# Random Forest evaluation
y_pred_rf = random_forest_model.predict(X_val)
print(f"Random Forest Accuracy: {accuracy_score(y_val, y_pred_rf)}")

# Boosted Tree evaluation
y_pred_bt = boosted_tree_model.predict(X_val)
print(f"Boosted Tree Accuracy: {accuracy_score(y_val, y_pred_bt)}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
modelFit = model.fit(X_train, y_train)

# Prepare the test data
X_test = test.drop(columns=['id', 'DateTime', 'meal'])

# Make predictions
pred = modelFit.predict(X_test)

pred
