import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from boosting import XGBoost
from xgboost import XGBClassifier

def main():
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # Initialize your XGBoost model (adjust parameters as necessary)
    xgb_model = XGBoost(
        num_tree=400,  # Number of trees
        learning_rate=0.01,
        reg_lambda=4,
        gamma=0.05,
        min_child_weight=5,
        max_depth=8,
    )
    xgb_model_ideal = XGBClassifier(
        n_estimators=200,  # Number of trees
        learning_rate=0.01,
        reg_lambda=1.5,
        gamma=0.03,
        min_child_weight=5,
        max_depth=30,
    )
    
    
    # Fit the model on the training data
    xgb_model.fit(X_train, y_train)

    # Predict on the testing data
    print(xgb_model.predict(X_test))
    y_pred = xgb_model.predict(X_test) > 0.5  # Thresholding at 0.5 for binary classification

    # Evaluate the model using accuracy
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")

if __name__ == "__main__":
    main()