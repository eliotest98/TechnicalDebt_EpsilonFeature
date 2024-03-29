import logging
import os
import time

import dagshub
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util import utils

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    mlflow.set_experiment("Adult")

    #
    # Load the adult dataset
    #
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'preprocessed_adult.csv'))
    df = pd.read_csv(csv_path, sep=',')

    x = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']]
    y = df['income']

    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    #
    # Feature scaling
    #
    sc = StandardScaler()
    sc.fit(x_train)
    X_train_std = sc.transform(x_train)
    X_test_std = sc.transform(x_test)

    #
    # Training / Test Dataframe
    #
    cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'age', 'fnlwgt', 'capital-gain',
            'capital-loss', 'hours-per-week']
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    forest = RandomForestClassifier(n_estimators=500, random_state=42)

    # Store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the model
    #
    forest.fit(X_train_std, y_train.values.ravel())

    # Execution time (Metric) at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    importances = forest.feature_importances_

    #
    # Prediction
    #
    y_pred_test = forest.predict(X_test_std)
    #
    # Sort the feature importance in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    # Confusion Matrix
    utils.confusion_matrix(y_test, y_pred_test)

    # Metrics
    utils.metrics_fi(y_test, y_pred_test, x_train, importances, sorted_indices, execution_time)
