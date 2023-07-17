import logging
import os
import time

import dagshub
import mlflow
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util import utils

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    mlflow.set_experiment("Iris")

    #
    # Load the iris dataset
    #
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)

    df[4] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.3,
                                                        random_state=42)
    #
    # Feature scaling
    #
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    #
    # Training / Test Dataframe
    #
    cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
    x_train_std = pd.DataFrame(x_train_std, columns=cols)
    x_test_std = pd.DataFrame(x_test_std, columns=cols)

    forest = RandomForestClassifier(n_estimators=500, random_state=42)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the model
    #
    forest.fit(x_train_std, y_train.values.ravel())

    # store the execution time for metrics
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    importances = forest.feature_importances_

    #
    # Prediction
    #
    y_pred_test = forest.predict(x_test_std)

    #
    # Sort the feature importance in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    # Confusion Matrix
    utils.confusion_matrix(y_test, y_pred_test)

    # Metrics
    utils.metrics_fi(y_test, y_pred_test, x_train, importances, sorted_indices, execution_time)

    # Epsilon Features
    utils.epsilon_features(x_train, importances, sorted_indices,
                           os.path.join(os.path.dirname(__file__), '../../resources/outputs', 'iris.txt'))
