import os

import pandas as pd
from sklearn import datasets
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow import log_metric, log_param
import mlflow
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def performance(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='macro')
    recall = recall_score(actual, pred, average='macro')
    return accuracy, precision, recall


if __name__ == "__main__":

    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    #
    # Load the iris datasets
    #
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)

    df[4] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.3, random_state=1)
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

    forest = RandomForestClassifier(n_estimators=500,
                                    random_state=1)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)
    #
    # Train the mode
    #
    forest.fit(x_train_std, y_train.values.ravel())
    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    importances = forest.feature_importances_
    #
    # Sort the feature importance in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    feat_labels = df.columns[1:]

    # Open of output file
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/outputs', 'iris.txt'))
    adultFile = open(file_name, "w")
    adultFile.write("Feature Importance:\n")

    # Log a params
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                x_train.columns[sorted_indices[f]],
                                importances[sorted_indices[f]]))
        log_param(x_train.columns[sorted_indices[f]], importances[sorted_indices[f]])
        adultFile.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                      importances[sorted_indices[f]]))

    adultFile.write("\nEpsilon-Features:\n")
    truePositive = x_train.columns.shape[0] // 5
    if truePositive <= 0:
        truePositive = 1
    for f in range(x_train.shape[1] - truePositive, x_train.shape[1]):
        adultFile.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                      importances[sorted_indices[f]]))

    # Close of file
    adultFile.close()

    # Metrics calculation
    tupla = performance([1, 2, 10], [1, 2, 20])

    # Log a metric; metrics can be updated throughout the run
    # log_metric("accuracy", tupla[0])
    # log_metric("precision", tupla[1])
    # log_metric("recall", tupla[2])
    # log_metric("execution_time", execution_time)

    plt.title('Feature Importance')
    plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
