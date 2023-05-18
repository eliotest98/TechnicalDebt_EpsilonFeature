import os
import pathlib
import pickle

import pandas as pd
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow import log_metric, log_param
import mlflow
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    #
    # Load the adult datasets
    #
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'adult.csv'))
    df = pd.read_csv(csv_path, sep=',')

    categorical = [' workclass', ' education', ' marital-status', ' occupation', ' relationship',
                   ' race', ' sex', ' native-country']
    label_encoder = LabelEncoder()
    for col in categorical:
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])

    x = df[[' workclass', ' education', ' marital-status', ' occupation', ' relationship',
            ' race', ' sex', ' native-country', 'age', ' fnlwgt', ' capital-gain', ' capital-loss', ' hours-per-week']]
    y = df[' income']

    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    #
    # Feature scaling
    #
    scaler = StandardScaler()

    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

    forest = RandomForestClassifier(n_estimators=500,
                                    random_state=1)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the mode
    #
    forest.fit(x_train, y_train.values.ravel())

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    importances = forest.feature_importances_
    #
    # Sort the feature importance in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    feat_labels = df.columns[1:]

    # Open of output file
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/outputs', 'adult.txt'))
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

    # create a plot for see the data of features importance
    plt.title('Feature Importance')
    plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
