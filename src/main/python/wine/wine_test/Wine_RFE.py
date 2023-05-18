import os
import pandas as pd
import dagshub
from mlflow import log_param, log_metric
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import logging
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def performance(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='macro')
    recall = recall_score(actual, pred, average='macro')
    return accuracy, precision, recall


def load_oracle(file_to_open):
    # Open of output file
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/outputs', file_to_open))
    file = open(file_name, "r")
    start = False
    # Load of Oracle Epsilon Features
    epsilon_features_oracle = []
    for riga in file:
        riga = riga.strip()
        if ("Epsilon-Features" in riga) | start:
            if start:
                epsilon_features_oracle.append(riga.split(":")[0])
            start = True
    return epsilon_features_oracle


if __name__ == "__main__":

    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    #
    # Load the wine datasets
    #
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data)
    df[13] = wine.target
    df.columns = ['alcohal', 'malic_acid', 'ash', 'ash_alcalinity', 'magnesium', 'total_phenols', 'flavanoids',
                  'nonflavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od_dilutedwines', 'proline',
                  'class']
    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.3, random_state=1)
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
    cols = ['alcohal', 'malic_acid', 'ash', 'ash_alcalinity', 'magnesium', 'total_phenols', 'flavanoids',
            'nonflavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od_dilutedwines', 'proline']
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    # Load of oracle epsilon features
    epsilon_features_oracle = load_oracle("wine.txt")

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the mode
    #
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=len(epsilon_features_oracle), step=1)
    rfe.fit(x_train, y_train.values.ravel())

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    importances = rfe.support_
    epsilon_features = []
    print("Epsilon-Features Detected:")
    i = 0
    for feature, selected in zip(x_train.columns, importances):
        if selected:
            epsilon_features.append(feature)
            print("%2d) %s" % (i + 1, feature))
            i = i + 1

    #
    # Sort the feature in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    feat_labels = df.columns[1:]

    # Log a params
    for f in range(x_train.shape[1]):
        log_param(x_train.columns[sorted_indices[f]], importances[sorted_indices[f]])

    # Order lexicographical of lists
    epsilon_features.sort()
    epsilon_features_oracle.sort()
    # Metrics calculation
    tupla = performance(epsilon_features, epsilon_features_oracle)

    # Log a metric; metrics can be updated throughout the run
    log_metric("accuracy", tupla[0])
    log_metric("precision", tupla[1])
    log_metric("recall", tupla[2])
    log_metric("execution_time", execution_time)

    # create a plot for see the data of features
    plt.title('Epsilon Features')
    plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
