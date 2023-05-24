import os
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


def sort_list_by_oracle_order(list_to_sort, oracle_list):
    oracle_mapping = {val: i for i, val in enumerate(oracle_list)}
    return sorted(list_to_sort, key=lambda x: oracle_mapping.get(x, float('inf')))


if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    #
    # Load the raisin dataset
    #
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets', 'raisin.csv'))
    df = pd.read_csv(csv_path, sep=';')

    categorical = ['Class']

    label_encoder = LabelEncoder()
    for col in categorical:
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])

    x = df[['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
            'ConvexArea', 'Extent', 'Perimeter']]
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()

    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

    # Load of oracle epsilon features
    epsilon_features_oracle = load_oracle("raisin.txt")

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
    # Sort the feature importance in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    # Log a params
    for f in range(x_train.shape[1]):
        log_param(x_train.columns[sorted_indices[f]], importances[sorted_indices[f]])

    # Sorting epsilon features list by oracle
    epsilon_features = sort_list_by_oracle_order(epsilon_features, epsilon_features_oracle)

    # Metrics calculation
    tupla = performance(epsilon_features, epsilon_features_oracle)

    # Log a metric; metrics can be updated throughout the run
    log_metric("accuracy", tupla[0])
    log_metric("precision", tupla[1])
    log_metric("recall", tupla[2])
    log_metric("execution_time", execution_time)

    plt.title('Feature Importance')
    plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center', data=x_train.values)
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
