import os
import pandas as pd
import dagshub
from mlflow import log_param, log_metric
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.feature_selection import mutual_info_classif

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


def sort_lists(list_to_sort, oracle_list):
    elementi_comuni = [item for item in list_to_sort if item in oracle_list]

    list_to_sort = elementi_comuni + [item for item in list_to_sort if item not in elementi_comuni]
    oracle_list = elementi_comuni + [item for item in oracle_list if item not in elementi_comuni]
    return list_to_sort, oracle_list


if __name__ == "__main__":

    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    #
    # Load the adult dataset
    #
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets', 'adult.csv'))
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #
    # Feature scaling
    #
    scaler = StandardScaler()

    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

    # Load of oracle epsilon features
    epsilon_features_oracle = load_oracle("adult.txt")

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the mode
    #
    importances = mutual_info_classif(x, y)

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    #
    # Sort the feature in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    print("Feature Importance Detected:")
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                x_train.columns[sorted_indices[f]],
                                importances[sorted_indices[f]]))
        log_param(x_train.columns[sorted_indices[f]], importances[sorted_indices[f]])

    epsilon_features = []
    print("Epsilon Features Detected:")
    for f in range(x_train.shape[1] - len(epsilon_features_oracle), x_train.shape[1]):
        print("%2d) %s" % (f + 1, x_train.columns[sorted_indices[f]]))
        epsilon_features.append(x_train.columns[sorted_indices[f]][1:])

    epsilon_features, epsilon_features_oracle = sort_lists(epsilon_features, epsilon_features_oracle)

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
