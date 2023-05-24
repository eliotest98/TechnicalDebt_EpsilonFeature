import os

import pandas as pd
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import mlflow
import logging
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow import log_metric, log_param

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


def sort_dict_by_values(dictionary):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1])
    sorted_dict = {item[0]: item[1] for item in sorted_items}
    return sorted_dict


def sort_dict_by_oracle_order(dictionary, oracle_list):
    oracle_mapping = {val: i for i, val in enumerate(oracle_list)}
    return dict(sorted(dictionary.items(), key=lambda x: oracle_mapping.get(x[0], float('inf'))))


def delete_features(dictionary, threshold):
    keys_to_delete = [k for k, v in dictionary.items() if abs(v) > threshold + 0.1]
    for chiave in keys_to_delete:
        del dictionary[chiave]
    return dictionary


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

    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

    # Correlation threshold corresponding to variables with a low relationship to each other
    threshold = 0.0

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    epsilon_features = {}
    # Select characteristics with a correlation above the threshold
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                if epsilon_features.get(colname) is None:
                    epsilon_features[colname] = correlation_matrix.iloc[i, j]
                if epsilon_features.get(colname) > correlation_matrix.iloc[i, j]:
                    epsilon_features[colname] = correlation_matrix.iloc[i, j]

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    # order features by values
    epsilon_features = sort_dict_by_values(epsilon_features)
    print("Order by value:")
    i = 0
    for key, value in epsilon_features.items():
        # Esegui le operazioni desiderate su chiave e valore
        print("%2d) %s: %f" % (i + 1, key, value))
        i = i + 1
        log_param(key, value)

    # Delete Features for get epsilon-features
    epsilon_features = delete_features(epsilon_features, threshold)

    # Sorting epsilon features list by oracle
    epsilon_features = sort_dict_by_oracle_order(epsilon_features, epsilon_features_oracle)

    print("\nEpsilon-Features Detected:")
    i = 0
    for key, value in epsilon_features.items():
        # Esegui le operazioni desiderate su chiave e valore
        print("%2d) %s: %f" % (i + 1, key, value))
        i = i + 1

    # Metrics calculation
    tupla = performance(list(epsilon_features.keys()), epsilon_features_oracle)

    # Log a metric; metrics can be updated throughout the run
    log_metric("accuracy", tupla[0])
    log_metric("precision", tupla[1])
    log_metric("recall", tupla[2])
    log_metric("execution_time", execution_time)
