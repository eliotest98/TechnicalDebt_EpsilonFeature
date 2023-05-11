import pandas as pd
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow import log_metric, log_param
import mlflow
import logging
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import time

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
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.3, random_state=1)
    #
    # Feature scaling
    #
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    #
    # Training / Test Dataframe
    #
    cols = ['alcohal', 'malic_acid', 'ash', 'ash_alcalinity', 'magnesium', 'total_phenols', 'flavanoids',
            'nonflavanoids_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od_dilutedwines', 'proline']
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    forest = RandomForestClassifier(n_estimators=500,
                                    random_state=1)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the mode
    #
    forest.fit(X_train_std, y_train.values.ravel())

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    importances = forest.feature_importances_
    #
    # Sort the feature importance in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    feat_labels = df.columns[1:]

    # Log a params
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[sorted_indices[f]],
                                importances[sorted_indices[f]]))
        log_param(feat_labels[sorted_indices[f]], importances[sorted_indices[f]])

    # Metrics calculation
    tupla = performance([1, 2, 10], [1, 2, 20])

    # Log a metric; metrics can be updated throughout the run
    log_metric("accuracy", tupla[0])
    log_metric("precision", tupla[1])
    log_metric("recall", tupla[2])
    log_metric("execution_time", execution_time)

    # create a plot for see the data of features importance
    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
