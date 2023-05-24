import os
import pandas as pd
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow import log_param
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


def performance(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='macro')
    recall = recall_score(actual, pred, average='macro')
    return accuracy, precision, recall


if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    #
    # Load the raisin dataset
    #
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'raisin.csv'))
    df = pd.read_csv(csv_path, sep=';')

    print(df.columns)

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

    forest = RandomForestClassifier(n_estimators=500, random_state=42)

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

    # Open of output file
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/outputs', 'raisin.txt'))
    raisinFile = open(file_name, "w")
    raisinFile.write("Feature Importance:\n")

    # Log a params
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                x_train.columns[sorted_indices[f]],
                                importances[sorted_indices[f]]))
        log_param(x_train.columns[sorted_indices[f]], importances[sorted_indices[f]])
        raisinFile.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                      importances[sorted_indices[f]]))

    raisinFile.write("\nEpsilon-Features:\n")
    truePositive = x_train.columns.shape[0] // 5
    if truePositive <= 0:
        truePositive = 1
    for f in range(x_train.shape[1] - truePositive, x_train.shape[1]):
        raisinFile.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                      importances[sorted_indices[f]]))

    # Close of file
    raisinFile.close()

    # Metrics calculation
    tupla = performance([1, 2, 10], [1, 2, 20])

    # Log a metric; metrics can be updated throughout the run
    # log_metric("accuracy", tupla[0])
    # log_metric("precision", tupla[1])
    # log_metric("recall", tupla[2])
    # log_metric("execution_time", execution_time)

    plt.title('Feature Importance')
    plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center', data=x_train.values)
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
