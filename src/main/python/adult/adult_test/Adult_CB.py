import logging
import os
import time

import dagshub
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        os.path.join(os.path.dirname(__file__), '../../../resources/datasets', 'preprocessed_adult.csv'))
    df = pd.read_csv(csv_path, sep=',')

    x = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'native-country', 'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']]
    y = df['income']

    # Correlation threshold corresponding to variables with a low relationship to each other
    threshold = 0.95

    correlation_matrix = df.corr()

    # Select upper triangle of correlation matrix
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Print Features high correlated
    if len(to_drop) != 0:
        print("Features high correlated")
    for x in range(len(to_drop)):
        print("%2d) %s" % (x + 1, to_drop[x]))
        if to_drop[x] == 'income':
            to_drop.remove('income')

    # Drop features
    df.drop(to_drop, axis=1, inplace=True)

    # Correlation Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.3,
                                                        random_state=42)

    #
    # Feature scaling
    #
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train_std = scaler.transform(x_train)
    X_test_std = scaler.transform(x_test)

    #
    # Training / Test Dataframe
    #
    cols = df.columns.to_list()
    cols.remove('income')
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the model
    #
    svc = SVC(kernel="linear", C=1)
    svc.fit(X_train_std, y_train.values.ravel())

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    #
    # Prediction
    #
    y_pred_test = svc.predict(X_test_std)

    # Confusion Metrix Creation
    utils.confusion_matrix(y_test, y_pred_test)

    # Metrics
    utils.metrics_adult(y_test, y_pred_test, execution_time)
