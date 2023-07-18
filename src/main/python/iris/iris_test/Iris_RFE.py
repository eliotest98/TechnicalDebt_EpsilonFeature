import logging
import os
import time

import dagshub
import mlflow
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    X_train_std = sc.transform(x_train)
    X_test_std = sc.transform(x_test)

    #
    # Training / Test Dataframe
    #
    cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the mode
    #
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, step=1)
    rfe.fit(X_train_std, y_train.values.ravel())

    # store the execution time for metrics
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    importances = rfe.support_

    #
    # Sort features by rfe ranking
    #
    sorted_indices = np.argsort(rfe.ranking_)[::-1]

    print("\nFeature Ranking:")
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %.3f" % (f + 1, 30,
                                  x_train.columns[sorted_indices[f]], importances[sorted_indices[f]]),
              "- Rank:", rfe.ranking_[sorted_indices[f]])

    #
    # Prediction
    #
    y_pred_test = rfe.predict(X_test_std)

    # Confusion Metrix Creation
    utils.confusion_matrix(y_test, y_pred_test)

    # Metrics
    utils.metrics_mi_rfe_c(y_test, y_pred_test, execution_time)

    # Epsilon Features
    utils.epsilon_features_methods(x_train, importances, np.argsort(rfe.ranking_),
                                   os.path.join(os.path.dirname(__file__), '../../../resources/outputs/rfe',
                                                'iris.txt'))
