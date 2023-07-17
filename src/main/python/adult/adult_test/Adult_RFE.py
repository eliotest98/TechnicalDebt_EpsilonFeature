import logging
import os
import time

import dagshub
import mlflow
import numpy as np
import pandas as pd
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

    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

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
    cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'native-country', 'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    # Store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the mode
    #
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, step=1)
    rfe.fit(X_train_std, y_train.values.ravel())

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    # Epsilon-features
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
    utils.metrics_adult(y_test, y_pred_test, execution_time)
