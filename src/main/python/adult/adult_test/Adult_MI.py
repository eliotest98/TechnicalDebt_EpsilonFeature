import logging
import os
import time

import dagshub
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train_std = scaler.transform(x_train)
    X_test_std = scaler.transform(x_test)

    #
    # Training / Test Dataframe
    #
    cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'native-country', 'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the model
    #
    selector = SelectKBest(mutual_info_classif, k="all")
    selected_features = selector.fit_transform(X_train_std, y_train.values.ravel())
    svc = SVC(kernel="linear", C=1)
    svc.fit(selected_features, y_train.values.ravel())

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    selected_indices = selector.get_support(indices=True)
    selected_features = X_train_std.columns[selected_indices]
    feature_scores = selector.scores_[selected_indices]
    sorted_indices = np.argsort(feature_scores)[::-1]
    sorted_features = selected_features[sorted_indices]
    sorted_scores = feature_scores[sorted_indices]

    print("\nFeatures Score:")
    for i, (feature, score) in enumerate(zip(sorted_features, sorted_scores), start=1):
        print(f"{i}) {feature} {score}")

    #
    # Prediction
    #
    y_pred_test = svc.predict(X_test_std)

    # Confusion Matrix
    utils.confusion_matrix(y_test, y_pred_test)

    # Metrics
    utils.metrics_adult(y_test, y_pred_test, execution_time)

    # Epsilon Features
    utils.epsilon_features_methods(x_train, feature_scores, sorted_indices,
                                   os.path.join(os.path.dirname(__file__), '../../../resources/outputs/mi', 'adult.txt'))
