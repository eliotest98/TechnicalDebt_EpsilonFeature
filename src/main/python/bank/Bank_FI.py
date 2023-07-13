import itertools
import os
import pandas as pd
import dagshub
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    precision_recall_fscore_support
from mlflow import log_param, log_metric
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


if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/eliotest98/Technical_Debt_Epsilon_Features.mlflow")
    dagshub.init("Technical_Debt_Epsilon_Features", "eliotest98", mlflow=True)

    #
    # Load the bank dataset
    #
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'balanced_bank.csv'))
    df = pd.read_csv(csv_path, sep=',')

    ''' categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'CLASS']

    label_encoder = LabelEncoder()
    for col in categorical:
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col]) '''

    x = df[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
            'duration', 'campaign', 'pdays', 'previous', 'poutcome']]
    y = df['CLASS']

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
    cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
            'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    X_train_std = pd.DataFrame(X_train_std, columns=cols)
    X_test_std = pd.DataFrame(X_test_std, columns=cols)

    forest = RandomForestClassifier(n_estimators=500, random_state=42)

    # store the execution time for metrics
    execution_time = round(time.time() * 1000)

    #
    # Train the mode
    #
    forest.fit(X_train_std, y_train.values.ravel())

    # execution time at the end of fit
    execution_time = (round(time.time() * 1000) - execution_time) / 1000

    # feature importances
    importances = forest.feature_importances_

    #
    # Prediction
    #
    y_pred_test = forest.predict(X_test_std)

    print("Confusion Matrix:")
    confusion_matrix = confusion_matrix(y_test, y_pred_test)
    print(confusion_matrix)
    report = classification_report(y_test, y_pred_test)
    print("Metrics Report:")
    print(report)

    #
    # Metrics
    #
    precision, recall, f1_score, support_val = precision_recall_fscore_support(y_test, y_pred_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    #
    # Sort the feature importance in descending order
    #
    sorted_indices = np.argsort(importances)[::-1]

    # Open of output file
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/outputs', 'bank.txt'))
    bankFile = open(file_name, "w")
    bankFile.write("Feature Importance:\n")

    #
    # Saving informations
    #
    for f in range(x_train.shape[1]):
        bankFile.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                      importances[sorted_indices[f]]))

    bankFile.write("\nEpsilon-Features:\n")
    truePositive = x_train.columns.shape[0] // 5
    if truePositive <= 0:
        truePositive = 1
    for f in range(x_train.shape[1] - truePositive, x_train.shape[1]):
        bankFile.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                      importances[sorted_indices[f]]))

    # Close of file
    bankFile.close()

    # Log of params
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                x_train.columns[sorted_indices[f]],
                                importances[sorted_indices[f]]))
        log_param(x_train.columns[sorted_indices[f]], importances[sorted_indices[f]])

    # Log of metrics
    for x in range(len(precision)):
        log_metric("precision class " + str(x), precision[x])
        log_metric("recall class " + str(x), recall[x])
    log_metric("accuracy", accuracy)
    log_metric("execution_time", execution_time)

    # create a plot for see the data of features importance
    plt.title('Feature Importance')
    plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center', data=x_train.values)
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # create a plot for see the data of confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(y_test)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Adding values on plot
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'), horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    # Show plot
    plt.tight_layout()
    plt.show()
