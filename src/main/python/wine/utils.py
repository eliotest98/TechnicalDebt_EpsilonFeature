import itertools
import os

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from mlflow import log_metric, log_param


def confusion_matrix(y_test, y_pred_test):
    print("\nConfusion Matrix:")
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred_test)
    print(cm)
    report = sklearn.metrics.classification_report(y_test, y_pred_test)
    print("\nMetrics Report:")
    print(report)

    # Create a plot for see the data of confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(y_test)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Adding values on plot
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Show plots
    plt.tight_layout()
    plt.show()


# Metrics for the three methods
def metrics_mi_rfe_c(y_test, y_pred_test, execution_time):
    #
    # Other metrics
    #
    precision, recall, f1_score, support_val = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred_test)

    singleton = list(set(y_pred_test))

    # Log of params
    for x in range(len(singleton)):
        log_param(str(singleton[x]), "Class Type")

    # Log of metrics
    for x in range(len(precision)):
        log_metric("precision class " + str(x), precision[x])
        log_metric("recall class " + str(x), recall[x])
    log_metric("accuracy", accuracy)
    log_metric("execution_time", execution_time)


# Metrics for Feature Importance
def metrics_fi(y_test, y_pred_test, x_train, importances, sorted_indices, execution_time):
    #
    # Other metrics
    #
    precision, recall, f1_score, support_val = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred_test)

    print("\nFeature Importance:")
    # Log of params
    for f in range(x_train.shape[1]):
        log_param(x_train.columns[sorted_indices[f]], importances[sorted_indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30,
                                x_train.columns[sorted_indices[f]],
                                importances[sorted_indices[f]]))

    # Log of metrics
    for x in range(len(precision)):
        log_metric("precision class " + str(x), precision[x])
        log_metric("recall class " + str(x), recall[x])
    log_metric("accuracy", accuracy)
    log_metric("execution_time", execution_time)


def metrics_adult(y_test, y_pred_test, execution_time):
    #
    # Other metrics
    #
    precision, recall, f1_score, support_val = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred_test)

    singleton = list(set(y_pred_test))

    # Log of params
    for x in range(len(singleton)):
        log_param(str(x), "Class " + singleton[x])

    # Log of metrics
    for x in range(len(precision)):
        log_metric("precision class " + str(x), precision[x])
        log_metric("recall class " + str(x), recall[x])
    log_metric("accuracy", accuracy)
    log_metric("execution_time", execution_time)


# Metrics for Epsilon-Feature
def epsilon_features(x_train, importances, sorted_indices, path):
    # Open of output file
    file_name = os.path.abspath(path)
    file = open(file_name, "w")

    #
    # Saving informations
    #
    file.write("Feature Importance:\n")
    for f in range(x_train.shape[1]):
        file.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                 importances[sorted_indices[f]]))

    file.write("\nEpsilon-Features:\n")
    print("\nEpsilon-Features:")
    true_positive = x_train.columns.shape[0] // 5
    if true_positive <= 0:
        true_positive = 1
    i = 1
    for f in range(x_train.shape[1] - true_positive, x_train.shape[1]):
        file.write("%s: %f\n" % (x_train.columns[sorted_indices[f]],
                                 importances[sorted_indices[f]]))
        print("%2d) %-*s %f" % (i, 30,
                                x_train.columns[sorted_indices[f]],
                                importances[sorted_indices[f]]))
        i = i + 1

        # Close of file
    file.close()

    # create a plot for see the data of features importance
    plt.title('Feature Importance')
    plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
