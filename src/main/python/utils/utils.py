import itertools

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from mlflow import log_metric, log_param


def confusion_matrix(y_test, y_pred_test):
    print("Confusion Matrix:")
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred_test)
    print(cm)
    report = sklearn.metrics.classification_report(y_test, y_pred_test)
    print("Metrics Report:")
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


def metrics(y_test, y_pred_test, execution_time):
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
