import os

from matplotlib import pyplot as plt


def load_oracle(folder, file_to_open):
    # Open of output file
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources/outputs/' + folder, file_to_open))
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


def calculate_metrics_for_methods(file_name):
    epsilon_features_o = load_oracle("oracle", file_name)
    epsilon_features_rfe = load_oracle("rfe", file_name)
    epsilon_features_cb = load_oracle("cb", file_name)
    epsilon_features_mi = load_oracle("mi", file_name)

    dataset_name = file_name.replace(".txt", "")

    print("\n\nMetrics for " + dataset_name)

    numberEpsilonFeatures = len(epsilon_features_o)
    numberEFRFE = len(epsilon_features_rfe)
    numberEFCB = len(epsilon_features_cb)
    numberEFMI = len(epsilon_features_mi)

    print("Number of Real Epsilon Features: " + str(numberEpsilonFeatures))
    print("Number of Epsilon Features for RFE: " + str(numberEFRFE))
    print("Number of Epsilon Features for CB: " + str(numberEFCB))
    print("Number of Epsilon Features for MI: " + str(numberEFMI))

    cbTP = 0
    rfeTP = 0
    miTP = 0
    for x in range(len(epsilon_features_o)):
        if epsilon_features_cb.__contains__(epsilon_features_o[x]):
            cbTP = cbTP + 1
        if epsilon_features_rfe.__contains__(epsilon_features_o[x]):
            rfeTP = rfeTP + 1
        if epsilon_features_mi.__contains__(epsilon_features_o[x]):
            miTP = miTP + 1

    print("Number of TP for RFE: " + str(rfeTP))
    print("Number of TP for CB: " + str(cbTP))
    print("Number of TP for MI: " + str(miTP))

    cbFP = numberEpsilonFeatures - numberEFRFE
    rfeFP = numberEpsilonFeatures - rfeTP
    miFP = numberEpsilonFeatures - miTP

    print("Number of FP for RFE: " + str(rfeFP))
    print("Number of FP for CB: " + str(cbFP))
    print("Number of FP for MI: " + str(miFP))

    cbFN = numberEpsilonFeatures - cbTP
    rfeFN = numberEpsilonFeatures - rfeTP
    miFN = numberEpsilonFeatures - miTP

    print("Number of FN for RFE: " + str(rfeFN))
    print("Number of FN for CB: " + str(cbFN))
    print("Number of FN for MI: " + str(miFN))

    methods = ['CB', 'RFE', 'MI']
    TP_values = [cbTP, rfeTP, miTP]
    FP_values = [cbFP, rfeFP, miFP]
    FN_values = [cbFN, rfeFN, miFN]

    plt.figure(figsize=(10, 6))

    plt.bar(methods, TP_values, label='True Positive (TP)', color='g', alpha=0.7)
    plt.bar(methods, FP_values, bottom=TP_values, label='False Positive (FP)', color='r', alpha=0.7)
    plt.bar(methods, FN_values, bottom=[TP + FP for TP, FP in zip(TP_values, FP_values)], label='False Negative (FN)',
            color='b', alpha=0.7)

    plt.xlabel('Methods')
    plt.ylabel('Counts')
    plt.title(
        'True Positive (TP), False Positive (FP), and False Negative (FN) for Different Methods in ' + dataset_name)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    calculate_metrics_for_methods("adult.txt")
    calculate_metrics_for_methods("bank.txt")
    calculate_metrics_for_methods("iris.txt")
    calculate_metrics_for_methods("raisin.txt")
    calculate_metrics_for_methods("wine.txt") 