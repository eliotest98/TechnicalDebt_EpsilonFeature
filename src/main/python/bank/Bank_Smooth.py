import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

    #
    # Load the bank dataset
    #
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'bank.csv'))
dataset = pd.read_csv(csv_path, sep=';')

''''data.head(3)

pd.value_counts(data['CLASS']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('CLASS')
plt.ylabel('Frequency')
data['CLASS'].value_counts()

plt.tight_layout()
plt.show()'''


# Dividi il dataset in feature e classe target
X = dataset.drop('CLASS', axis=1)
y = dataset['CLASS']

# Identifica le colonne con dati categorici
categorical_columns = X.select_dtypes(include=['object']).columns

# Identifica le colonne con dati numerici
numeric_columns = X.select_dtypes(include=['int', 'float']).columns

# Applica la codifica one-hot alle colonne categoriche
ct = ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Ricombina le variabili numeriche con il dataset codificato
X_encoded = pd.DataFrame(X_encoded, columns=list(ct.get_feature_names_out()))
X_encoded[numeric_columns] = X[numeric_columns]

# Applica SMOTE per il bilanciamento delle classi
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X_encoded, y)

# Crea un nuovo dataframe bilanciato
balanced_dataset = pd.concat([pd.DataFrame(X_balanced), y_balanced], axis=1)

# Salva il dataframe bilanciato in un nuovo file
balanced_dataset.to_csv('smote-bank.csv', index=False)

