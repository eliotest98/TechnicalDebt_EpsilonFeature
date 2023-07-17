import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the bank dataset
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'bank.csv'))
df = pd.read_csv(csv_path, sep=';')

pd.value_counts(df['CLASS']).plot.bar()
plt.title('Bank: Target class before Data Balancing')
plt.xlabel('CLASS')
plt.ylabel('Frequency')
df['CLASS'].value_counts()

plt.show()

categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'CLASS']

# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in categorical:
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])

# Balancing the dataset using SMOTE
features = df.drop('CLASS', axis=1)
target = df['CLASS']
smote = SMOTE()
balanced_features, balanced_target = smote.fit_resample(features, target)

# Creating a new DataFrame with balanced data
balanced_df = pd.concat([balanced_features, balanced_target], axis=1)

# Saving the balanced dataset to a new CSV file
balanced_csv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'balanced_bank.csv'))
balanced_df.to_csv(balanced_csv_path, index=False)

pd.value_counts(balanced_df['CLASS']).plot.bar()
plt.title('Bank: Target class before Data Balancing')
plt.xlabel('CLASS')
plt.ylabel('Frequency')
balanced_df['CLASS'].value_counts()
plt.show()
