import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler


# Load the bank dataset
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'adult.csv'))
df = pd.read_csv(csv_path, sep=',')

# Remove white space before features name
df.columns = df.columns.str.strip()

# Remove white space before data values
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


cols = ['workclass','occupation', 'native-country']

# Replace "?" with the most frequence value for each columns
for c in cols:
    val_freq = df[c].mode()[0]
    df[c] = df[c].replace("?", val_freq)

'''
# Saving the balanced dataset to a new CSV file
pre_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'preprocessed_adult.csv'))
df.to_csv(pre_csv_path, index=False) '''


### ADULT UNDERSAMPLING ###

pd.value_counts(df['income']).plot.bar()
plt.title('Adult: Target class before Data BYalancing')
plt.xlabel('Income')
plt.ylabel('Frequency')
df['income'].value_counts()
plt.show()

# Separation of target classes
class_minority = df[df["income"] == ">50k"]
class_majority = df[df["income"] == "<=50k"]

# Coding categorical variables
categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                   'race', 'sex', 'native-country']
label_encoder = LabelEncoder()
for col in categorical:
    df[col] = label_encoder.fit_transform(df[col])

# Separation of features and target class
features = df.drop("income", axis=1)
target = df["income"]

# Applies random undersampling to the most frequent class
undersampler = RandomUnderSampler(random_state=42)
undersampled_features, undersampled_target = undersampler.fit_resample(features, target)

# Reconstitute the balanced DataFrame
balanced_df = pd.concat([undersampled_features, undersampled_target], axis=1)


#balanced_df.to_csv("new_adult.csv", index=False)
# Saving the balanced dataset to a new CSV file
pre_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/datasets', 'preprocessed_adult.csv'))
df.to_csv(pre_csv_path, index=False)

pd.value_counts(balanced_df['income']).plot.bar()
plt.title('Adult: Target class after Data Balancing')
plt.xlabel('Income')
plt.ylabel('Frequency')
balanced_df['income'].value_counts()
plt.show()