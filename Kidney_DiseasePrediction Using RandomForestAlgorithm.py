## Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

##Loading dataset
df = pd.read_csv("kidney_disease.csv")

print(df.head())

## There are 26 features and 400 observations in dataset

print(df.shape)

#Let's check missing values in dataset
print(df.isnull().sum())

print(df.info())

print(df.columns)

##We rename the columns for meaningful names
col_dict = {
    "bp": "blood_pressure",
    "sg": "specific_gravity",
    "al": "albumin",
    "su": "sugar",
    "rbc": "red_blood_cells",
    "pc": "pus_cell",
    "pcc": "pus_cell_clumps",
    "ba": "bacteria",
    "bgr": "blood_glucose_random",
    "bu": "blood_urea",
    "sc": "serum_creatinine",
    "sod": "sodium",
    "pot": "potassium",
    "hemo": "hemoglobin",
    "pcv": "packed_cell_volume",
    "wc": "white_blood_cell_count",
    "rc": "red_blood_cell_count",
    "htn": "hypertension",
    "dm": "diabetes_mellitus",
    "cad": "coronary_artery_disease",
    "appet": "appetite",
    "pe": "pedal_edema",
    "ane": "anemia"
    }

print(df.rename(columns= col_dict, inplace = True))

##Let's check the unique target values in dataset
print(df.classification.value_counts(normalize = True))


df['classification'] = df['classification'].replace("ckd\t", "ckd")

for c in df.columns:
    print(c, df[c].unique())

#df['classification'].value_counts(normalize = True)


df["diabetes_mellitus"] = df["diabetes_mellitus"].replace({"\tno": "no", "\tyes": "yes", " yes": "yes"})

df["coronary_artery_disease"] = df["coronary_artery_disease"].replace({"\tno": "no"})

df["red_blood_cell_count"] = df["red_blood_cell_count"].replace({"\t?": np.nan})

df["white_blood_cell_count"] = df["white_blood_cell_count"].replace({"\t?": np.nan, "\t8400": "8400", "\t6200": "6200"})

df["packed_cell_volume"] = df["packed_cell_volume"].replace({"\t43": "43", "\t?": np.nan})

##Let's chek again unique values in dataset
#for c in df.columns:
#    print(c, df[c].unique())


df['red_blood_cells'] = df["red_blood_cells"].replace({"normal": 1, 'abnormal': 0})
df["pus_cell"] = df["pus_cell"].replace({"normal": 1, "abnormal": 0})
df["pus_cell_clumps"] = df["pus_cell_clumps"].replace({"notpresent": 0, "present": 1})
df["bacteria"] = df["bacteria"].replace({"notpresent":0, "present": 1})
df["hypertension"] = df["hypertension"].replace({"yes":1, "no": 0})
df["diabetes_mellitus"] = df["diabetes_mellitus"].replace({"yes":1, "no":0})
df["coronary_artery_disease"] = df["coronary_artery_disease"].replace({"yes":1, "no":0})
df["appetite"] = df["appetite"].replace({"poor": 0, "good": 1})
df["pedal_edema"] = df["pedal_edema"].replace({"no": 0, "yes": 1})
df["anemia"] = df["anemia"].replace({"no": 0, "yes": 1})
df["classification"] = df["classification"].replace({"ckd":1, "notckd": 0})

print(df.isnull().sum())

#### We can see that some features have missing values in dataset

from sklearn.impute import KNNImputer

imputer = KNNImputer( n_neighbors=5,weights='uniform',metric='nan_euclidean')

##Now we spit dataset into independent variable and dependent variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


print(X.drop(columns='id', axis = 1, inplace= True))

print(X.head())

X = imputer.fit_transform(X)

X = pd.DataFrame(X, columns=[['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
       'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
       'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
       'potassium', 'hemoglobin', 'packed_cell_volume',
       'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
       'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
       'pedal_edema', 'anemia']])


##Now let's check missing values in dataset
X.isnull().sum()

#### We can see that there is no missing values in dataset

print(X.shape, y.shape)

cols = ["blood_pressure", "albumin","specific_gravity", "red_blood_cells", "pus_cell", "pus_cell_clumps", "sugar","hypertension", "anemia"]
feature = X[cols]
print(feature.head())
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
##Now we split dataset into training data and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature,y, test_size = 0.2, random_state = 1234)

##RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)

model_pred = classifier.predict(X_test)
print("Accuracy score :\n",accuracy_score(y_test, model_pred))

print("Classification report:\n",classification_report(y_test, model_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, model_pred))

import joblib
joblib.dump(classifier, "kidney_model.pkl")






