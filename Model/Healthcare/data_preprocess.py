import numpy as np
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import csv

data_path="./heart_2022_no_nans.csv"

df = pd.read_csv(data_path)  

# data pre-processing
col_obj = df.select_dtypes('object').columns
le = LabelEncoder()
for col in col_obj:
    print("a new column: " + col)
    df[col] = le.fit_transform(df[col])
    for index, class_label in enumerate(le.classes_):
        print(f"{class_label} --> {index}")

X = df.drop(['HadHeartAttack'], axis=1)
Y = df['HadHeartAttack'] # 0 is "No", 1 is "Yes"
sc = StandardScaler()
X.iloc[:,:-1] = sc.fit_transform(X.iloc[:,:-1])

X_train, X_test, y_train, y_test = train_test_split(X, Y,stratify=Y, test_size=0.2, random_state=42)

## undersampling
rUs = RandomUnderSampler(random_state=42)
rUs_x_train, rUs_y_train = rUs.fit_resample(X_train, y_train)
print('original dataset shape:', X_train.shape, " -- ", y_train.shape)
print('Resample dataset shape', rUs_x_train.shape, " -- ", rUs_y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(rUs_x_train, rUs_y_train, stratify=rUs_y_train, test_size=0.2, random_state=42)

np.savetxt("undersampling_X_train_interpretability.csv",X_train,delimiter=",")
np.savetxt("undersampling_Y_train_interpretability.csv",y_train,delimiter=",",fmt="%d")
np.savetxt("undersampling_X_test_interpretability.csv",X_test,delimiter=",",fmt="%.5f")
np.savetxt("undersampling_Y_test_interpretability.csv",y_test,delimiter=",",fmt="%d")