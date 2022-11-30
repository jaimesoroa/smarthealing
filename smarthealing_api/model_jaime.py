import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

df = pd.read_csv('../raw_data/datos.csv', delimiter = ";")
df = df.drop_duplicates()
index1 = df[ (df['numtreb'] == 0)].index
df.drop(index1 , inplace=True)
index2 = df[  (df['contracte'] == '-') ].index
df.drop(index2 , inplace=True)
index3 = df[ (df['cnae'] == 0) ].index
df.drop(index3 , inplace=True)
index4 = df[  (df['codipostal'] == '-')].index
df.drop(index4 , inplace=True)
index5 = df[ (df['tiempo_en_empresa'] == 0.00)].index
df.drop(index5 , inplace=True)
index6 = df[ (df['grupcoti'] == '-')].index
df.drop(index6 , inplace=True)
df_2 = df.sort_values('icd9').head(50000)
df_2['icd9'] = pd.to_numeric(df_2['icd9'])
df_2['codipostal'] = pd.to_numeric(df_2['codipostal'])
df_2['contracte'] = pd.to_numeric(df_2['contracte'])
df_2['grupcoti'] = pd.to_numeric(df_2['grupcoti'])
X = df_2.drop(columns = ['duracion_baja'])
y = df_2["duracion_baja"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train = X.head(30000)
y_train = y.head(30000)


def basic_model():
    model = LinearRegression()
    model_fit = model.fit(X_train, y_train)
    
    return model_fit