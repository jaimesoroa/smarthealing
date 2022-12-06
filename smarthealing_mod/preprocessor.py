import pandas as pd
import numpy as np
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import os

dirname = os.path.dirname(__file__)

def clean_data(data):
    
    # 1. Handle Null Values
    # CNAE (only 482)
    data = data.query('cnae != 0')

    # CodiPostal (Most Frequent)
    data.loc[:, 'codipostal'] = data.loc[:, 'codipostal'].replace('-', (data.loc[:, 'codipostal']).mode()[0])

    # Contracte (Delete few wrongs and then Most Frequent)
    data = data[data['contracte'] != '001']
    data = data[data['contracte'] != '019']
    data = data[data['contracte'] != '999']

    mf_contracte = data['contracte'].mode()[0]
    data['contracte'] = data['contracte'].replace('000', mf_contracte).replace('-', mf_contracte)

    # Grupcoti (Most Frequent)
    mf_grupcoti = data['grupcoti'].mode()[0]
    data['grupcoti'] = data['grupcoti'].replace('-', mf_grupcoti).replace('0', mf_grupcoti)

    # 2. Data Cleaning
    # ContadorBajasCCC (non-sense)
    data.drop('ContadorBajasCCC', axis = 1, inplace = True)

    # ContadorBajasDNI vs ContadordiasBajasDNI
    data = data.query('ContadorBajasDNI <= ContadordiasBajasDNI')
    
    return data

def outliers(data):
    
    # Remove some outliers

    data.drop(data[(data['ContadorBajasDNI']>25)].index, axis = 0, inplace = True)
    data.drop(data[(data['numtreb']>3200)].index, axis = 0, inplace = True)
    data.drop(data[(data['ContadordiasBajasDNI']>750)].index, axis = 0, inplace = True)
    
    return data


def categorize(data):
    
    # Categorize CNAE
    cnae_filename = os.path.join(dirname, '../raw_data/Tabla_cnae.xlsx')
    c = pd.read_excel(cnae_filename, usecols = ['GRUPO', 'COD_CNAE2009'])
    c.rename(columns = {'GRUPO': 'cnae_cat', 'COD_CNAE2009': 'cnae'}, inplace = True)
    c['cnae_cat'] = c['cnae_cat'].map(lambda x: ord(x) - 64)
    c = c.query('cnae.str.isnumeric()').astype('int')
    data = data.merge(c, how = 'left', on = 'cnae')

    # Sickness Types
    block_ends = [140,240,280,290,320,390,460,520,580,630,680,710,740,760,780,800,1000]
    blocks = [[i+1, s] for i, s in enumerate(block_ends)]
    special_cases = ['E','V','M']

    def make_icd9_cat(row):
        first_str = row.icd9[0]
        if first_str == 'E':
            return 18
        if first_str == 'V':
            return 19
        if first_str == 'M':
            return 20
        else:
            code = float(row.icd9)
            for block in blocks:
                if code < block[1]:
                    return block[0]
    data['icd9_cat'] = data.apply(lambda row: make_icd9_cat(row), axis=1)

    # Sickness Wrong Values
    data = data.query('sexo != 1 | icd9_cat != 11')
    data = data[~((data['sexo'] == 1) & (data['icd9'].str[:2].isin(['61', '62'])))]
    data = data[~((data['sexo'] == 2) & (data['icd9'].str[:2] == '60'))]

    # Post Code Categories.
    data['codipostal_cat'] = data['codipostal'].str[:2]

    # Time in company
    # Add 0.01 to have the upper limit. (0.00 means up to 3 days)
    data['proporcion_baja'] = (data['duracion_baja'] / (365 * (data['tiempo_en_empresa'] + 0.01)))
    by_wrong_proportion = data.sort_values(by = 'proporcion_baja', axis = 0, ascending = False)
    data = data.query('proporcion_baja <= 1')
    data.drop(columns = 'proporcion_baja', axis = 1, inplace = True)

    # Month not required
    data.drop(columns = 'mes_baja', axis = 1, inplace = True)

    # Cyclic Calendar
    data.loc[:, 'time'] = (2 * pi * (data.loc[:, 'epiweek'] - 1) + ((data.loc[:, 'diasemana'] - 1) / 7)) / 52
    data.loc[:, 'sin_time'] = np.sin(data.loc[:, 'time'])
    data.loc[:, 'cos_time'] = np.cos(data.loc[:, 'time'])
    data.drop(['time', 'diasemana', 'epiweek'], axis = 1, inplace = True)
    
    return data


def data_for_ml(data):

    # 3. Prepare Data for ML
    data.drop(['cnae', 'icd9', 'codipostal', 'contracte'], axis = 1, inplace = True)

    # sample = data.sample(200000, random_state = 8)
    sample = data
    X_train, X_test, y_train, y_test = train_test_split(
                            sample.drop('duracion_baja', axis = 1), sample['duracion_baja'], test_size=0.3)

    def prepare_data(X_train, X_test):
        # One hot Encode Categorical Features.
        ohe_cols = ['sexo', 'recaida', 'grupcoti', 'pluriempleo',
                    'cnae_cat', 'icd9_cat', 'codipostal_cat']
        for col in ohe_cols:
            ohe = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
            ohe.fit(X_train[[col]])
            for X_t in [X_train, X_test]:
                X_t[ohe.get_feature_names_out()] = ohe.transform(X_t[[col]])
                X_t.drop(col, axis = 1, inplace = True)


        # Scale numerical features.
        robust_cols = ['ContadorBajasDNI', 'ContadordiasBajasDNI', 'numtreb', 'tiempo_en_empresa']
        for col in robust_cols:
            rb_scaler = RobustScaler()
            rb_scaler.fit(X_train[[col]])
            for X_t in [X_train, X_test]:
                X_t[col] = rb_scaler.transform(X_t[[col]])

        st_scaler = StandardScaler()
        st_scaler.fit(X_train[['edad']])
        for X_t in [X_train, X_test]:
            X_t['edad'] = st_scaler.transform(X_t[['edad']])

        return X_train, X_test, ohe, rb_scaler, st_scaler

    X_train_p, X_test_p, ohe, rb_scaler, st_scaler = prepare_data(X_train, X_test)
    
    return X_train_p, X_test_p, y_train, y_test, ohe, rb_scaler, st_scaler


def prepare_pred(X_new, ohe, rb_scaler, st_scaler):
    # 3. Prepare Data for ML
    X_new.drop(['cnae', 'icd9', 'codipostal', 'contracte'], axis = 1, inplace = True)
    # One hot Encode Categorical Features.
    ohe_cols = ['sexo', 'recaida', 'grupcoti', 'pluriempleo',
                'cnae_cat', 'icd9_cat', 'codipostal_cat']
    for col in ohe_cols:
        for X_t in [X_new]:
            X_t[ohe.get_feature_names_out()] = ohe.transform(X_t[[col]])
            X_t.drop(col, axis = 1, inplace = True)
            
    # Scale numerical features.
    robust_cols = ['ContadorBajasDNI', 'ContadordiasBajasDNI', 'numtreb', 'tiempo_en_empresa']
    for col in robust_cols:
        for X_t in [X_new]:
            X_t[col] = rb_scaler.transform(X_t[[col]])
        
    for X_t in [X_new]:
        X_t['edad'] = st_scaler.transform(X_t[['edad']])
        
    return X_new
    
    
def preprocess_train(data):
    data = clean_data(data)
    data = outliers(data)
    data = categorize(data)
    X_train, X_test, y_train, y_test, ohe, rb_scaler, st_scaler = data_for_ml(data)
    
    return X_train, X_test, y_train, y_test, ohe, rb_scaler, st_scaler


def preprocess_new(X_new, ohe, rb_scaler, st_scaler):
    X_new = categorize(X_new)
    X_new_prepr = prepare_pred(X_new, ohe, rb_scaler, st_scaler)
    
    return X_new_prepr
