import pandas as pd
import numpy as np
from math import pi
import os

# path = os.path.join

# 0. Load Data and Drop Duplicates
def load_data():
    data = pd.read_csv('../raw_data/datos.csv', delimiter = ';', low_memory = False)
    data.drop_duplicates(inplace = True)
    return data

def preprocess(data):

    # 1. Handle Null Values
    # CNAE (only 482)
    data = data.query('cnae != 0')

    # CodiPostal (Most Frequent)
    data['codipostal'] = data['codipostal'].replace('-', (data['codipostal']).mode()[0])

    # Contracte (Delete few wrongs and then Most Frequent)
    data = data[data['contracte'] != '001']
    data = data[data['contracte'] != '019']

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

    # Categorize CNAE
    c = pd.read_excel('Tabla_cnae.xlsx', usecols = ['GRUPO', 'COD_CNAE2009'])
    c.rename(columns = {'GRUPO': 'cnae_cat', 'COD_CNAE2009': 'cnae'}, inplace = True)
    c['cnae_cat'] = c['cnae_cat'].map(lambda x: ord(x) - 64)
    c = c.query('cnae.str.isnumeric()').astype('int')
    data = data.merge(c, how = 'left', on = 'cnae')

    # Sickness Types
    block_ends = [140,240,280,290,320,390,460,520,580,630,680,710,740,760,780,800,1000]
    blocks = [[i+1, s] for i, s in enumerate(block_ends)]
    # special_cases = ['E','V','M']

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

    # Time in company
    # Add 0.01 to have the upper limit. (0.00 means up to 3 days)
    data['proporcion_baja'] = (data['duracion_baja'] / (365 * (data['tiempo_en_empresa'] + 0.01)))
    by_wrong_proportion = data.sort_values(by = 'proporcion_baja', axis = 0, ascending = False)
    data = data.query('proporcion_baja <= 1')
    data.drop('proporcion_baja', axis = 1, inplace = True)

    # Month not required
    data.drop('mes_baja', axis = 1, inplace = True)

    # Cyclic Calendar
    data['time'] = (2 * pi * (data['epiweek'] - 1) + ((data['diasemana'] - 1) / 7)) / 52
    data['sin_time'] = np.sin(data['time'])
    data['cos_time'] = np.cos(data['time'])
    data.drop(['time', 'diasemana', 'epiweek'], axis = 1, inplace = True)

    # Categorise postal code with the first 2 numbers
    data['codipostal_cat'] = data['codipostal'].str[:2]

    return data