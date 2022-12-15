import pandas as pd
import numpy as np
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import xgboost as xgb
import os
import pickle
from imblearn.over_sampling import SMOTE

dirname = os.path.dirname(__file__)
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))

def load_data(csv_file = os.path.join(dirname, '../raw_data/datos.csv')):
    '''
    Loads the csv file with pandas.
    '''
    data = pd.read_csv(csv_file, delimiter = ';', low_memory = False)
    data.drop_duplicates(inplace = True)
    # Remove not needed columns.
    data.drop(['ContadorBajasCCC', 'sexo', 'mes_baja'], axis = 1, inplace = True)

    return data

def null_data(data):
    '''
    Handles hidden null values from the data.
    '''
    # CNAE
    data = data.query('cnae != 0')
    # Codipostal
    mf_codipostal = data['codipostal'].mode()[0]
    data['codipostal'] = data['codipostal'].replace('-', mf_codipostal)
    # Contracte
    data = data[data['contracte'] != '001']
    data = data[data['contracte'] != '019']
    data = data[data['contracte'] != '999']
    mf_contracte = data['contracte'].mode()[0]
    data['contracte'] = data['contracte'].replace('000', mf_contracte).replace('-', mf_contracte)
    # Grupcoti
    mf_grupcoti = data['grupcoti'].mode()[0]
    data['grupcoti'] = data['grupcoti'].replace('-', mf_grupcoti).replace('0', mf_grupcoti)

    return data

def bad_counter(df):
    df = df.query('ContadorBajasDNI <= ContadordiasBajasDNI')
    df.drop('ContadorBajasDNI', axis = 1, inplace = True)
    # if df.empty:
    #     return ValueError('The number of sick times must be <= than total days sick.')

    return df

def wrong_time_in_company(df_x, df_y):
    df = pd.concat([df_x, df_y], axis = 1)
    # Add 0.01 to have the upper limit. (0.00 means up to 3 days)
    df['proporcion_baja'] = (df['duracion_baja'] / (365 * (df['tiempo_en_empresa'] + 0.01)))
    df = df.query('proporcion_baja <= 1')
    df.drop('proporcion_baja', axis = 1, inplace = True)

    return df.drop('duracion_baja', axis = 1), df['duracion_baja']

# Feature Engineering Functions.

def categorize_cnae(df, tabla_cnae = os.path.join(dirname, '../raw_data/Tabla_cnae.xlsx')):
    c = pd.read_excel(tabla_cnae, usecols = ['GRUPO', 'COD_CNAE2009'])
    c.rename(columns = {'GRUPO': 'cnae_cat', 'COD_CNAE2009': 'cnae'}, inplace = True)

    def letter_to_number(x):
        return ord(x) - 64
    c['cnae_cat'] = c['cnae_cat'].apply(letter_to_number)

    c = c.query('cnae.str.isnumeric()').astype('int')
    df = df.merge(c, how = 'left', on = 'cnae')
    df.drop('cnae', axis = 1, inplace = True)

    return df, c

def categorize_icd9_train(df_x, df_y):
    def make_icd9_cat(row):
        first_str = row.icd9[0]
        if first_str == 'E':
            return 1010
        if first_str == 'V':
            return 1020
        if first_str == 'M':
            return 1030
        else:
            code = float(row.icd9)
            return round(code)

    def group_icd9_cat_by_target(df):
        """
        Input a datafreame with at least 'icd9' and 'duracion_baja'.
        Create new colum with main categories of 'icd9' (only int part), colum name = 'icd9_cat'
        Group by 'icd9_cat' and take the 'duracion_baja' mean for each group.
        Order by 'duracion_baja' and reset index.
        Take the log of the 'duracion_baja', column name = 'duracion_baja_log'.
        Return a df with 'icd9_cat', duracion_baja, 'duracion_baja_log'
        """
        #Group by 'icd9_cat' and take the 'duracion_baja' mean for each group
        df_icd9 = df[['icd9_cat', 'duracion_baja']].groupby(['icd9_cat']).mean()

        #Order by 'duracion_baja' and reset index
        df_icd9 = df_icd9.sort_values(by='duracion_baja', ascending=False)
        df_icd9.reset_index(inplace=True)

        #Take the log of the 'duracion_baja'
        def rounded_log(x):
            return round(np.log(x), 4)
        df_icd9['duracion_baja_log'] = df_icd9[['duracion_baja']].apply(rounded_log)

        #Return a df with 'icd9_cat', duracion_baja, 'duracion_baja_log'
        return df_icd9

    def make_categories(max_num, num_cat=10):
        """
        create 'num_cat' same size categories,
        """
        min_cat= max_num/num_cat
        return [(i, min_cat*i) for i in range(1, num_cat + 1)]

    def category_duration(row, categories):
        for cat in categories:
            if row.duracion_baja_log <= cat[1]:
                return cat[0]

    def encode_icd9_basic(df, num_cat=10):
        """
        return a dataframe with new columns:
            'icd9_cat': int part of icd9
            'icd9_cat_target': ascending categories using the mean of the 'duracion_baja'
        """
        #Create new colum with main categories of 'icd9' (only int part)
        df['icd9_cat'] = df.apply(make_icd9_cat, axis=1)

        df_icd9 = group_icd9_cat_by_target(df)
        max_num = df_icd9['duracion_baja_log'].max()

        categories = make_categories(max_num, num_cat)
        df_icd9['icd9_cat_target'] = df_icd9.apply(category_duration, args = (categories,), axis=1)
        df_icd9.drop(['duracion_baja', 'duracion_baja_log'], axis = 1, inplace = True)
        icd9_mapper = dict(zip(df_icd9.icd9_cat, df_icd9.icd9_cat_target))
        df = df.merge(df_icd9[['icd9_cat', 'icd9_cat_target']] , on = "icd9_cat" , how = "left")

        return df, icd9_mapper

    df = pd.concat([df_x, df_y], axis = 1)
    df, icd9_mapper = encode_icd9_basic(df)
    df.drop(['icd9', 'icd9_cat'], axis = 1, inplace = True)

    return df.drop('duracion_baja', axis = 1), df['duracion_baja'], icd9_mapper

def fill_icd9_mapper(icd9_mapper):
    for i in range(1001):
        if i in icd9_mapper.keys():
            continue

        # Search closest lower.
        number = i-1
        while True:
            if number < 1:
                # Lowest Value hard-coded.
                number = 5
                break
            if number in icd9_mapper.keys():
                break
            number -= 1

        icd9_mapper[i] = icd9_mapper[number]

    for i in [1010, 1020, 1030]:
        if i not in icd9_mapper.keys():
            # Hard Coded.
            icd9_mapper[i] = 6

    return icd9_mapper

def icd9_non_train(df_x, icd9_mapper):
    def make_icd9_cat(row):
        first_str = row.icd9[0]
        if first_str == 'E':
            return 1010
        if first_str == 'V':
            return 1020
        if first_str == 'M':
            return 1030
        else:
            code = float(row.icd9)
            return round(code)

    df_x['icd9_cat'] = df_x.apply(make_icd9_cat, axis=1)
    df_x['icd9_cat_target'] = df_x['icd9_cat'].map(icd9_mapper)
    df_x.drop(['icd9', 'icd9_cat'], axis = 1, inplace = True)

    return df_x


def rent_by_postal(df, codigos = os.path.join(dirname, '../raw_data/Codigos_postales_res.xlsx')):
    postal = pd.read_excel(codigos, usecols = ['Codigo', 'Renta_disp_media'])
    postal.rename(columns = {'Codigo': 'Code', 'Renta_disp_media': 'Value'}, inplace = True)

    postal = postal[postal['Code'] != 'Provincia'].reset_index(drop = True)
    # Add a 0 at the front of 4 digit codes.
    def fill_codes(x):
        if len(x) == 4:
            return '0' + x
        else:
            return x
    postal['Code'] = postal['Code'].astype('str').apply(fill_codes)

    # Create a dataframe of province values.
    rest_postal = {}

    for i in postal[postal['Code'] == 'Resto'].index:
        code = postal.iloc[i-1, 0][:2]
        value = postal.iloc[i, 1]
        if code in rest_postal.keys():
            rest_postal[code].append(value)
        else:
            rest_postal[code] = [value]

    rest_postal = {k: round(sum(v) / len(v)) for k, v in rest_postal.items()}
    # Add nulls for all other province post codes.
    for i in range(1, 53):
        province = str(i).rjust(2, '0')
        if province not in rest_postal.keys():
            rest_postal[province] = np.nan

    full_postal = postal[postal['Code'] != 'Resto'].reset_index(drop = True)
    df.rename(columns = {'codipostal': 'Code'}, inplace = True)
    df = df.merge(full_postal, on = 'Code', how = 'left')

    df['Code'] = df['Code'].str[:2]
    df['Value'] = df['Value'].fillna(df['Code'].map(rest_postal))
    extra = '''31	Navarra	22.786
    13	CiudadReal	21.575
    44	Teruel	18.420
    01	Álava	21.516
    48	Vizcaya	21.853
    45	Toledo	18.413
    05	Ávila	17.483
    19	Guadalajara	21.082
    49	Zamora	17.324
    10	Caceres	16.845
    27	Lugo	17.510
    20	Guipuzkoa	23.512
    22	Huesca	19.974
    40	Segovia	19.257
    34	Palencia	19.551
    52	Melilla	26.632
    51	Ceuta	26.835
    16	Cuenca	17.255
    42	Soria	19.221'''

    extra = extra.split()
    index = 0
    for i in extra:
        if index == 0:
            new_code = i
            index = 1
            continue
        elif index == 1:
            index = 2
            continue
        else:
            number = i.split('.')
            value = int(number[0]) * 1000 + int(number[1])
            rest_postal[new_code] = value
            index = 0
            continue

    df['Value'] = df['Value'].fillna(df['Code'].map(rest_postal))
    def postal_group(code):
        if code == '08':
            return '1'
        elif code == '28':
            return '2'
        elif code == '17':
            return '3'
        else:
            return '4'

    df['Code'] = df['Code'].apply(postal_group)

    return df

def cyclic_calendar(df):
    df['time'] = (2 * pi * (df['epiweek'] - 1) + ((df['diasemana'] - 1) / 7)) / 52
    df['sin_time'] = np.sin(df['time'])
    df['cos_time'] = np.cos(df['time'])
    df.drop(['time', 'diasemana', 'epiweek'], axis = 1, inplace = True)

    return df

def split_data_ml(data):
    X_train, X_test, y_train, y_test = train_test_split(
                                data.drop('duracion_baja', axis = 1), data['duracion_baja'],
                                test_size=0.25, random_state = 8)

    # Remove huge outliers from train set.
    train = pd.concat([X_train, y_train], axis = 1)
    train = train.query('duracion_baja < 400')
    train = train.query('ContadordiasBajasDNI < 1000')
    train = train.query('numtreb < 3200')

    X_train = train.drop('duracion_baja', axis = 1)
    y_train = train['duracion_baja']

    return X_train, X_test, y_train, y_test

def pipeline():
    robust = RobustScaler()
    standard = StandardScaler()
    categorical = OneHotEncoder(handle_unknown = 'ignore',
                                drop = 'if_binary', sparse = False)

    robust_cols = ['ContadordiasBajasDNI', 'numtreb', 'tiempo_en_empresa']
    std_cols = ['edad']
    cat_cols = ['grupcoti', 'recaida', 'pluriempleo',
                'cnae_cat', 'icd9_cat_target']

    col_transformer = make_column_transformer((robust, robust_cols),
                                          (standard, std_cols),
                                          (categorical, cat_cols),
                                          remainder = 'passthrough')

    return col_transformer

def preprocess_new(row, cnae_categories, icd9_mapper):
    '''
    Assume row is a dataframe with 1 row.
    Columns :
    [cnae, icd9, recaida, numtreb, codipostal,
    ContadordiasBajasDNI, contracte, grupcoti, pluriempleo,
    diasemana, tiempo_en_empresa, edad, epiweek]
'''
    row = row.merge(cnae_categories, how = 'left', on = 'cnae')
    row.drop('cnae', axis = 1, inplace = True)
    row = icd9_non_train(row, icd9_mapper)
    row = rent_by_postal(row)
    # Sin and Cos do not accept a single object value.
    row[['epiweek', 'diasemana']] = row[['epiweek', 'diasemana']].astype('float64')
    row = cyclic_calendar(row)

    return row


# Process Data
def preprocess_raw_data():
    data = load_data()
    data = null_data(data)
    data = bad_counter(data)
    data, cnae_categories = categorize_cnae(data)
    data = rent_by_postal(data)
    data = cyclic_calendar(data)

    X_train, X_test, y_train, y_test = split_data_ml(data)
    X_train, y_train = wrong_time_in_company(X_train, y_train)
    X_train, y_train, icd9_mapper = categorize_icd9_train(X_train, y_train)
    icd9_mapper = fill_icd9_mapper(icd9_mapper)
    X_test = icd9_non_train(X_test, icd9_mapper)

    return X_train, X_test, y_train, y_test, cnae_categories, icd9_mapper


def fit_pipelines(X_train, y_train):
    
    preproc = pipeline()
    preproc_2 = make_pipeline(preproc)
    preproc_2.fit(X_train, y_train)
            
    return preproc_2

def y_encode_2(y):
        if y<15:
            return 0
        else:
            return 1


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, cnae_categories, icd9_mapper = preprocess_raw_data()
        
    # Save cnae_categories
    cnae_categories_path = os.path.join(LOCAL_REGISTRY_PATH, "cnae_categories.pkl")
    with open(cnae_categories_path, "wb") as file:
        pickle.dump(cnae_categories, file)
    
    print("\n✅ Cnae categories saved locally")
    
    # Save icd9_mapper
    icd9_mapper_path = os.path.join(LOCAL_REGISTRY_PATH, "icd9_mapper.pkl")
    with open(icd9_mapper_path, "wb") as file:
        pickle.dump(icd9_mapper, file)
    
    print("\n✅ Icd9 mapper saved locally")
    
    # Fit preprocessor pipeline
    preproc = fit_pipelines(X_train, y_train)
    preproc_path = os.path.join(LOCAL_REGISTRY_PATH, "preproc_pipeline.pkl")
    with open(preproc_path, "wb") as file:
        pickle.dump(preproc, file)
        
    print("\n✅ Preprocessor pipeline saved locally")
    
    X_train_final = preproc.transform(X_train)
    X_test_final = preproc.transform(X_test)
            
    X_train_path = os.path.join(LOCAL_REGISTRY_PATH, "X_train_final.pkl")
    with open(X_train_path, "wb") as file:
        pickle.dump(X_train_final, file)
    X_test_path = os.path.join(LOCAL_REGISTRY_PATH, "X_test_final.pkl")
    with open(X_test_path, "wb") as file:
        pickle.dump(X_test_final, file)
    y_train_path = os.path.join(LOCAL_REGISTRY_PATH, "y_train_final.pkl")
    with open(y_train_path, "wb") as file:
        pickle.dump(y_train, file)
    y_test_path = os.path.join(LOCAL_REGISTRY_PATH, "y_test_final.pkl")
    with open(y_test_path, "wb") as file:
        pickle.dump(y_test, file)
        
    print("\n✅ Train and test sets saved locally")
