import pandas as pd
import numpy as np

def get_data():
    df = pd.read_csv('../../raw_data/datos.zip', delimiter=';')
    return df

def clean_data(df, columns = ['cnae', 'icd9', 'codipostal', 'contracte', 'grupcoti']):
    df = df.drop(columns=columns)
    df.drop_duplicates(inplace=True)
    df.replace(to_replace='-', value=np.nan, inplace=True)
    df.dropna(inplace=True);

    return df

def get_X_y(df, target=['duracion_baja']):
    X = df.drop(columns=target)
    y = df.duracion_baja

    total = y.count()
    mean = y.mean()
    print(f'there is a total of {total} entries, the target mean is {mean}')

    return X, y

def get_X_y_log(df, target=['duracion_baja']):
    X = df.drop(columns=target)
    y_log = df.duracion_baja > 15

    total = y_log.count()
    true_vals = y_log.sum()
    print(f'there are {true_vals} true values, \
        {100*np.round(true_vals/total, 4)} % of the total')

    return X, y_log

def make_train_test_split(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=test_size,
                                        random_state=random_state)

    return X_train, X_test, y_train, y_test

def ready_to_model():
    df = get_data()
    df = clean_data(df)
    return get_X_y(df)

def ready_to_model_log():
    df = get_data()
    df = clean_data(df)
    return get_X_y_log(df)
