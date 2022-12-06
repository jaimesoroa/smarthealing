import pandas as pd
import numpy as np

def get_data(file_path = '../raw_data/datos.zip'):
    df = pd.read_csv(file_path, delimiter=';')
    return df

def clean_data(df, columns = ['cnae', 'icd9', 'codipostal', 'contracte', 'grupcoti']):
    df = df.drop(columns=columns)
    df.drop_duplicates(inplace=True)
    df.replace(to_replace='-', value=np.nan, inplace=True)
    df.dropna(inplace=True);

    return df

def get_X_y(df):
    X = df.drop(columns=['duracion_baja'])
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


#make category for icd9
special_cases = ['E','V','M']
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
    df_icd9['duracion_baja_log'] = df_icd9[['duracion_baja']].apply(lambda x: round(np.log(x),4))

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


def enconde_icd9_basic(df, num_cat=10):
    """
    return a dataframe with new columns:
        'icd9_cat': int part of icd9
        'icd9_cat_target': ascending categories using the mean of the 'duracion_baja'
    """
    #Create new colum with main categories of 'icd9' (only int part)
    df['icd9_cat'] = df.apply(lambda row: make_icd9_cat(row), axis=1)

    df_icd9 = group_icd9_cat_by_target(df)
    #breakpoint()
    max_num = df_icd9['duracion_baja_log'].max()
    #breakpoint()

    categories = make_categories(max_num, num_cat)
    df_icd9['icd9_cat_target'] = df_icd9.apply(lambda row: category_duration(row, categories), axis=1)
    df = df.merge( df_icd9[['icd9_cat', 'icd9_cat_target']] , on = "icd9_cat" , how = "left")
    breakpoint()

    return df

#scrape pdf for icd10 codes and time
def get_values(rows, data):
    for row in rows:
        row.strip()
        words = row.split()
        if len(words) > 0:
            if words[-1].isdigit():
                new_row = pd.DataFrame({'icd10': words[0], 'tiempo': words[-1]}, index=[0])
                data = pd.concat([data, new_row], axis=0, ignore_index=True)

    return data

def get_icd10_codes_and_time(file_path, first_page=37, last_page=116):
    import PyPDF2

    pdfFileObj = open(file_path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    data = pd.DataFrame(columns=['icd10', 'tiempo'])

    for page in range(first_page, last_page):
        pageObj = pdfReader.getPage(page)
        rows = pageObj.extractText().splitlines()
        data = get_values(rows, data)

    return data

#get icd9 icd10 code equivalence
def get_icd9_df():
    df = get_data()
    icd9_df = df.icd9.unique()
    return pd.DataFrame(icd9_df)

def save_icd9_codes_to_txt(icd9_df, filename='icd_9_codes.txt'):
    icd9_df.to_csv(filename, index=False)

def load_icd9_to_icd10_conversion(file_name='icd_9_codes.txt.out'):
    return pd.read_csv(file_name)

def clean_icd9_to_icd10(icd_9to10_df):
    #unpack the file with icd9 to icd10 conversion onto list of tupples
    list_9_to_10 = []
    for num in range(len(icd_9to10_df)):
        row = icd_9to10_df.iloc[num,0].split('\t')
        list_9_to_10.append((row[0],row[1]))

    #save only the values that have a matchand return a dataframe
    has_icd10 = {'icd9':[], 'icd10':[]}
    for row in list_9_to_10:
        if row[1] != 'NA':
            has_icd10['icd9'].append(row[0])
            has_icd10['icd10'].append(row[1])

    return pd.DataFrame(has_icd10)

def get_final_df(has_icd10, icd10_time_df, how='left'):
    return has_icd10.merge(icd10_time_df, on='icd10', how=how)
