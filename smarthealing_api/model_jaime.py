import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from smarthealing_mod.preprocessor import preprocess_train, preprocess_new
from smarthealing_mod.model.model import fit_model

print(f"Let's see if this is working")

dirname = os.path.dirname(__file__)

def load_data():
    
    # 0. Load Data and Drop Duplicates
    csv_filename = os.path.join(dirname, '../raw_data/datos.csv')
    data = pd.read_csv(csv_filename, delimiter = ';', low_memory = False)
    data.drop_duplicates(inplace = True)
    # outliers = data.query('duracion_baja >= 250')  # Save outliers for analysis.
    data = data.query('duracion_baja < 250')
    # data.drop_duplicates(inplace = True)
    
    return data


data = load_data()
X_train, X_test, y_train, y_test, ohe, rb_scaler, st_scaler = preprocess_train(data)
model = fit_model(X_train, y_train)

# X_new = pd.DataFrame([ContadorBajasCCC, ContadorBajasDNI, sexo, cnae, icd9,
#        recaida, numtreb, codipostal, ContadordiasBajasDNI,
#        contracte, grupcoti, pluriempleo, diasemana,
#        tiempo_en_empresa, edad, mes_baja, epiweek]).T
#     X_new.columns = COLUMN_NAMES_RAW
#     X_new = X_new.astype(DTYPES_RAW_OPTIMIZED)
# X_new = preprocess_features(X_new)
# y_pred = float(app.state.model.predict(X_new))
    
X_new = X_test[0]
X_new_prepr = preprocess_new(X_new, ohe, rb_scaler, st_scaler)    
y_pred = float(np.exp(model.predict(X_new_prepr)))

print(f'The prediction from {X_new} is {y_pred}')