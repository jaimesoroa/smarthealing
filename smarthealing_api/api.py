from datetime import datetime
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from smarthealing_mod.preprocessor import preprocess_train, preprocess_new
from smarthealing_mod.model.model import fit_model


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


app = FastAPI()
app.state.model = model

# "key": "O",
DTYPES_RAW_OPTIMIZED = {
    "ContadorBajasCCC": "int64",
    "ContadorBajasDNI": "int64",
    "sexo": "int64",
    "cnae": "str",
    "icd9": "str",
    "recaida": "int64",
    "numtreb": "int64",
    "codipostal": "str",
    "ContadordiasBajasDNI": "int64",
    "contracte": "str",
    "grupcoti": "str",
    "pluriempleo": "int64",
    "diasemana": "int64",
    "tiempo_en_empresa": "float64",
    "edad": "float64",
    "mes_baja": "int64",
    "epiweek": "int64"
}

COLUMN_NAMES_RAW = DTYPES_RAW_OPTIMIZED.keys()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define a prediction `/` endpoint
@app.get('/predict')
def predict(ContadorBajasCCC: int,
            ContadorBajasDNI: int,
            sexo: int,
            cnae: str,
            icd9: str,
            recaida: int,
            numtreb: int,
            codipostal: str,
            ContadordiasBajasDNI: int,
            contracte: str,
            grupcoti: str,
            pluriempleo: int,
            diasemana: int,
            tiempo_en_empresa: float,
            edad: float,
            mes_baja: int,
            epiweek: int):
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """
    # key = ContadorBajasCCC
    X_new = pd.DataFrame([ContadorBajasCCC, ContadorBajasDNI, sexo, cnae, icd9,
       recaida, numtreb, codipostal, ContadordiasBajasDNI,
       contracte, grupcoti, pluriempleo, diasemana,
       tiempo_en_empresa, edad, mes_baja, epiweek]).T
    X_new.columns = COLUMN_NAMES_RAW
    X_new = X_new.astype(DTYPES_RAW_OPTIMIZED)
    # X_new = preprocess_features(X_new)
    # y_pred = float(app.state.model.predict(X_new))
    X_new_prepr = preprocess_new(X_new, ohe, rb_scaler, st_scaler)    
    y_pred = float(np.exp(app.state.model.predict(X_new_prepr)))
        
    # return y_pred
    
    
    return {'leave_duration': y_pred}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
