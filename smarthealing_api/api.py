from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from smarthealing_api.model_jaime import basic_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split



app = FastAPI()
app.state.model = basic_model()

# "key": "O",
DTYPES_RAW_OPTIMIZED = {
    "ContadorBajasCCC": "int64",
    "ContadorBajasDNI": "int64",
    "sexo": "int64",
    "cnae": "int64",
    "icd9": "float64", #string
    "recaida": "int64",
    "numtreb": "int64",
    "codipostal": "int64",
    "ContadordiasBajasDNI": "int64",
    "contracte": "int64",
    "grupcoti": "int64",
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
            cnae: int,
            icd9: float,
            recaida: int,
            numtreb: int,
            codipostal: int,
            ContadordiasBajasDNI: int,
            contracte: int,
            grupcoti: int,
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
    key = ContadorBajasCCC
    X_new = pd.DataFrame([ContadorBajasCCC, ContadorBajasDNI, sexo, cnae, icd9,
       recaida, numtreb, codipostal, ContadordiasBajasDNI,
       contracte, grupcoti, pluriempleo, diasemana,
       tiempo_en_empresa, edad, mes_baja, epiweek]).T
    X_new.columns = COLUMN_NAMES_RAW
    X_new = X_new.astype(DTYPES_RAW_OPTIMIZED)
    # X_new = preprocess_features(X_new)
    y_pred = float(app.state.model.predict(X_new))
    
    return {'leave_duration': y_pred}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
