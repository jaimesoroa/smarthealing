from datetime import datetime
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from smarthealing_mod.preprocessor import preprocess_new
import pickle


dirname = os.path.dirname(__file__)

LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))


def load_models():
    '''
    persist trained model, params and metrics
    '''
        
    reg_model_path = os.path.join(dirname, '../trained_models/xgb_regression_model.pkl')
    with open(reg_model_path, "rb") as file:
        model_reg = pickle.load(file)
        
    class_model_path = os.path.join(dirname, '../trained_models/xgb_classifier_model.pkl')
    with open(class_model_path, "rb") as file:
        model_class = pickle.load(file)
    
    return model_reg, model_class

model_reg, model_class = load_models()



def load_files():
    # Load cnae_categories 
    cnae_categories_path = os.path.join(dirname, '../trained_models/cnae_categories.pkl')
    with open(cnae_categories_path, "rb") as file:
        cnae_categories = pickle.load(file)

    # Load icd9_mapper
    icd9_mapper_path = os.path.join(dirname, '../trained_models/icd9_mapper.pkl')
    with open(icd9_mapper_path, "rb") as file:
        icd9_mapper = pickle.load(file)
        
    # Load preprocessor
    preproc_path = os.path.join(dirname, '../trained_models/preproc_pipeline.pkl')
    with open(preproc_path, "rb") as file:
        preproc = pickle.load(file)

    """
    # Load regressor pipeline
    regressor_path = os.path.join(LOCAL_REGISTRY_PATH, "regressor_pipeline.pkl")
    with open(regressor_path, "rb") as file:
        regressor_pipeline = pickle.load(file)

    # Load classifier pipeline
    classifier_path = os.path.join(LOCAL_REGISTRY_PATH, "classifier_pipeline.pkl")
    with open(classifier_path, "rb") as file:
        classifier_pipeline = pickle.load(file)
    """

    return cnae_categories, icd9_mapper, preproc

cnae_categories, icd9_mapper, preproc = load_files()

app = FastAPI()
app.state.model_reg = model_reg
app.state.model_class = model_class
app.state.preproc = preproc

# app.state.reg_pipeline = regressor_pipeline
# app.state.class_pipeline = classifier_pipeline

# "key": "O",
DTYPES_RAW_OPTIMIZED = {
    "cnae": "int",
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
def predict(cnae: int,
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
    X_new = pd.DataFrame([cnae, icd9,
       recaida, numtreb, codipostal, ContadordiasBajasDNI,
       contracte, grupcoti, pluriempleo, diasemana,
       tiempo_en_empresa, edad, epiweek]).T
    X_new.columns = COLUMN_NAMES_RAW
    X_new = X_new.astype(DTYPES_RAW_OPTIMIZED)
    
    X_new_1 = preprocess_new(X_new, cnae_categories, icd9_mapper)
    X_new_preproc = app.state.preproc.transform(X_new_1)
    
    y_pred_reg = int(np.exp(app.state.model_reg.predict(X_new_preproc)))
    y_pred_class = int(app.state.model_class.predict(X_new_preproc))
    
    return {'classifier_leave_duration': y_pred_class, 'regression_leave_duration': y_pred_reg}

@app.get("/")
def root():
    return {'Greeting': 'Hello'}
