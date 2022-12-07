import pandas as pd
import numpy as np
from smarthealing_mod.preprocessor import preprocess_new, preprocess_raw_data
import xgboost as xgb
import time
import os
import pickle
from imblearn.over_sampling import SMOTE

# Extract the path to save models
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
dirname = os.path.dirname(__file__)

def fit_model_reg(X_train, y_train):
    '''Instantiate the regression model and fit it with train and test sets'''
    
    # Instantiate model
    model = xgb.XGBRegressor(booster= 'dart', objective ='reg:absoluteerror', n_estimators = 750, learning_rate= 0.01,
                             min_child_weight= 4, max_depth= 8, colsample_bytree= 0.7, n_jobs=-1, eval_metric = 'mae')
    # Logarithm of target
    y_log_train = np.log(y_train)
    # Train model
    model.fit(X_train, y_log_train)
    
    return model

def fit_model_class(X_train, y_train):
    """
    Instantiate the classification model and fit it with train and test sets
    """
    # Instantiate model
    model = xgb.XGBClassifier(objective= 'binary:logistic', max_depth= 3, eval_metric= 'aucpr', n_jobs=-1)
    # Encode the target for values higher or lower than 15 days.
    def y_encode_2(y):
        if y<15:
            return 0
        else:
            return 1
    y_train_encoded = y_train.map(y_encode_2)
    # Add linear combinations of the smaller category rows
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train_encoded)
    # Train model
    model.fit(X_train_resampled, y_train_resampled)
    
    return model

def save_model_reg(X_train, y_train):
    """
    Save trained models
    """
    
    # Get fitted model
    model_reg = fit_model_reg(X_train, y_train)
   
    # Save regression model    
    reg_model_path = os.path.join(LOCAL_REGISTRY_PATH, "xgb_regression_model.pkl")
    with open(reg_model_path, "wb") as file:
        pickle.dump(model_reg, file)
    
    print("\n✅ Regression model saved locally")

    return None

def save_model_class(X_train, y_train):
    """
    Save trained models
    """
    
    # Get fitted model
    model_class = fit_model_class(X_train, y_train)

    # Save classification model    
    class_model_path = os.path.join(LOCAL_REGISTRY_PATH, "xgb_classifier_model.pkl")
    with open(class_model_path, "wb") as file:
        pickle.dump(model_class, file)
    
    print("\n✅ Classification model saved locally")

    return None

if __name__ == '__main__':
        
    # X_train_filename = os.path.join(dirname, '../../raw_data/X_train')
    # X_train = pd.read_csv(X_train_filename, delimiter = ',', low_memory = False)
    # y_train_filename = os.path.join(dirname, '../../raw_data/y_train')
    # y_train = pd.read_csv(y_train_filename, delimiter = ',', low_memory = False)
    
    X_train_path = os.path.join(LOCAL_REGISTRY_PATH, "X_train_final.pkl")
    with open(X_train_path, "rb") as file:
        X_train = pickle.load(file)
    X_test_path = os.path.join(LOCAL_REGISTRY_PATH, "X_test_final.pkl")
    with open(X_test_path, "rb") as file:
        X_test = pickle.load(file)
    y_train_path = os.path.join(LOCAL_REGISTRY_PATH, "y_train_final.pkl")
    with open(y_train_path, "rb") as file:
        y_train = pickle.load(file)
    y_test_path = os.path.join(LOCAL_REGISTRY_PATH, "y_test_final.pkl")
    with open(y_test_path, "rb") as file:
        y_test = pickle.load(file)
        
    save_model_class(X_train, y_train)
    save_model_reg(X_train, y_train)
