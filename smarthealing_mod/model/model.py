import pandas as pd
import numpy as np
from smarthealing_mod.preprocessor import preprocess_new
import xgboost as xgb
import time
import os
import pickle


LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))


def fit_model_reg(X_train, y_train):
    model = xgb.XGBRegressor(booster= 'dart', objective ='reg:squarederror', n_estimators = 750, learning_rate= 0.01,
                             min_child_weight= 4, max_depth= 8, colsample_bytree= 0.7, n_jobs=-1)
    y_log_train = np.log(y_train)
    model.fit(X_train, y_log_train)
    
    return model

def fit_model_class(X_train, y_train):
    model = xgb.XGBClassifier(objective= 'reg:logistic', max_depth= 3, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model

def save_models(X_train, y_train)
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    model_reg = fit_model_reg(X_train, y_train)
    model_class = fit_model_class(X_train, y_train)

    # save model
    
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
    reg_filename = 'xgb_regression_model.pkl'
    filehandler = open(reg_filename, 'wb')
    pickle.dump(model_reg,filehandler)
    filehandler.close()
    
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "trained_models", timestamp + ".pickle")
    with open(metrics_path, "wb") as file:
        pickle.dump(metrics, file)
    
    print("\n✅ data saved locally")

    return None

if __name__ == '__main__':
    save_models()