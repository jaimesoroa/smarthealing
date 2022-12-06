import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from smarthealing_mod.preprocessor import preprocess_new
import xgboost as xgb


def fit_model(X_train, y_train):
    model = xgb.XGBRegressor(booster= 'dart', objective ='reg:squarederror', n_estimators = 750, learning_rate= 0.01,
                             min_child_weight= 4, max_depth= 8, colsample_bytree= 0.7)
    y_log_train = np.log(y_train)
    model.fit(X_train, y_log_train)
    
    return model


