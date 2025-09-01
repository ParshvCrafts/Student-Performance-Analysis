import os
import sys
import dill
from src.exception import CustomException
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from typing import Dict, Tuple

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model, param, cv=5)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = r2_square
            logging.info(f"{list(models.keys())[i]} R2 Score: {r2_square}")
        
        return report

    except Exception as e:
        raise CustomException(e, sys)