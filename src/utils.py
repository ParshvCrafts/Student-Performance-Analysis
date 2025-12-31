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
    
def load_object(file_path):
    """
    Load a pickled object with version compatibility handling

    Args:
        file_path: Path to the pickle file

    Returns:
        Loaded object

    Raises:
        CustomException: If loading fails after all retry attempts
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except AttributeError as ae:
        error_msg = str(ae)
        # Check if this is a sklearn version mismatch issue
        sklearn_indicators = ['sklearn', 'SimpleImputer', 'StandardScaler',
                              'OneHotEncoder', '_fill_dtype', 'ColumnTransformer']
        is_sklearn_issue = any(indicator in error_msg for indicator in sklearn_indicators)

        if is_sklearn_issue:
            logging.error(f"AttributeError when loading {file_path}: {error_msg}")
            logging.error("This is due to scikit-learn version incompatibility.")
            logging.error("The pickled model was created with a different sklearn version.")
            logging.error("Please retrain the model with the current environment.")
            raise CustomException(ae, sys) from ae
        else:
            # Re-raise non-sklearn AttributeErrors with chaining
            logging.error(f"AttributeError when loading {file_path}: {error_msg}")
            raise CustomException(ae, sys) from ae
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {str(e)}")
        raise CustomException(e, sys) from e