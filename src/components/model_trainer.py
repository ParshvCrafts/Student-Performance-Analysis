import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.data_transformation = DataTransformation()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },
                "CatBoosting Regressor": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'iterations': [100, 200, 300]
                },
                "Linear Regression": {'fit_intercept': [True, False]},
                
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
            
            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)   
            
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with R2 score: {model_report[best_model_name]}")
            
            if model_report[best_model_name] < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model)
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            rmse = np.sqrt(mse)
            logging.info(f"Model Report: R2 Score: {r2_square}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
            return r2_square
        
        except Exception as e:
            logging.info("Exception occurred at Model Training stage")
            raise CustomException(e, sys)

