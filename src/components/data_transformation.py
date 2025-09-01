import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            logging.info("Data Transformation initiated")
            df = pd.read_csv(r'src\data\student_performance.csv')
            X = df.drop(columns=['math_score'], axis=1)
            y = df['math_score']
            numeric_features = list(X.select_dtypes(exclude="object").columns)
            cat_features = list(X.select_dtypes(include="object").columns)
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Numerical and categorical pipeline completed")
            
            preprocessor = ColumnTransformer(
                [('num_pipeline', num_pipeline, numeric_features),
                 ('cat_pipeline', cat_pipeline, cat_features)
                ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                logging.info("Read train and test data completed")
                
                logging.info("Obtaining preprocessor object")
                preprocessing_obj = self.get_data_transformer_object()
                
                target_column_name = 'math_score'
                numerical_columns = list(train_df.select_dtypes(exclude="object").columns)
                categorical_columns = list(train_df.select_dtypes(include="object").columns)
                
                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]
                
                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]
                
                logging.info("Applying preprocessing object on training and testing datasets")
                
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                
                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
                
                logging.info("Saved preprocessing object")
                
                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj)
                
                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )
                
            except Exception as e:
                logging.info("Exception occurred in the initiate_data_transformation")
                raise CustomException(e, sys)
            
            
            