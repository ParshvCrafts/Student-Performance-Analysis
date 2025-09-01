import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from src.utils import load_object

@dataclass

class PredictPipelineConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class PredictPipeline:
    def __init__(self):
        self.model_path = PredictPipelineConfig.model_path
        self.preprocessor_path = PredictPipelineConfig.preprocessor_path
        
    def predict(self, features):
        try:
            logging.info("Loading preprocessor and model")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            
            logging.info("Transforming features")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making predictions")
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            logging.info("Exception occurred during prediction")
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        
        except Exception as e:
            logging.info("Exception occurred while creating dataframe")
            raise CustomException(e, sys)
        
        