import pickle
import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')

def index():
    return render_template('index.html')
    
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['Get','POST'])

def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")))
        
        logging.info("Data received from form")
        
        pred_df = data.get_data_as_dataframe()
        prediction_pipeline = PredictPipeline()
        results = prediction_pipeline.predict(pred_df)
        return render_template('results.html', results= results[0])

if __name__ == "__main__":
    logging.info("Starting the Flask server for prediction")
    app.run(host='0.0.0.0', debug=True)