"""
Model Retraining Script
Run this script locally to regenerate pickle files when needed.
After running, commit the new artifacts to git.
"""
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

# Show sklearn version for reference
import sklearn
print(f"Training with scikit-learn version: {sklearn.__version__}")


def retrain_pipeline():
    """
    Complete retraining pipeline that regenerates all artifacts
    """
    try:
        logging.info("=" * 80)
        logging.info("STARTING MODEL RETRAINING PIPELINE")
        logging.info("=" * 80)

        # Step 1: Data Ingestion
        logging.info("Step 1: Initiating Data Ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train: {train_data_path}, Test: {test_data_path}")

        # Step 2: Data Transformation
        logging.info("Step 2: Initiating Data Transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )
        logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        logging.info("Step 3: Initiating Model Training")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training completed. Best model R2 Score: {r2_score}")

        logging.info("=" * 80)
        logging.info("RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)

        return {
            "status": "success",
            "r2_score": r2_score,
            "train_data": train_data_path,
            "test_data": test_data_path,
            "preprocessor": preprocessor_path
        }

    except Exception as e:
        logging.error("Exception occurred during retraining pipeline")
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        result = retrain_pipeline()
        print("\n" + "=" * 80)
        print("RETRAINING SUCCESSFUL!")
        print("=" * 80)
        print(f"R2 Score: {result['r2_score']}")
        print(f"Preprocessor: {result['preprocessor']}")
        print(f"Model artifacts saved in: artifacts/")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("RETRAINING FAILED!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        sys.exit(1)
