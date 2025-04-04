import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.logger import logging
from src.exception import customexception
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    try:
        logging.info("Starting the training pipeline...")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        model_trainer.initate_model_training(train_arr, test_arr)
        logging.info("Model Training completed successfully.")

    except Exception as e:
        logging.error("An error occurred in the training pipeline.")
        raise customexception(e, sys)