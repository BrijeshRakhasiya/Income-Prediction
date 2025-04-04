import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import customexception
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return a report with precision scores.
    """
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            # Train the model
            model.fit(X_train, y_train)

            # Predict on test data
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
            report[model_name] = {
                'accuracy': accuracy,
                'precision': precision
            }

            logging.info(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}")

        return report

    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise customexception(e, sys)

def load_object(file_path):
    """
    Load a Python object from a file using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise customexception(e, sys)

