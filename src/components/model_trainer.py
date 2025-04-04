import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import customexception
from dataclasses import dataclass
from src.utils import save_object
from src.utils import evaluate_model

from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models
            models = {
                'GaussianNB': GaussianNB(),
                'XGBClassifier': XGBClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier()
            }

            model_report = {}
            for model_name, model in models.items():
                logging.info(f'Training model: {model_name}')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
                model_report[model_name] = precision

                logging.info(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}')

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            # Get the best model based on precision score
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name}, Precision Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found: {best_model_name}, Precision Score: {best_model_score}')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully.")

        except Exception as e:
            logging.error('Exception occurred during model training')
            raise customexception(e, sys)


