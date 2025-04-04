from sklearn.preprocessing import OrdinalEncoder  # Ordinal Encoding
from sklearn.preprocessing import StandardScaler  # Handling Feature Scaling
from sklearn.impute import SimpleImputer          # Handling Missing Values
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.exception import customexception
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        This function creates and returns the preprocessing pipeline for numerical and categorical features.
        """
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
            numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),  # Handle unknown categories
                    ('scaler', StandardScaler())
                ]
            )

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.error('Error in Data Transformation')
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        This function applies the preprocessing pipeline to the training and testing datasets.
        """
        try:
            logging.info('Reading train and test data')
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            print(train_df['workclass'].unique())

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Income'
            drop_columns = [target_column_name]

            # Split features into independent and dependent features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the transformation
            logging.info('Applying preprocessing object on training and testing datasets')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # After fitting the preprocessor
            logging.info(f"Categories seen by OrdinalEncoder: {preprocessing_obj.named_transformers_['cat_pipeline'].named_steps['encoder'].categories_}")

            # Encode the target variable
            logging.info('Encoding target variable')
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            # Combine transformed features and target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Exception occurred in the initiate_data_transformation")
            raise customexception(e, sys)

