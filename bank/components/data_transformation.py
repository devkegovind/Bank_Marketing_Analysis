import os
import sys
import pandas as pd
import numpy as np
import pickle

from bank.logger import logging
from bank.exception import CustomException
from bank.components.data_ingestion import DataIngestion
from bank.utils import save_object
from bank.utils import evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    
# Create a Class Data Transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    # Initiate class Data Transformation
    
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            
            # Define Categorical Columns of Dataset
            categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                                    'month', 'poutcome']
            
            # Define Numerical Columns of Dataset
            numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

            # Define Custom Rating
            job_categories = ['admin.', 'technician', 'services', 'management', 'retired',
            'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
            'unknown', 'self-employed', 'student']
            marital_categories = ['married', 'single', 'divorced']
            education_categories = ['secondary', 'tertiary', 'primary', 'unknown']
            default_categories = ['no', 'yes']
            housing_categories  = ['yes', 'no']
            loan_categories = ['no', 'yes']
            contact_categories  = ['unknown', 'cellular', 'telephone']
            month_categories = ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
            'mar', 'apr', 'sep']
            poutcome_categories = ['unknown', 'other', 'failure', 'success']
            
            
            logging.info("Pipeline Initiated")
            
            # Numerical Pipeline
            
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Categorical Pipeline
            
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy= 'most_frequent')),
                    ('onehotencoding', OneHotEncoder(categories = [job_categories,
                    marital_categories,
                    education_categories,
                    default_categories,
                    housing_categories,
                    loan_categories,
                    contact_categories,
                    month_categories,
                    poutcome_categories,
                    
                    ])),
                    
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            # Preprocessor Combine Two Pipeline
            
            preprocessor = ColumnTransformer([
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                    ]
            )
            
            return preprocessor
            logging.info("Pipeline Completed")
            
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    # Read Train & Test Data

    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            # Reading Train & Test Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read Train & Test Data Completed")
            
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")
            
            logging.info("Obtaining Processing Object")
            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'deposit'
            drop_columns = [target_column_name, 'duration']
            
            input_feature_train_df = train_df.drop(columns = target_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns =  target_column_name, axis = 1)
            target_feature_test_df = test_df[target_column_name]
            
            # Transforming Using Preprocessor Obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying Preprocessing Object on Training & Testing Datasets")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            logging.info("Preprocessor pickle file saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception Occured in the initiate_data_transformation")
            raise CustomException(e, sys)



        

