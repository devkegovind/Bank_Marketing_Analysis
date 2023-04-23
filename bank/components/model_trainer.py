import numpy as np
import pandas as pd
import os
import sys

from bank.exception import CustomException
from bank.logger import logging
from bank.components.data_ingestion import DataIngestion

from bank.utils import save_object
from bank.utils import evaluate_model

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score


@dataclass
class ModelTrainerConfig:
    trained_file_path = os.path.join('artifacts', 'model.pkl') 

class ModelTrainer:
    
    def __init__(self):
        self.model_trained_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent & Independent Variables From Train & Test Dataset")
            X_train, y_train, X_test, y_test =(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
                                
            )
            
            models = {
                'LogisticRegression' : LogisticRegression(),
                'DecisionTreeClassifier' : DecisionTreeClassifier(),
                'RandomForestClassifier' : RandomForestClassifier(),
                'NaiveBayes': GaussianNB()
                
            }
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print(f"\n{'*'*85}")
            logging.info(f"Model Report : {model_report}")
            
            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            print(f"Best Model Found, Model Name : {best_model_name}, F1 Score : {best_model_score}")
            
            print(f"\n{'*'*85}")
            
            logging.info(f"Best Model Found, Model Name : {best_model_name},F1 score : {best_model_score}")
            
            save_object(file_path = self.model_trained_config.trained_file_path, obj = best_model)
            
        except Exception as e:
            logging.info("Exception is occured in the Model Trainer")
            raise CustomException(e, sys)
            