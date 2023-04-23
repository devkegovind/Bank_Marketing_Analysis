import os
import sys
import pickle
import numpy as np
import pandas as pd


from bank.exception import CustomException
from bank.logger import logging


from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

# Data Transformation
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path, "wb") as file_obj:
             pickle.dump(obj, file_obj)
             
    except Exception as e:
        raise CustomException(e, sys)
    
# Model Trainer

def  evaluate_model(X_train, y_train, X_test, y_test, models):
    
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Train Model
            model.fit(X_train, y_train)
            
            # Predict Testing Data
            y_test_pred = model.predict(X_test)
            
            # Get F1 scores for Train & Test Data
            
            #train_model_score = f1_score(y_train, y_train_pred)
            test_model_score = f1_score(y_test, y_test_pred)*100
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        logging.info("Exception is occured during Model Training")
        raise CustomException(e, sys)
            
            
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info("Exception Occured in load_object function utils")
        raise CustomException(e, sys)