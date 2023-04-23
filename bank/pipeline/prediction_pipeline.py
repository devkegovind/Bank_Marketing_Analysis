import sys
import os
import pandas as pd

from bank.exception import CustomException
from bank.logger import logging
from bank.utils import load_object

class PredictPipeline:
    
    def __init__(self):
        pass
    
    def predict(self, features):
        
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            
            return pred
        
        except Exception as e:
            logging.info("Exception Pccured in prediction")
            raise CustomException(e, sys)
        
    
class CustomData:
    
    def __init__(self, 
                 job:str, 
                 marital:str,
                 education:str,
                 default:str, 
                 housing:str, 
                 loan:str, 
                 contact:str,
                 month:str, 
                 poutcome:str,
                 age:int,
                 balance:int,
                 day:int,
                 campaign:int,
                 pdays:int,
                 previous:int):
        self.job= job
        self.marital=marital
        self.education=education
        self.default=default
        self.housing=housing
        self.loan=loan
        self.contact=contact
        self.month=month
        self.poutcome=poutcome
        self.age=age
        self.balance=balance
        self.day=day
        self.campaign=campaign
        self.pdays=pdays
        self.previous=previous
    
    
    def get_data_as_dataframe(self):
        
        try:
            custom_data_input_dict = {
                'job' : [self.job], 
                'marital':[self.marital],
                'education':[self.education],
                'default':[self.default],
                'housing':[self.housing],
                'loan':[self.loan],
                'contact' :[self.contact],
                'month':[self.month],
                'poutcome':[self.poutcome],
                'age':[self.age],
                'balance':[self.balance],
                'day':[self.day],
                'campaign':[self.campaign],
                'pdays':[self.pdays],
                'previous':[self.previous]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df
        
        except Exception as e:
            logging.info("Exception Occured in prediction pipeline")
            raise CustomData(e,sys)
            
            
            
