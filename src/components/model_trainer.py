import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join("artifacs","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split training and test inpu data")

            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
                "RandomForest": RandomForestRegressor(),
                "Decission Tree":DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-neighbors Classifier": KNeighborsRegressor(),
                "XGB Classsifier":XGBRFRegressor(),
                "Catboost classifier":CatBoostRegressor(verbose=False),
                "Adaboost Calssifier":AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X=x_train,y=y_train,X_test=x_test,y_test=y_test,models=models)
        except:
            pass