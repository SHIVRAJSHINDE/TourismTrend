import os
import sys
from dataclasses import dataclass
import numpy as np


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_train_arr,y_train_arr,X_test_arr,y_test_arr,):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                X_train_arr,
                y_train_arr,
                X_test_arr,
                y_test_arr,
            )

            params = {'Svc': {
                'C': [0.1, 1, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'degree': [1, 2, 3, 4, 5, 6]
            }

            }

            models = {
                "Svc": SVC()
            }



            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            print('best_model_score : ',best_model_score)
            ## To get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print('best_model_name : ',best_model_name)

            best_model = models[best_model_name]
            print('best_model : ',best_model)

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )




        except Exception as e:
            raise CustomException(e, sys)


