import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )

            models = {
                "Random_Forest_Classifier": RandomForestClassifier(),
                #"Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression()
            }

            params= {
                        "Random_Forest_Classifier" :{
                            'n_estimators': [100],
                            'max_depth': [10],
                            'min_samples_split': [2],
                            'min_samples_leaf': [1],
                            'max_features': ['sqrt'],
                            'bootstrap': [True],
                            'random_state': [42],
                            'n_jobs': [-1],
                            'oob_score': [True]

                        },

                    "Logistic Regression" :{
                        'penalty': ['l2'],
                        'solver' : ['lbfgs'],
                        'max_iter' : [1000],  # Increase the value
                        'random_state' : [42]

            },
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            print(model_report)

            best_model_score = max(sorted(model_report.values()))
            print(best_model_score)

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            print(best_model_name)
            best_model= models[best_model_name]
            print(best_model)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

        except Exception as e:
            raise CustomException(e,sys)




