
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_tranformer_object(self):
        try:
            colReplaceWithMean = ['Age','DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar',
                                  'NumberOfTrips', 'NumberOfChildrenVisited', 'MonthlyIncome']

            colReplaceWithMode = ['TypeofContact']
            ColsOneHot = ['Occupation', 'Gender', 'MaritalStatus']
            ColsOrdinal = ['ProductPitched', 'Designation']

            ordinal_categories = [['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'],
                                  ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']]

            S1RepMean = Pipeline([
                ('S1', SimpleImputer(add_indicator=False))
            ])

            S2RepMedianNoneOHE = Pipeline([
                ('S2RepMedian', SimpleImputer(strategy='most_frequent', add_indicator=False)),
                ('S2OHE', OneHotEncoder(sparse=False, handle_unknown='ignore'))
            ])

            S3OHE = Pipeline([
                ('S3OHE', OneHotEncoder(sparse=False, handle_unknown='ignore'))
            ])

            S4OrdEncd = Pipeline([
                ('S4OrdEncd', OrdinalEncoder(categories=ordinal_categories))
            ])

            preprocessor = ColumnTransformer([
                ('S1RepMean', S1RepMean, colReplaceWithMean),
                ('S2RepMedian', S2RepMedianNoneOHE, colReplaceWithMode),
                ('S3OHE', S3OHE, ColsOneHot),
                ('S4OrdEncd', S4OrdEncd, ColsOrdinal)
            ], remainder='passthrough')
            return preprocessor

        except Exception as e:
            raise Exception(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            print("Read data")
            print("Read data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            preprocessing_obj = self.get_data_tranformer_object()

            tragetCol = "ProdTaken"
            print("Read data2")

            X_train = train_df.drop(columns=[tragetCol],axis=1)
            print("X_train")

            print(X_train)
            y_train = train_df[tragetCol]
            X_train = preprocessing_obj.fit_transform(X_train)
            train_arr = np.c_[X_train, np.array(y_train)]

            X_test = test_df.drop(columns=[tragetCol],axis=1)
            y_test = test_df[tragetCol]
            X_test = preprocessing_obj.transform(X_test)
            test_arr = np.c_[X_test, np.array(y_test)]


            print(train_arr)

            print(pd.DataFrame(train_arr))
            #print(pd.DataFrame(test_arr))

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise Exception(e,sys)
