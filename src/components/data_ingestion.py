import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    try:
        train_data_path: str=os.path.join('artifacts','train.csv')
        test_data_path: str=os.path.join('artifacts','test.csv')
        raw_data_path: str=os.path.join('artifacts','raw.csv')

    except Exception as e:
        raise CustomException(e, sys)



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_iqr(self,data, column_name, q1_range, q3_range):
        try:
            q1 = data[column_name].quantile(q1_range)
            q3 = data[column_name].quantile(q3_range)
            IQR = q3 - q1
            upper_fence = q3 + (1.5 * IQR)
            lower_fence = q1 - (1.5 * IQR)


            indexUL = data[data[column_name] > upper_fence].index
            indexLM = data[data[column_name] < lower_fence].index

            # print("indexUL: "+str(indexUL))
            # print("indexLM: "+str(indexUL))

            print(data.shape)

            data.drop(indexUL, inplace=True)
            data.drop(indexLM, inplace=True)
            print(data.shape)
            return data
        except Exception as e:
            raise CustomException(e,sys)


    def encodeTheData(self, data, colName1, colName2):
        try:
            data = pd.get_dummies(data, columns=[colName1[0], colName1[1], colName1[2], colName1[3]], drop_first=True)
            data[colName2[0]] = data[colName2[0]].replace({'Basic': 1, 'Standard': 2, 'Deluxe': 3, 'Super Deluxe': 4, 'King': 5})

            data[colName2[1]] = data[colName2[1]].replace({'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'AVP': 4, 'VP': 5})


            return data
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self):

        try:

            data=pd.read_csv('notebook/data/tourismData.csv')
            data = data[['CustomerID','Age','TypeofContact','CityTier','DurationOfPitch','Occupation','Gender','NumberOfPersonVisited',
                         'NumberOfFollowups','ProductPitched','PreferredPropertyStar','MaritalStatus','NumberOfTrips','Passport',
                         'PitchSatisfactionScore','OwnCar','NumberOfChildrenVisited','Designation','MonthlyIncome','ProdTaken']]

            data.drop(['CustomerID'], axis=1,inplace=True)

            data['Age'].fillna(data['Age'].median(), inplace=True)
            data['TypeofContact'].fillna(data['TypeofContact'].mode()[0], inplace=True)
            data['DurationOfPitch'].fillna(data['DurationOfPitch'].median(), inplace=True)
            data['NumberOfFollowups'].fillna(data['NumberOfFollowups'].median(), inplace=True)
            data['PreferredPropertyStar'].fillna(data['PreferredPropertyStar'].median(), inplace=True)
            data['NumberOfTrips'].fillna(data['NumberOfTrips'].median(), inplace=True)
            data['NumberOfChildrenVisited'].fillna(data['NumberOfChildrenVisited'].median(), inplace=True)
            data['MonthlyIncome'].fillna(data['MonthlyIncome'].median(), inplace=True)

            data = self.get_iqr(data, 'MonthlyIncome',0.20,0.90)
            data = self.get_iqr(data, 'DurationOfPitch',0.10,0.85)
            data = self.get_iqr(data, 'NumberOfTrips',0.20,0.75)
            data = data

            col1 = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus']
            col2 = ['ProductPitched', 'Designation']

            data = self.encodeTheData(data, col1, col2)

            lastCol = data['ProdTaken']
            data = data.drop(['ProdTaken'], axis=1)
            data = pd.concat([data, lastCol], axis=1)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)
            print("files step-1")


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,

            )

            print("files step-2")

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    dataTransformation = DataTransformation()
    X_train_arr,y_train_arr,X_test_arr,y_test_arr,_ = dataTransformation.initiate_data_transformation(train_path,test_path)

    #modelTrainer = ModelTrainer()
    #print(modelTrainer.initiate_model_trainer(X_train_arr,y_train_arr,X_test_arr,y_test_arr))

