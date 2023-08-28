import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object
from sklearn.preprocessing import StandardScaler

class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds


        except Exception as e:
            raise  CustomException(e,sys)




class CustomData:
    def __init__(self, Age: int, TypeofContact: str, CityTier: int, DurationOfPitch: int,
                 Occupation: str,Gender: str,NumberOfPersonVisited: int, NumberOfFollowups: int,
                 ProductPitched: str,PreferredPropertyStar: int,MaritalStatus: str,
                 NumberOfTrips: int, Passport: int, FeedbackScore: int,OwnCar: int,
                 NumberOfChildrenVisited: int, Designation: str, MonthlyIncome: int):

        self.Age = Age
        self.TypeofContact = TypeofContact
        self.CityTier = CityTier
        self.DurationOfPitch = DurationOfPitch
        self.Occupation = Occupation
        self.Gender = Gender
        self.NumberOfPersonVisited = NumberOfPersonVisited
        self.NumberOfFollowups = NumberOfFollowups
        self.ProductPitched = ProductPitched
        self.PreferredPropertyStar = PreferredPropertyStar
        self.MaritalStatus = MaritalStatus
        self.NumberOfTrips = NumberOfTrips
        self.Passport = Passport

        self.FeedbackScore = FeedbackScore
        self.OwnCar = OwnCar

        self.NumberOfChildrenVisited = NumberOfChildrenVisited
        self.Designation = Designation

        self.MonthlyIncome = MonthlyIncome


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {

                "Age": [self.Age],
                "TypeofContact": [self.TypeofContact],
                "CityTier": [self.CityTier],
                "DurationOfPitch": [self.DurationOfPitch],
                "Occupation": [self.Occupation],
                "Gender": [self.Gender],
                "NumberOfPersonVisited": [self.NumberOfPersonVisited],
                "NumberOfFollowups": [self.NumberOfFollowups],
                "ProductPitched": [self.ProductPitched],
                "PreferredPropertyStar": [self.PreferredPropertyStar],
                "MaritalStatus": [self.MaritalStatus],
                "NumberOfTrips": [self.NumberOfTrips],
                "Passport": [self.Passport],
                "FeedbackScore": [self.FeedbackScore],
                "OwnCar": [self.OwnCar],
                "NumberOfChildrenVisited": [self.NumberOfChildrenVisited],
                "Designation": [self.Designation],
                "MonthlyIncome": [self.MonthlyIncome]

            }

            print("custom_data_input_dict")
            print(custom_data_input_dict)
            abc = pd.DataFrame(custom_data_input_dict)
            abc.to_csv("D:\\MachineLearningProjects\\PROJECT\TourismTrend-main\\artifacts\\data.csv",index=True,header=True)
            data1 = pd.read_csv("D:\\MachineLearningProjects\\PROJECT\\TourismTrend-main\\artifacts\\data.csv")
            check = pd.read_csv("D:\\MachineLearningProjects\\PROJECT\\TourismTrend-main\\artifacts\\check.csv")

            return data1

        except Exception as e:
            raise CustomException(e, sys)

