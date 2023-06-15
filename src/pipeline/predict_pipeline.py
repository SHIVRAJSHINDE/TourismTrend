import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            print(features)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, Age: int, TypeofContact: str, CityTier: int, DurationOfPitch: int, Occupation: str,Gender: str, NumberOfPersonVisited: int,
                 NumberOfFollowups: int, ProductPitched: str,PreferredPropertyStar: int, MaritalStatus: str, NumberOfTrips: int, Passport: str,
                 PitchSatisfactionScore: int, OwnCar: str, NumberOfChildrenVisited: int, Designation: str,MonthlyIncome: int):


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
        self.PitchSatisfactionScore = PitchSatisfactionScore
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
                "PitchSatisfactionScore": [self.PitchSatisfactionScore],
                "OwnCar": [self.OwnCar],
                "NumberOfChildrenVisited": [self.NumberOfChildrenVisited],
                "Designation": [self.Designation],
                "MonthlyIncome": [self.MonthlyIncome]
            }
            abc = pd.DataFrame(custom_data_input_dict)
            abc.to_csv('D:/ProjectIneuron/ML/TourismTrend/artifacts/abc.csv')
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
