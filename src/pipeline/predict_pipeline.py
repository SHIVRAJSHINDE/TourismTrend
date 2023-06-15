import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object
from sklearn.preprocessing import StandardScaler


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            print("features")

            print(features)
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)

            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            print(features)
            data_scaled = preprocessor.transform(features)
            #scaler = StandardScaler()

            #data_scaled=scaler.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, Age: int, TypeofContact: str, CityTier: int, DurationOfPitch: int, Occupation: str,Gender: str,
                 NumberOfPersonVisited: int,NumberOfFollowups: int, ProductPitched: str,PreferredPropertyStar: int,
                 MaritalStatus: str, NumberOfTrips: int, Passport: str,PitchSatisfactionScore: int, OwnCar: str,
                 NumberOfChildrenVisited: int, Designation: str,MonthlyIncome: int):
        self.Age = Age
        self.CityTier = CityTier
        self.DurationOfPitch = DurationOfPitch
        self.NumberOfPersonVisited = NumberOfPersonVisited
        self.NumberOfFollowups = NumberOfFollowups

        if ProductPitched == "Basic":
            self.ProductPitched = 1
        elif ProductPitched == "Standard":
            self.ProductPitched = 2
        elif ProductPitched == "Deluxe":
            self.ProductPitched = 3
        elif ProductPitched == "Super Deluxe":
            self.ProductPitched = 4
        elif ProductPitched == "King":
            self.ProductPitched = 5

        self.PreferredPropertyStar = PreferredPropertyStar
        self.NumberOfTrips = NumberOfTrips

        if Passport == "Yes":
            self.Passport = 1
        elif Passport == "No":
            self.Passport = 1

        self.PitchSatisfactionScore = PitchSatisfactionScore

        if OwnCar == "Yes":
            self.OwnCar = 1
        elif OwnCar == "No":
            self.OwnCar = 0

        self.NumberOfChildrenVisited = NumberOfChildrenVisited
        if Designation == "Executive":
            self.Designation = 1
        elif Designation == "Manager":
            self.Designation = 2
        elif Designation == "Senior Manager":
            self.Designation = 3
        elif Designation == "AVP":
            self.Designation = 4
        elif Designation == "AVP":
            self.Designation = 5

        self.MonthlyIncome = MonthlyIncome

        if TypeofContact == "Self Enquiry":
            self.TypeofContact_Self_Enquiry = 1
        elif TypeofContact == "Company Invited":
            self.TypeofContact_Self_Enquiry = 0

        if Occupation == "Large Business":
            self.Occupation_Large_Business=1
            self.Occupation_Salaried=0
            self.Occupation_Small_Business=0
        elif Occupation == "Salaried":
            self.Occupation_Large_Business=0
            self.Occupation_Salaried=1
            self.Occupation_Small_Business=0
        elif Occupation == "Small Business":
            self.Occupation_Large_Business=0
            self.Occupation_Salaried=0
            self.Occupation_Small_Business=1
        elif Occupation == "Free Lancer":
            self.Occupation_Large_Business=0
            self.Occupation_Salaried=0
            self.Occupation_Small_Business=0

        if Gender == "Male":
            self.Gender_Female=0
            self.Gender_Male = 1
        elif Gender == "Female":
            self.Gender_Female=1
            self.Gender_Male = 0
        elif Gender == "Fe male":
            self.Gender_Female=0
            self.Gender_Male = 0

        if MaritalStatus == "Married":
            self.MaritalStatus_Married = 1
            self.MaritalStatus_Single = 0
            self.MaritalStatus_Unmarried = 0
        elif MaritalStatus == "Single":
            self.MaritalStatus_Married = 0
            self.MaritalStatus_Single = 1
            self.MaritalStatus_Unmarried = 0
        elif MaritalStatus == "Unmarried":
            self.MaritalStatus_Married = 0
            self.MaritalStatus_Single = 0
            self.MaritalStatus_Unmarried = 1
        elif MaritalStatus == "Divorced":
            self.MaritalStatus_Married = 0
            self.MaritalStatus_Single = 0
            self.MaritalStatus_Unmarried = 0


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "CityTier": [self.CityTier],
                "DurationOfPitch": [self.DurationOfPitch],
                "NumberOfPersonVisited": [self.NumberOfPersonVisited],
                "NumberOfFollowups": [self.NumberOfFollowups],
                "ProductPitched": [self.ProductPitched],
                "PreferredPropertyStar": [self.PreferredPropertyStar],
                "NumberOfTrips": [self.NumberOfTrips],
                "Passport": [self.Passport],
                "PitchSatisfactionScore": [self.PitchSatisfactionScore],
                "OwnCar": [self.OwnCar],
                "NumberOfChildrenVisited": [self.NumberOfChildrenVisited],
                "Designation": [self.Designation],
                "MonthlyIncome": [self.MonthlyIncome],
                "TypeofContact_Self Enquiry": [self.TypeofContact_Self_Enquiry],
                "Occupation_Large Business": [self.Occupation_Large_Business],
                "Occupation_Salaried": [self.Occupation_Salaried],
                "Occupation_Small Business": [self.Occupation_Small_Business],
                "Gender_Female": [self.Gender_Female],
                "Gender_Male": [self.Gender_Male],
                "MaritalStatus_Married": [self.MaritalStatus_Married],
                "MaritalStatus_Single": [self.MaritalStatus_Single],
                "MaritalStatus_Unmarried": [self.MaritalStatus_Unmarried]

            }

            abc = pd.DataFrame(custom_data_input_dict)
            xyz = abc[['Age','CityTier','DurationOfPitch','NumberOfPersonVisited','NumberOfFollowups','ProductPitched',
                       'PreferredPropertyStar','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisited',
                       'Designation','MonthlyIncome','TypeofContact_Self Enquiry','Occupation_Large Business',
                       'Occupation_Salaried','Occupation_Small Business','Gender_Female','Gender_Male','MaritalStatus_Married',
                       'MaritalStatus_Single','MaritalStatus_Unmarried']]

            xyz.to_csv('D:/ProjectIneuron/ML/TourismTrend/artifacts/xyz.csv',index=False,header=True)
            return xyz

        except Exception as e:
            raise CustomException(e, sys)
