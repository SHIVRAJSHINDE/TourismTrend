from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

application = Flask(__name__)
app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return  render_template('home.html')
    else:

        data = CustomData(
            Age=int(request.form.get('Age')),
            TypeofContact=request.form.get('TypeofContact'),
            CityTier=int(request.form.get('CityTier')),
            DurationOfPitch=int(request.form.get('DurationOfPitch')),
            Occupation=request.form.get('Occupation'),
            Gender=request.form.get('Gender'),
            NumberOfPersonVisited=int(request.form.get('NumberOfPersonVisited')),
            NumberOfFollowups=int(request.form.get('NumberOfFollowups')),
            ProductPitched=request.form.get('ProductPitched'),
            PreferredPropertyStar=int(request.form.get('PreferredPropertyStar')),
            MaritalStatus=request.form.get('MaritalStatus'),
            NumberOfTrips=int(request.form.get('NumberOfTrips')),
            Passport=int(request.form.get('Passport')),
            FeedbackScore=int(request.form.get('FeedbackScore')),
            OwnCar=int(request.form.get('OwnCar')),
            NumberOfChildrenVisited=int(request.form.get('NumberOfChildrenVisited')),
            Designation=request.form.get('Designation'),
            MonthlyIncome=int(request.form.get('MonthlyIncome'))

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictionPipeline()
        result = predict_pipeline.predict(pred_df)
        print("result")

        print(result)
        return render_template('home.html',result=result[0])



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
