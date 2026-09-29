import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
#Cross origin resource shell
from  fastapi.middleware.cors import CORSMiddleware

# Loading model
model = joblib.load("Mental_Health_Model.pkl")
top_countries = ["Other", "India", "USA","Canada","Australia", "UK", "Germany", "Mexico", "Turkey","France",]

# Creating FastAPI object
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# First pydantic model
class studentData(BaseModel):
    age                         : int  =   Field(..., ge=10, le=100)
    gender                      :Literal['Male', 'Female']
    country                     : str
    academic_level              : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform          : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat' ,
                                  'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat' ]
    purpose_of_use              : Literal['Netwroking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours       : float = Field(..., ge=0, le=24)
    daily_unlocks               : int   = Field(..., ge=0)
    study_hours                 : int   = Field(..., ge=0, le=24)
    physical_activity_hours     : float = Field(..., ge=0, le=24)
    sleep_hours_per_night       : float = Field(..., ge=0, le=24)
    stress_level                : Literal['Medium', 'Low', 'Very High', 'High']


#Describe what we want to send back 

class PredictionResponse(BaseModel):
    predicated_mental_health_score: float  

# Creating GET endpoint
@app.get("/")
def func():
    return {"message": "Welcome to SCOUT AI"}

@app.post("/Predict", response_model=PredictionResponse)
def Predict(data: studentData):

    country_group = data.country if data.country in top_countries else "Other"
    input_row = pd.DataFrame(
        [
            {
                "Age": data.age,
                "Gender": data.gender,
                "Country": data.country,
                "Academic_Level": data.academic_level,
                "Most_Used_Platform": data.most_used_platform,
                "Purpose_Of_Use": data.purpose_of_use,
                "Avg_Daily_Usage_Hours": data.avg_daily_usage_hour,
                "Daily_Unlocks": data.daily_unlocks,
                "Study_Hours": data.study_hour,
                "Physical_Activity_Hours": data.physical_activity_hours,
                "Sleep_Hours_Per_Night": data.sleep_hour_per_night,
                "Stress_Level": data.stress_level,
                "Grouped_country": country_group,
            }
        ]
    )

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicated_mental_health_score=round(float(prediction),2))
