from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="API for predicting real estate prices based on input features.",
    version="1.0.0",
)

# Load the trained model

# Load the model artifacts using joblib
artifacts = joblib.load("C:\\Users\\afshi\\Documents\\GitHub\\immo-eliza-deployment\\Updated_trained_catboost_model.pkl")

# Unpack the artifacts
num_features = artifacts["features"]["num_features"]
fl_features = artifacts["features"]["fl_features"]
cat_features = artifacts["features"]["cat_features"]

imputer = artifacts["imputer"]
enc = artifacts["enc"]
model = artifacts["model"]




class InputData(BaseModel):
    cadastral_income: float
    surface_land_sqm: float 
    total_area_sqm: float 
    construction_year: int
    latitude: float 
    longitude: float 
    garden_sqm: float 
    primary_energy_consumption_sqm: float 
    nbr_frontages: int
    nbr_bedrooms: int 
    terrace_sqm: float 
    fl_garden: int 
    fl_furnished: int 
    fl_open_fire: int
    fl_terrace: int 
    fl_swimming_pool: int 
    fl_floodzone: int 
    fl_double_glazing: int 
    property_type: str 
    subproperty_type: str 
    region: str 
    province: str
    locality: str
    zip_code: str 
    state_building: str 
    epc: str 
    heating_type: str 
    equipped_kitchen: str 


# New route at the root path
@app.get("/")
async def read_root():
    return {"message": "alive: immoliza deployment project"}

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Impute missing values using mean for numerical features
        input_data[num_features] = imputer.transform(input_data[num_features])

        # One-hot encode categorical features
        input_data_cat_encoded = enc.transform(input_data[cat_features]).toarray()

        # Concatenate numerical and encoded categorical features
        # Combine the numerical and one-hot encoded categorical columns
        input_data = pd.concat(
            [
                input_data[num_features + fl_features].reset_index(drop=True),
                pd.DataFrame(input_data_cat_encoded, columns=enc.get_feature_names_out()),
            ],
            axis=1,
        )

        # Make prediction
        prediction = model.predict(input_data)[0]  # Assuming model returns a single prediction

        return {"predicted_price": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
