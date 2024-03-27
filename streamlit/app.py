import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the trained CatBoost model
model_path = 'C:\\Users\\afshi\\Documents\\GitHub\\immo-eliza-deployment\\streamlit\\trained_catboost_model.pkl'
catboost_model = joblib.load(model_path)

# Function to preprocess data and make predictions
def preprocess_data(data):
    # Define numerical, binary, and categorical features
    numerical_features = ["cadastral_income", "surface_land_sqm", "total_area_sqm", "construction_year", 
                          "latitude", "longitude", "garden_sqm", "primary_energy_consumption_sqm",  
                          "nbr_frontages", "nbr_bedrooms", "terrace_sqm"]
    fl_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace", "fl_swimming_pool", 
                   "fl_floodzone", "fl_double_glazing"]
    cat_features = ['property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 
                    'state_building', 'epc', 'heating_type', 'equipped_kitchen']
    
    # Preprocess numerical and binary features
    numerical_data = data[numerical_features].values
    binary_data = data[fl_features].values

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    
    # One-hot encode categorical features
    encoded_features = encoder.fit_transform(data[cat_features]).toarray()

    # Concatenate numerical, binary, and encoded categorical features
    X = np.concatenate([numerical_data, binary_data, encoded_features], axis=1)
    
    return X

# Streamlit UI
st.title('House Price Prediction')
st.write('Enter the details of the house for price prediction:')

# Input fields for numerical features
cadastral_income = st.number_input('Cadastral Income', value=0)
surface_land_sqm = st.number_input('Surface Land (sqm)', value=0)
total_area_sqm = st.number_input('Total Area (sqm)', value=0)
construction_year = st.number_input('Construction Year', value=0)
latitude = st.number_input('Latitude', value=0)
longitude = st.number_input('Longitude', value=0)
garden_sqm = st.number_input('Garden (sqm)', value=0)
primary_energy_consumption_sqm = st.number_input('Primary Energy Consumption (sqm)', value=0)
nbr_frontages = st.number_input('Number of Frontages', value=0)
nbr_bedrooms = st.number_input('Number of Bedrooms', value=0)
terrace_sqm = st.number_input('Terrace (sqm)', value=0)

# Input fields for binary features
fl_garden = st.checkbox('Garden')
fl_furnished = st.checkbox('Furnished')
fl_open_fire = st.checkbox('Open Fire')
fl_terrace = st.checkbox('Terrace')
fl_swimming_pool = st.checkbox('Swimming Pool')
fl_floodzone = st.checkbox('Floodzone')
fl_double_glazing = st.checkbox('Double Glazing')

# Input fields for categorical features
property_type = st.selectbox('Property Type', ['House', 'Apartment', 'Villa'])
subproperty_type = st.selectbox('Subproperty Type', ['Studio', 'Duplex', 'Penthouse'])
region = st.text_input('Region')
province = st.text_input('Province')
locality = st.text_input('Locality')
zip_code = st.text_input('Zip Code')
state_building = st.selectbox('State of Building', ['New', 'To renovate', 'Good'])
epc = st.selectbox('EPC', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
heating_type = st.selectbox('Heating Type', ['Gas', 'Electric', 'Oil', 'Other'])
equipped_kitchen = st.checkbox('Equipped Kitchen')

# Combine inputs into a DataFrame
input_data = pd.DataFrame({
    'cadastral_income': [cadastral_income],
    'surface_land_sqm': [surface_land_sqm],
    'total_area_sqm': [total_area_sqm],
    'construction_year': [construction_year],
    'latitude': [latitude],
    'longitude': [longitude],
    'garden_sqm': [garden_sqm],
    'primary_energy_consumption_sqm': [primary_energy_consumption_sqm],
    'nbr_frontages': [nbr_frontages],
    'nbr_bedrooms': [nbr_bedrooms],
    'terrace_sqm': [terrace_sqm],
    'fl_garden': [fl_garden],
    'fl_furnished': [fl_furnished],
    'fl_open_fire': [fl_open_fire],
    'fl_terrace': [fl_terrace],
    'fl_swimming_pool': [fl_swimming_pool],
    'fl_floodzone': [fl_floodzone],
    'fl_double_glazing': [fl_double_glazing],
    'property_type': [property_type],
    'subproperty_type': [subproperty_type],
    'region': [region],
    'province': [province],
    'locality': [locality],
    'zip_code': [zip_code],
    'state_building': [state_building],
    'epc': [epc],
    'heating_type': [heating_type],
    'equipped_kitchen': [equipped_kitchen]
})

# When the user clicks the predict button
if st.button('Predict'):
    # Preprocess the input data
    X_input = preprocess_data(input_data)
    # Make prediction
    prediction = catboost_model.predict(X_input)
    # Display the prediction
    st.write(f'Predicted Price: {prediction}')

