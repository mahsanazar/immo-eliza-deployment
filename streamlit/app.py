import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Function to preprocess data and make predictions
def preprocess_data(data):
    # Extract features from artifacts
    cat_features = artifacts['features']['cat_features']
    numerical_features = artifacts['features']['num_features']
    fl_features = artifacts['features']['fl_features']
    encoder = artifacts['enc']
    
    # Preprocess numerical features
    numerical_data = data[numerical_features].values
    
    # Preprocess binary features (if any)
    binary_data = data[fl_features].values
    
    # One-hot encode categorical features using the loaded encoder
    encoded_features = encoder.transform(data[cat_features]).toarray()

    # Combine all features
    X = np.concatenate([numerical_data, binary_data, encoded_features], axis=1)
    
    return X

# Load the trained model and artifacts
artifacts = joblib.load('C:\Users\afshi\Documents\GitHub\immo-eliza-deployment\streamlit\trained_catboost_model.pklU')
catboost_model = artifacts['model']

# Streamlit UI
st.title('House Price Prediction')
st.write('Enter the details of the house for price prediction:')

# Input fields for features
cadastral_income = st.number_input('Cadastral Income', value=0.0)
surface_land_sqm = st.number_input('Surface Land (sqm)', value=0.0)
total_area_sqm = st.number_input('Total Area (sqm)', value=0.0)
construction_year = st.number_input('Construction Year', value=0)
latitude = st.number_input('Latitude', value=0.0)
longitude = st.number_input('Longitude', value=0.0)
garden_sqm = st.number_input('Garden (sqm)', value=0.0)
primary_energy_consumption_sqm = st.number_input('Primary Energy Consumption (sqm)', value=0.0)
nbr_frontages = st.number_input('Number of Frontages', value=0)
nbr_bedrooms = st.number_input('Number of Bedrooms', value=0)
terrace_sqm = st.number_input('Terrace (sqm)', value=0.0)

fl_garden = st.radio('Garden', [0, 1])
fl_furnished = st.radio('Furnished', [0, 1])
fl_open_fire = st.radio('Open Fire', [0, 1])
fl_terrace = st.radio('Terrace', [0, 1])
fl_swimming_pool = st.radio('Swimming Pool', [0, 1])
fl_floodzone = st.radio('Floodzone', [0, 1])
fl_double_glazing = st.radio('Double Glazing', [0, 1])

property_type = st.selectbox('Property Type', ['House', 'Apartment', 'Villa'])
subproperty_type = st.selectbox('Subproperty Type', ['Studio', 'Duplex', 'Penthouse'])
region = st.text_input('Region')
province = st.text_input('Province')
locality = st.text_input('Locality')
zip_code = st.text_input('Zip Code')
state_building = st.selectbox('State of Building', ['New', 'To renovate', 'Good'])
epc = st.selectbox('EPC', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
heating_type = st.selectbox('Heating Type', ['Gas', 'Electric', 'Oil', 'Other'])
equipped_kitchen = st.selectbox('Equipped Kitchen', ['Yes', 'No'])

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
