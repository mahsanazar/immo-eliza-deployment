# immo-eliza-deployment


This repository contains the deployment code for the ImmoEliza Real Estate Price Prediction project.


## Overview

The deployment consists of two main components:

1. **API**: This folder contains the FastAPI application (`app.py`), Dockerfile, and requirements.txt necessary for running the prediction API.
2. **Streamlit**: This folder contains the Streamlit application for interacting with the prediction model.

## API

The API is built using FastAPI framework. It exposes an endpoint `/predict` that accepts POST requests with input data in JSON format. The API loads a pre-trained machine learning model and uses it to predict real estate prices based on the provided input features.

### How to Use

1. Navigate to the `api` folder.
2. Install the necessary dependencies by running:
    ```
    pip install -r requirements.txt
    ```
3. Build the Docker image using the provided Dockerfile:
    ```
    docker build -t immoeliza-api .
    ```
4. Run a container using the built Docker image:
    ```
    docker run -p 8000:8000 immoeliza-api
    ```
5. Access the API at `http://localhost:8000`.

### Example Input

```json
{
  "cadastral_income": 1000.0,
  "surface_land_sqm": 200.0,
  "total_area_sqm": 300.0,
  "construction_year": 2000,
  "latitude": 50.0,
  "longitude": 5.0,
  "garden_sqm": 50.0,
  "primary_energy_consumption_sqm": 200.0,
  "nbr_frontages": 2,
  "nbr_bedrooms": 3,
  "terrace_sqm": 20.0,
  "fl_garden": 1,
  "fl_furnished": 0,
  "fl_open_fire": 1,
  "fl_terrace": 0,
  "fl_swimming_pool": 1,
  "fl_floodzone": 0,
  "fl_double_glazing": 1,
  "property_type": "house",
  "subproperty_type": "detached",
  "region": "Brussels",
  "province": "Brussels",
  "locality": "Brussels",
  "zip_code": "1000",
  "state_building": "good",
  "epc": "A",
  "heating_type": "gas",
  "equipped_kitchen": "yes"
}

## Demo

You can find a live demo of the deployed API at [https://immo-eliza-deployment-31.onrender.com](https://immo-eliza-deployment-31.onrender.com)(https://immo-eliza-deployment-31.onrender.com/docs#/default/predict_predict_post) 
Feel free to explore the API and make predictions using the provided input features.

## Author
- [Mahsa Nazarian](https://github.com/mahsanazar)
