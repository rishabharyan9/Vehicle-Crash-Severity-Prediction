import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('xgb_classifier.joblib')

# Define the app
st.title("XGBoost Classifier Deployment with Streamlit")

# Create input fields for the features
st.header("Input Features")

# List the feature names as per your dataset (adjust accordingly)
feature_names = ['Crash_location_suburban',
	'Vehicle_make_Maruti Suzuki',
	'Crash_location_urban',
	'Vehicle_year_old_model',
	'Safety_rating',
	'Vehicle_make_Hyundai',
	'Vehicle_make_Tata Motors',
	'Vehicle_make_Mahindra',
	'Day_of_week_Saturday',	
	'Engine_type_electric',	
	'Weather_conditions_rain',	
	'Vehicle_type_pickup',	
	'Time_of_day_morning',	
	'Engine_type_diesel',	
	'Driver_age_old_age',	
	'Time_of_day_evening',	
	'Weather_conditions_fog',	
	'Time_of_day_night',	
	'Day_of_week_Tuesday',	
	'ESC_presence',
	'Vehicle_type_sedan',	
	'Day_of_week_Wednesday',	
	'Driver_gender_Male',	
	'Driver_age_young_age',	
	'Day_of_week_Sunday',	
	'TCS_presence',	
	'ABS_presence',	
	'Number_of_cylinders',	
	'TPMS_presence',
	'Day_of_week_Thursday',
	'Transmission_type_manual',
	'Day_of_week_Monday',	
	'Vehicle_weight',	
	'Road_surface_conditions_wet',
	'Engine_displacement',	
	'Engine_type_petrol',	
	'Vehicle_type_hatchback',	
	'Car_side_m2',
	'Road_surface_conditions_muddy',	
	'Car_vol_m2',
	'Number_of_airbags',
	'Car_Front_m2',	
	'Weather_conditions_rainy',
	'Weather_conditions_foggy',	
	'Weather_conditions_cloudy',	
	'Engine_type_hybrid'	]  # Replace with actual feature names
inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    inputs.append(value)

# Convert inputs to numpy array for prediction
inputs = np.array(inputs).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(inputs)
    prediction_proba = model.predict_proba(inputs)
    st.write(f"Predicted Class: {prediction[0]}")
    if prediction[0]==0:
	    st.write("Severe")
    if prediction[0]==1:
	    st.write("Moderate")
    if prediction[0]==2:
	    st.write("Minor")
    st.write(f"Prediction Probability: {prediction_proba[0]}")

# Optional: Display feature importance or other model details
if st.checkbox("Show Feature Importance"):
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, len(importances)])
    st.pyplot(plt)
