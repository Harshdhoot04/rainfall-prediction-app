import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Function to prepare and train the model
@st.cache(allow_output_mutation=True)
def prepare_model():
    # Load the dataset
    df = pd.read_csv('Rainfall.csv')
    
    # Data preprocessing
    df.replace({'yes': 1, 'no': 0}, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df.drop(['maxtemp', 'mintemp', 'day'], axis=1, inplace=True)
    
    # Features and target
    features = df.drop(['rainfall'], axis=1)
    target = df['rainfall']
    
    # Train-test split
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, stratify=target, random_state=2)
    
    # Handle class imbalance
    ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
    X_balanced, Y_balanced = ros.fit_resample(X_train, Y_train)
    
    # Normalize features
    scaler = StandardScaler()
    X_balanced = scaler.fit_transform(X_balanced)
    
    # Train SVC model
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_balanced, Y_balanced)
    
    return model, scaler, features.columns

# Load and train model
model, scaler, feature_names = prepare_model()

# Streamlit interface
st.title("Rainfall Prediction App")
st.write("Enter the values for the features to predict rainfall:")

# Input fields
input_data = {}
for col in feature_names:
    input_data[col] = st.number_input(f"{col.capitalize()}", value=0.0)

# Convert user input to DataFrame with correct column order
input_df = pd.DataFrame([input_data], columns=feature_names)

# Normalize the input
scaled_input = scaler.transform(input_df)

# Predict button
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    st.write("Prediction:", "Rain" if prediction[0] == 1 else "No Rain")
