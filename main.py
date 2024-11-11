import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('Trainingset.csv')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

y = df.iloc[:,-1]

x = df.drop(columns=["Weight"], axis=0)

from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(x, y)

model = clf

file = 'trained.sav'
pickle.dump(model,open(file,'wb'))

model_load = pickle.load(open('trained.sav','rb'))

# main
import streamlit as st
from langchain_groq import ChatGroq

# Securely load the API key with error handling
try:
    groq_api = st.secrets["GROQ"]
except KeyError:
    st.error("API Key not found. Please check your secrets.toml or app settings.")

import os
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir(os.getcwd()))

llm = 0;
# Initialize the ChatGroq model
if 'groq_api_key' in locals():
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        groq_api_key=groq_api
    )

# Load the pre-trained model
model_load = 0;

try:
    model_load = pickle.load(open('trained.sav','rb'))
except FileNotFoundError:
    st.error("The trained model file 'trained.sav' was not found.")

# Custom CSS for better styling
st.markdown("""
    <style>
        .title {
            color: #4CAF50;
            font-size: 50px;
            font-weight: bold;
        }
        .section-title {
            font-size: 24px;
            margin-top: 30px;
            font-weight: bold;
            color: #48ed07;
        }
        .input-box {
            background-color: #f0f8ff;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .generate-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            font-size: 18px;
        }
        .gender-warning {
            font-size: 15px;
            font-weight: bold;
            color: #ff0303;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<p class="title">🏋️ Training Schedule Maker</p>', unsafe_allow_html=True)
st.write("Provide your height, weight, and training goals to get a personalized gym schedule tailored to your body metrics.")

# Section: Input Fields
st.markdown('<p class="section-title">🔢 Enter Your Information</p>', unsafe_allow_html=True)

# Initialization of variables
height = st.text_input("Enter your height (in feet)", placeholder="e.g., 5.8")
weight = st.text_input("Enter your weight (in kg)", placeholder="e.g., 70")
sex = st.text_input("Enter your Gender", placeholder="e.g., male or female")
interest = st.text_input("What are your fitness goals?", placeholder="e.g., weight loss, muscle gain, etc.")

# Input validation and processing
try:
    height = float(height)  # Convert height to a float
except ValueError:
    st.warning("Please enter a valid height (e.g., 5.8).")
    height = None

try:
    weight = float(weight)  # Convert weight to a float
except ValueError:
    st.warning("Please enter a valid weight in kg (e.g., 70).")
    weight = None

# Handling gender input
if sex.lower() == 'male':
    sex_code = 1
elif sex.lower() == 'female':
    sex_code = 0
else:
    st.markdown('<p class="gender-warning">Please enter your gender correctly (male or female).</p>', unsafe_allow_html=True)
    sex_code = None

# Prediction model for ideal weight range
weight_max = 0
weight_min = 0

if height is not None and sex_code is not None:
    try:
        weight_max = model_load.predict([[height, sex_code]])[0]
        weight_min = weight_max - 10
    except Exception as e:
        st.error(f"Error in model prediction: {str(e)}")

# Button to generate the gym schedule
st.markdown('<p class="section-title">⚙️ Generate Your Gym Schedule</p>', unsafe_allow_html=True)

if st.button("Generate Schedule", key="generate"):
    if height and weight and interest and sex_code is not None:
        # Determine fitness goal based on weight
        if weight <= weight_min:
            st.write("Generating a schedule for weight gain.")
            output_str = f"Generate a weekly gym schedule with diet plan for a {height} feet tall {sex} with {weight} kg weight, interested in weight gain. {interest}"
        elif weight >= weight_max:
            st.write("Generating a schedule for weight loss.")
            output_str = f"Generate a weekly gym schedule with diet plan for a {height} feet tall {sex} with {weight} kg weight, interested in weight loss. {interest}"
        else:
            st.write("Generating a schedule for maintaining body fitness.")
            output_str = f"Generate a weekly gym schedule with diet plan for a {height} feet tall {sex} with {weight} kg weight, interested in normal exercise to maintain body. {interest}"

        # Get the response from the LLM
        try:
            response = llm.invoke(output_str)
            st.markdown('<p class="section-title">🏋️‍♂️ Your Gym Schedule</p>', unsafe_allow_html=True)
            st.success(response.content)
        except Exception as e:
            st.error(f"Error in generating gym schedule: {str(e)}")
    else:
        st.warning("Please fill in all the fields before generating the schedule.")
