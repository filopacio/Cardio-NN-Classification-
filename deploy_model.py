import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestCentroid
from pathlib import Path
import pickle

CURR_DIR = Path(__file__)
MAIN_DIR = CURR_DIR.parents[1]
SELECTED_MODEL_PATH = str(MAIN_DIR / "selected_model.sav")

with open('selected_model.sav', 'rb') as file:
    pipeline = pickle.load(file)

st.title("Medical Cure Prediction")
sysbp = st.slider("Systolic Blood Pressure", 80, 200)
glucose = st.slider("Glucose Level", 50, 300)
totchol = st.slider("Total Cholesterol", 100, 400)
age = st.slider("Age", 20, 90)
bmi = st.slider("BMI", 10, 50)
cursmoke = st.checkbox("Smoker")
diabetes = st.checkbox("Diabetes")
sex = st.selectbox("Sex", ("Male", "Female"))
sex_binary = 1 if sex == "Female" else 0
features = [totchol, age, sysbp, cursmoke, bmi, diabetes, glucose, sex_binary]
X_pred = pd.DataFrame({'totchol': [totchol],
                     'sysbp': [sysbp],
                     'glucose': [glucose],
                     'age': [age],
                     'bmi': [bmi],
                     'cursmoke': [cursmoke],
                     'diabetes': [diabetes],
                     'sex': [sex_binary]})

prediction = pipeline.predict(X_pred)


if prediction == 'Yes':
    st.title(":red[You may need a medical treatment]")
else:
    st.title(":green[No medical treatment needed]")
    
st.write('DISCLAIMER: This is the conclusion of a small experimental work in data science and does not substitute a professional medical consult.')


         
