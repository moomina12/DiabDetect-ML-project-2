# Importing essential libraries
import numpy as np
import pandas as pd
import streamlit as st
from pickle import load
import matplotlib.pyplot as plt
import seaborn as sns
import Image

st.sidebar.title("Welcome to DiabeticSmart- an Application to detect Diabetes")
col1, col2 = st.columns(2)
with col1:
    st.image("Docgif.gif")
#with col2:
    #st.image("image2.jpg")

# Load the Random Forest CLassifier model
model = load(open("diabetespredictionRFbmodel.pkl", 'rb'))
#with below code tried to change the background color of sidebar in streamlit but it was unsuccessful
#st.markdown("<style>.sidebar .sidebar-content { background-color: #fffff; }</style>", unsafe_allow_html=True)


#st.sidebar.markdown("<h2 style='text-align: center;'>DiabDetect</h1>", unsafe_allow_html=True)
#st.sidebar.header("         DiabDetect")

#st.title("DiabDetect")
#st.image("Diabeticsmart.png")
st.write("<h1 style='text-align: center; color: #fffff;'>GET HIGH ACCURACY DIABETES DIAGNOSIS BASED ON BELOW FEATURES:</h1>", unsafe_allow_html = True)
#st.write("Get High Accuracy diabetes diagnosis on Patient data </h1>", unsafe_allow_html = True)
#st.write('SMART. \n INTELLIGENT. \nMODERN DIABETES DIAGNOSIS TOOL. \nGet High Accuracy diabetes diagnosis on Patient data')
#st.image("Diabeticsmart.png")

Pregnancies = st.slider('No of pregnanices',0,20,5)
Glucose=st.slider('Glucose',0,500)
BloodPressure=st.slider('Blood Pressure',50,150)
SkinThickness=st.slider('SkinThickness',0,100)
Insulin=st.slider('Insulin',0,1000)
BMI = st.slider('BMI', 12.0, 50.0, 25.0, step = 0.2)
# Set the initial value of the input to 0.062
DPF = st.number_input('DPF(DiabetesPedigreeFunction)', min_value=0.000, max_value=1.000,format="%.3f" )
#DPF=st.slider('DPF',0.000,1.000,format="%.3f")
Age = st.slider('Age', 0, 100, 25)

def detect(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age):
    features = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

if st.button('DetectDiabetes'):
    prediction = detect(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age)
    if prediction == 1:
        prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person is diabetic</span>' 
        #prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person is diabetic - ${prediction:.2f}</span>' if we went to display the outcome or value
        st.write(prediction_text, unsafe_allow_html = True)
    else:
        prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person has no diabetes</span>' 
        #prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person has no diabetes - ${prediction:.2f}</span>' if we want to display the outcome or
        st.write(prediction_text, unsafe_allow_html = True)
