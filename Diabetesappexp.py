#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing essential libraries
import numpy as np
import pandas as pd
import streamlit as st
from pickle import load
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


col1, col2 = st.columns(2)
with col1:
    st.image("Diabeticsmart.jpeg")


# In[7]:


# Load the XGBRegressorModel
model = load(open("diabetespredictionxgbmodel.pkl", 'rb'))


# In[8]:


st.write("<h1 style='text-align: center; color: #fffff;'>GET HIGH ACCURACY DIABETES DIAGNOSIS BASED ON BELOW FEATURES:</h1>", unsafe_allow_html = True)


# In[9]:


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


# In[10]:


def predict(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age):
    features = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

if st.button('Predict'):
    prediction = predict(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age)
    if prediction >= 0.50:
        prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person is diabetic</span>' 
        #prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person is diabetic - ${prediction:.2f}</span>' if we went to display the outcome or value
        st.write(prediction_text, unsafe_allow_html = True)
    else:
        prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person has no diabetes</span>' 
        #prediction_text = f'<span style = "font-size:30px; color:#FFD700;">The person has no diabetes - ${prediction:.2f}</span>' if we want to display the outcome or
        st.write(prediction_text, unsafe_allow_html = True)


# In[ ]:




