# Importing essential libraries
import numpy as np
import pandas as pd
import streamlit as st
from pickle import load
from PIL import Image
import treeinterpreter
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    st.sidebar.title("Welcome to DiabeticSmart- an Application to detect Diabetes")
    #st.title("Welcome to DiabeticSmart- an Application to detect Diabetes")
    #st.sidebar.header("Upload your CSV data file")
    data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    #data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Data overview:")
        st.write(data.head())

        st.sidebar.header("Visualizations")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()


    col1, col2 = st.columns(2)
    with col1:
        st.image("Diabeticsmart.jpeg")

#with col2:
    #st.image("image2.jpg")

# Load the Random Forest CLassifier model
model = load(open("diabetespredictionrfcmodelnew.pkl", 'rb'))
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
