# DiabDetect-ML-project-2
Name: DiabDetect
DiabDetect is an application designed to assist in the detection of diabetes based on various patient features like age, glucose, BMI, etc

Features:
Main Page: Users are greeted with a welcome message and prompted to upload a CSV file containing patient data.
Data Upload: The main page includes a file uploader component where users can upload their CSV file.
Data Visualization: After uploading the CSV file, users can visualize the data through various plots, including bar plots, scatter plots, histograms, and box plots. They can select different columns for the x-axis and y-axis, adjust bin sizes for histograms, and choose columns for box plots.
Prediction: Also, on same page there is an application to perform diabetes prediction. They input patient features such as number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI (Body Mass Index), diabetes pedigree function (DPF), and age.
Machine Learning Model: The application uses a machine learning model (XGBoost) trained on data to predict whether a patient has diabetes based on the input features.
Prediction Display: After inputting the patient features and clicking the predict button, the application displays the prediction outcome, indicating whether the person is diabetic or not.
Navigation: The application uses Streamlit's multi-page functionality to navigate between the main page (for data upload and visualization) and the application page (for prediction).

Technologies: The app is built using the Streamlit framework in Python. It leverages libraries such as pandas for data manipulation, seaborn for data visualization, and scikit-learn for machine learning model deployment.

Overall, DiabDetect provides users with a user-friendly interface to upload patient data, visualize the data, and obtain predictions regarding diabetes status based on input features.
