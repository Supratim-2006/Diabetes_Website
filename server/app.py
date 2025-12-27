import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

import pathlib

pikle_in = open('diabetes_model.pkl', 'rb')
model = pickle.load(pikle_in)

def Welcome():
    return "Welcome All"

def predict_note_authentication(
        Pregnancies: int,
        Glucose: float,
        BloodPressure: float,
        SkinThickness: float,
        Insulin: float,
        BMI: float,
        DiabetesPedigreeFunction: float,
        Age: int
):
    
    prediction = model.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    print(prediction)
    return prediction[0]

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)




def main():
    
    st.set_page_config(page_title="Health Input App", layout="wide", initial_sidebar_state="expanded",page_icon="🤖")



    st.title("🤖Diabetes Prediction Web App")
    st.info("This is a Diabetes Prediction Web Application built using Streamlit and Python")
    
    st.sidebar.title("Diabetes Prediction Web App\n ")

    #Check male or female to show Pregnancies input
    #a css code to change Style of selectbox
    css_path=pathlib.Path("style.css")
    load_css(css_path)


    with st.container(key="green"):
        gender = st.radio("Gender", ("Male", "Female", "Other"))
        if gender == "Female":
            Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0,)
        else:
            Pregnancies = 0


    predict=False
    with st.form("Health Form"):
        Glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=0.0)
        BloodPressure = st.number_input("BloodPressure", min_value=0.0, max_value=200.0, value=0.0)
        SkinThickness = st.number_input("SkinThickness", min_value=0.0, max_value=100.0, value=0.0)
        Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=0.0)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.0)
        Age = st.number_input("Age", min_value=0, max_value=120, value=0)

        predict=st.form_submit_button("Predict")
    if predict:
            if None in (Glucose,BloodPressure,SkinThickness,BMI,DiabetesPedigreeFunction,Age):

                st.error("❌ Please fill all the fields")
            result=0
            result = predict_note_authentication(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                                DiabetesPedigreeFunction, Age) 
            if result[0]==1:
                st.success('The output is Non Diabetic')
            else:
                st.error('The output is Diabetic')
    if "show_content" not in st.session_state:
        st.session_state.show_content = False
    if st.button("About"):
        st.session_state.show_content = not st.session_state.show_content
        
    if st.session_state.show_content:
        st.write("Developed by Supratim Kukri")
        st.write("Built with Streamlit and Python")
if __name__ == '__main__':
    main()