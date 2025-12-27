

#------------Import-------------------------------------------------------------------------------

import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import pathlib


def get_clean_data():
    data=pd.read_csv("model/diabetes.csv")
    return data

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def add_sidebar():
    st.sidebar.header("Details")
    data = get_clean_data()

    slider_labels=[
        ("Pregnancies","Pregnancies"),
        ("Glucose(mg/dL)","Glucose"),
        ("BloodPressure(mmHg)","BloodPressure"),
        ("SkinThickness(mm)","SkinThickness"),
        ("Insulin(mU/ml)","Insulin"),
        ("BMI","BMI"),
        ("DiabetesPedigreeFunction","DiabetesPedigreeFunction"),
        ("Age","Age")
    ]
    input_data={}

    for label,key in slider_labels:
            if key=="Pregnancies" or key=="Age":
                input_data[key]=st.sidebar.slider(  
                label,
                min_value=int(data[key].min()),
                max_value=int(data[key].max()),
                value=int(data[key].mean()),
                key=key
                )
            else:
                input_data[key]=st.sidebar.slider(
                label,
                min_value=float(data[key].min()),
                max_value=float(data[key].max()),
                value=float(data[key].mean()),
                key=key
                )

    return input_data
    

def get_radar_chart(input_data):
    categories = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    mean_values = np.array([[3, 120.89, 69.11, 20.54, 79.79, 31.99, 0.47, 33]])

    user_values = np.array([[
        input_data['Pregnancies'],
        input_data['Glucose'],
        input_data['BloodPressure'],
        input_data['SkinThickness'],
        input_data['Insulin'],
        input_data['BMI'],
        input_data['DiabetesPedigreeFunction'],
        input_data['Age']
    ]])

    
    scaler = StandardScaler()
    scaler.fit(mean_values)  

    
    scaled_mean = scaler.transform(mean_values).flatten()
    scaled_user = scaler.transform(user_values).flatten()

    
    fig = go.Figure()

    
    fig.add_trace(go.Scatterpolar(
        r=scaled_mean,
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))

    
    fig.add_trace(go.Scatterpolar(
        r=scaled_user,
        theta=categories,
        fill='toself',
        name='User Input'
    ))

    

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )
        ),
        showlegend=True)
    return fig


def add_Prediction(input_data):

    st.subheader("Pediction")
    st.write("The Patient is ")
    model=pickle.load(open('model/diabetes_model.pkl','rb'))
    scaler=pickle.load(open('model/scaler.pkl','rb'))

    input_array=np.array([list(input_data.values())])

    input_array_scaled=scaler.transform(input_array)
    prediction=model.predict(input_array_scaled)
    if prediction[0]==1:
        st.write("<span class='Diabetic yes'>Diabetic</span>", unsafe_allow_html=True) 
    else:
        st.write("<span class='Diabetic no'>Non-Diabetic</span>", unsafe_allow_html=True) 
    st.write("Probability of being Diabetic: ",model.predict_proba(input_array_scaled)[0][0]*100,"%")
    st.write("Probability of being non-Diabetic: ",model.predict_proba(input_array_scaled)[0][1]*100,"%")
    st.write("The prediction shown is solely on the basis of dataset provided and should not be used as a substitute for professional diagnosis.")



def main():
    
    st.set_page_config(page_title="Health Input App", layout="wide", initial_sidebar_state="expanded",page_icon="🤖")

    with open("assests/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("🤖Diabetes Prediction Web App")

    col1,col2=st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    
    with col2:
        add_Prediction(input_data)

    st.info("This is a Diabetes Prediction Web Application built using Streamlit and Python")

    css_path=pathlib.Path("style.css")
    load_css(css_path)

    if "show_content" not in st.session_state:
        st.session_state.show_content = False
    if st.button("About"):
        st.session_state.show_content = not st.session_state.show_content
        
    if st.session_state.show_content:
        st.write("Developed by Supratim Kukri")
        st.write("Built with Streamlit and Python")



if __name__ == '__main__':
    main()