import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

from streamlit_option_menu import option_menu

st.title(":rainbow[My Penguin Prediction App] 🐧")
st.divider()

with st.sidebar:
    mainmenu = option_menu("📍 Main Menu", ['🏠 Home', 'Information 📎', 'Predictions 📊'],
                           default_index=0)

if mainmenu == '🏠 Home':
    st.header('Welcome to this super app! 📱')
    st.subheader(':violet[_Here you can browse into a new world of penguins._]')
    #st.divider()

    container = st.container(border=True)

    container.markdown(""":violet[Information :] This app was made for a course pertaining to :violet[**MLOps**] 
             at :violet[_Université Lyon 2_] headed by [Fanilo ANDRIANASOLO](https://github.com/andfanilo).
                    The aim of this project was to create a _full-stack Dockerized ML app_, using a pretrained 
                    ML model.
             """)
    #st.divider()

    st.image('./images/chonkycat.jpg', caption='A Chonky Cat')
    col1, col2 = st.columns(2)
    with col1:
        st.image('./images/chonkypingoo.webp', caption='A Chonky Pingoo generated by CRAIYON (https://www.craiyon.com)')
    with col2:
        st.image('./images/adeliepingoo.webp', caption='An Adelie Pingoo generated by CRAIYON (https://www.craiyon.com)')

if mainmenu == 'Information 📎':
    st.title(":green[INFORMATION] 📎")
    st.divider()

    container = st.container()

    container.markdown("""
## **Penguin Species Prediction**
""")
    
    container.markdown("""
### _Dataset_
""")

    container.markdown("""
To create this app, I used the Palmer Archipelago (Antarctica) penguin dataset to train a Logistic Regression model. 
                       All the information on this dataset is available [here](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris). However, this dataset is about 
                    the classification of penguin species on the basis of several characteristics such as culmen length (mm), 
                    culmen depth (mm), flipper length (mm), and body mass (g) - features that I used to train a logistic regression.
""")
    
    #container.markdown("""

#""")
    
    container.markdown("""
### _Some Vocabulary_
                       
                       
The culmen is the upper ridge of a bird's beak. Flippers are the wings of a penguin. On the basis of these 
                    characteristics, we can predict which species a penguin belongs to, that is, Adélie, Gentoo or Chinstrap.
""")
    
    container.markdown("""
This dataset is made up of:
""")
    
    container.markdown("""
- 152 Adélie penguins,
- 124 Gentoo penguins, 
- 68 Chinstrap penguins.
""")

API_URL = "http://server:8000/predict"

if mainmenu == 'Predictions 📊':
    st.title(":orange[PREDICTIONS] 📊")
    st.header("_Here you can determine whether your penguin is an Adélie, Gentoo, or Chinstrap one._")
    st.divider()

    st.subheader("Select features' values to predict your penguin species:")
    st.markdown("""Thanks to a Logistic Regression model, you can predict the species of a penguin based on 
                the following characteristics.
                """)

    container = st.container(border=True)

    culmen_length = container.slider('Culmen Length (mm)', 30, 60)
    culmen_depth = container.slider('Culmen Depth (mm)', 10, 25)
    flipper_length = container.slider('Flipper Length (mm)', 150, 250)
    body_mass = container.slider('Body Mass (g)', 2500, 6500)
    delta_15n = container.slider('Delta 15 N', 5, 15)
    bouton = container.button('Predict!')

    if bouton:
        data = {
            "culmen_length": culmen_length,
            "culmen_depth": culmen_depth,
            "flipper_length": flipper_length,
            "body_mass": body_mass,
            "delta_15n": delta_15n
        }

        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            prediction = response.json()
            st.divider()
            st.subheader(':green[Predicted Class:] 🔎')
            st.write(prediction)
            st.balloons()
        else:
            st.error(":red[Failed to make predictions. Please try again.]")

st.divider()
st.caption('_Made by Annabelle NARSAMA (https://github.com/annarsama) - March 2024_')

