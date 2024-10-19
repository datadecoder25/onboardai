import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from regression.regression_train import regression_page, regression_predict_page
from classification.classification_train import classification_page, classification_predict_page
from forecasting.forecasting import forecasting_page, forecast_predict_page
from nlp.math_tutor import math_solver_page

# Sample credentials
USER_CREDENTIALS = {
    "username": "admin",
    "password": "1"
}

def login(username, password):
    return username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']

def login_page():
    st.title("Login Page")
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if login(username, password):
                st.session_state['logged_in'] = True
                st.success("Login Successful! Redirecting...")
                st.balloons()
                st.rerun()  # Refresh the app to show the welcome page
            else:
                st.error("Invalid username or password")

def welcome_page():
    st.title("Welcome to AI")
    st.write("You have successfully logged in! What ML model do you want to build?")
    option = st.selectbox("Select an option", ["Select an option", "Regression", "Classification", "Math Solver","Forecasting"])

    if option != "Select an option":
        st.session_state['model_type'] = option  # Store selected option
        st.rerun()  # Refresh to show the corresponding page

def model_page():
    model_type = st.session_state['model_type']
    
    if model_type == "Regression":
        regression_page()
    elif model_type == "Classification":
        classification_page()
    elif model_type == "Math Solver":
        math_solver_page()
    elif model_type == "Forecasting":
        forecasting_page()
    else:
        st.title(f"You chose: {model_type} Model")
        st.write(f"You can start building your {model_type.lower()} model here!")

    if st.button("Go to Model Page"):
        model_page()
        st.session_state['model_type'] = ''  # Reset the model type if needed
        # st.rerun()  # Navigate back to the model page


def main():
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        if 'model_type' in st.session_state:
            if 'model_trained' in st.session_state and st.session_state['model_trained']:
                if st.session_state['model_type'] == "Regression":
                    regression_predict_page()
                elif st.session_state['model_type'] == "Classification":
                    classification_predict_page()
                elif st.session_state['model_type'] == "Forecasting":
                    forecast_predict_page()
            else:
                model_page()  # Stay on the training page
        else:
            welcome_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
    