import streamlit as st
# import sqlite3
# import bcrypt
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from regression.regression_train import regression_page, regression_predict_page
from classification.classification_train import classification_page, classification_predict_page
from forecasting.forecasting import forecasting_page, forecast_predict_page
from nlp.math_tutor import math_solver_page
from data_processing.data_processing import data_processing_page
from st_paywall import add_auth

# # Database connection
# def create_connection():
#     conn = sqlite3.connect("users.db")
#     return conn

# def create_user_table():
#     conn = create_connection()
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             username TEXT UNIQUE,
#             password TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# # Hashing the password
# def hash_password(password):
#     return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# # Checking the hashed password
# def check_password(hashed_password, user_password):
#     return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))

# def signup(username, password):
#     conn = create_connection()
#     cursor = conn.cursor()
#     try:
#         hashed_password = hash_password(password)  # Hash the password
#         cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
#         conn.commit()
#         return "Signup successful!"  
#     except sqlite3.IntegrityError:
#         return "Username already exists. Please choose a different username."
#     finally:
#         conn.close()

# def login(username, password):
#     conn = create_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
#     result = cursor.fetchone()
    
#     if result:
#         hashed_password = result[0]
#         return check_password(hashed_password, password)  # Check hashed password
#     return False

# # Initialize the database
# create_user_table()

# # Pages
# def login_page():
#     st.title("Login Page")
#     with st.form(key='login_form'):
#         username = st.text_input("Username")
#         password = st.text_input("Password", type='password')
#         login_button = st.form_submit_button("Login")

#         if login_button:
#             if login(username, password):
#                 st.session_state['logged_in'] = True
#                 st.session_state['username'] = username
#                 st.success("Login Successful! Redirecting...")
#                 st.balloons()
#                 st.rerun()
#             else:
#                 st.error("Invalid username or password. Please try again.")

#     if st.button("Sign Up"):
#         st.session_state['signup'] = True
#         st.rerun()  # Refresh the page to show the signup form

# def signup_page():
#     st.title("Signup Page")
#     with st.form(key='signup_form'):
#         username = st.text_input("Username")
#         password = st.text_input("Password", type='password')
#         signup_button = st.form_submit_button("Sign Up")

#         if signup_button:
#             message = signup(username, password)
#             if message == "Signup successful!":
#                 st.success(message)  
#                 st.session_state['signup'] = False  
#                 st.rerun()  
#             else:
#                 st.error(message)  

def model_page():
    model_type = st.session_state['model_type']
    
    if model_type == "Regression":
        regression_page()
    elif model_type == "Classification":
        classification_page()
    elif model_type == "Math Solver":
        math_solver_page()
    elif model_type == "Data Analysis":
        data_processing_page()
    elif model_type == "Forecasting":
        forecasting_page()
    else:
        st.title(f"You chose: {model_type} Model")
        st.write(f"You can start building your {model_type.lower()} model here!")

    if st.button("Go to Model Page"):
        model_page()
        st.session_state['model_type'] = ''  # Reset the model type if needed
        # st.rerun()  # Navigate back to the model page

def welcome_page():
    st.write("You have successfully logged in! What ML model do you want to build?")
    option = st.selectbox("Select an option", ["Select an option", "Regression", "Classification", "Math Solver","Forecasting", "Data Analysis"])

    if option != "Select an option":
        st.session_state['model_type'] = option  # Store selected option
        st.rerun()  # Refresh to show the corresponding page

def main():
    # if 'logged_in' in st.session_state and st.session_state['logged_in']:
    st.title("Please login using Google")
    add_auth(required=True)
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
    # else:
    #     if 'signup' in st.session_state and st.session_state['signup']:
    #         signup_page()  
    #     else:
    #         login_page()  

if __name__ == "__main__":
    main()
