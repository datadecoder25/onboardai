import time
# from openai import OpenAI
import streamlit as st
import os
import pandas as pd
from pandasai import PandasAI
# from pandasai.llm.openai import OpenAI


def load_password(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        password = file.read().strip()  # Remove any leading/trailing whitespace
    return password

# Use the function
password_file = 'password.txt'
try:
    api_key = load_password(password_file)
except FileNotFoundError as e:
    print(e)

API_KEY = api_key
openai_client = OpenAI(api_key = API_KEY)
pandas_ai = PandasAI(openai_client)


def data_processing_page():
    st.title("Data Processing in Natural Language")
    uploaded_file = st.file_uploader("Upload a csv file for analysis", e=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(3))

        prompt = st.text_area("Enter your prompt:")
        if st.button("Generate"):
            if prompt:
                with st.spinner("generating response..."):
                    st.write(pandas_ai.run(df, prompt=prompt))
            else:
                st.write("Please enter a prompt...")