import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Sample credentials
USER_CREDENTIALS = {
    "username": "admin",
    "password": "12345"
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
                st.experimental_rerun()  # Refresh the app to show the welcome page
            else:
                st.error("Invalid username or password")

def welcome_page():
    st.title("Welcome to AI")
    st.write("You have successfully logged in! What ML model do you want to build?")
    option = st.selectbox("Select an option", ["Select an option", "Regression", "Classification"])

    if option != "Select an option":
        st.session_state['model_type'] = option  # Store selected option
        st.experimental_rerun()  # Refresh to show the corresponding page

def model_page():
    model_type = st.session_state['model_type']
    
    if model_type == "Regression":
        regression_page()
    else:
        st.title(f"You chose: {model_type} Model")
        st.write(f"You can start building your {model_type.lower()} model here!")

def regression_page():
    st.title("Regression Model")

    # Upload CSV file for training if it hasn't been uploaded
    # if 'train_df' not in st.session_state:
    #     uploaded_file = st.file_uploader("Upload a CSV file for training", type=["csv"])
    uploaded_file = st.file_uploader("Upload a CSV file for training", type=["csv"])
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        st.session_state['train_df'] = pd.read_csv(uploaded_file)

        # Show the top 5 rows of the DataFrame
        st.subheader("Top 5 rows of the uploaded training CSV file:")
        st.write(st.session_state['train_df'].head())

    train_df = st.session_state.get('train_df', None)

    if train_df is not None:
        # Select predictor column
        st.subheader("Select Y variable (Predictor column):")
        predictor_column = st.selectbox("Choose a Y variable", train_df.columns.tolist())

        if predictor_column:
            # Get numerical columns excluding the predictor column
            numeric_columns = train_df.select_dtypes(include=['number']).columns.tolist()
            if predictor_column in numeric_columns:
                numeric_columns.remove(predictor_column)

            # Restrict the user to choose from numerical columns only
            st.subheader("Select X variables (Numerical features):")
            selected_x_columns = st.multiselect("Choose X variables", numeric_columns)

            # Imputation option
            imputation_choice = st.selectbox("How would you like to handle missing values?", 
                                             ["Drop rows with NaN values", "Fill NaN with mean"])

            # Button to train the model
            if st.button("Train Model"):
                if selected_x_columns:
                    st.session_state['predictor_column'] = predictor_column
                    
                    # Store trained DataFrame in session state for later use
                    st.session_state['selected_x_columns'] = selected_x_columns
                    st.session_state['imputation_choice'] = imputation_choice

                    # Model training process
                    train_model(train_df, predictor_column, selected_x_columns, imputation_choice)

                    # Now redirect to prediction page after training
                    st.session_state['model_trained'] = True
                    st.success("Model trained successfully! Navigate to the prediction page.")

                    st.rerun()  # Refresh to show the updated state

def train_model(train_df, predictor_column, selected_x_columns, imputation_choice):
    # Separate features and the target variable
    X = train_df[selected_x_columns]
    y = train_df[predictor_column]

    # Handle missing values based on user selection
    if imputation_choice == "Drop rows with NaN values":
        X = X.dropna()
        y = y.loc[X.index]  # Make sure y aligns with X after dropping

    elif imputation_choice == "Fill NaN with mean":
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

    # Fit the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model using joblib
    joblib.dump(model, 'linear_regression_model.pkl')

def predict_page():
    st.title("Make Predictions")

    # Upload CSV file for prediction
    prediction_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"], key="prediction_file")
    # Show the top 5 rows of the DataFrame
    st.subheader("Top 5 rows of the uploaded test CSV file:")
    st.write(prediction_file.head())

    if prediction_file is not None and 'model_trained' in st.session_state and st.session_state['model_trained']:
        # Load the prediction CSV
        predict_df = pd.read_csv(prediction_file)
        
        # Getting the relevant details
        predictor_column = st.session_state['predictor_column']
        selected_x_columns = st.session_state['selected_x_columns']
        imputation_choice = st.session_state['imputation_choice']

        # Preprocess the prediction DataFrame
        if set(selected_x_columns).issubset(set(predict_df.columns)):
            if imputation_choice == "Drop rows with NaN values":
                predict_df = predict_df.dropna()

            elif imputation_choice == "Fill NaN with mean":
                imputer = SimpleImputer(strategy='mean')
                predict_df[selected_x_columns] = imputer.fit_transform(predict_df[selected_x_columns])

            # Make predictions
            model = joblib.load('linear_regression_model.pkl')  # Load the trained model
            predictions = model.predict(predict_df[selected_x_columns])
            
            # Add predictions to the prediction DataFrame
            predict_df[predictor_column] = predictions
            
            # Provide an option to download the modified prediction DataFrame
            st.subheader("Predictions:")
            st.write(predict_df.head())
            
            # Save the result to a new CSV file
            output_csv = 'predictions.csv'
            predict_df.to_csv(output_csv, index=False)
            st.success(f"Predictions added to the CSV file. You can download it below.")

            # Download link for the results
            with open(output_csv, 'rb') as f:
                st.download_button("Download predictions CSV", f, file_name=output_csv)
        else:
            st.error("Prediction CSV does not contain the necessary columns for prediction.")

def main():
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        if 'model_type' in st.session_state:
            if 'model_trained' in st.session_state and st.session_state['model_trained']:
                predict_page()  # Direct to prediction page if model is trained
            else:
                model_page()  # Stay on the training page
        else:
            welcome_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
