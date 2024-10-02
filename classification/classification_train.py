import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

def classification_page():
    st.title("Classification Model")

    # Upload CSV file for training
    uploaded_file = st.file_uploader("Upload a CSV file for training", type=["csv"])
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        st.session_state['train_df'] = pd.read_csv(uploaded_file)

        # Show the top 5 rows of the DataFrame
        st.subheader("Top 5 rows of the uploaded training CSV file:")
        st.write(st.session_state['train_df'].head())

    train_df = st.session_state.get('train_df', None)

    if train_df is not None:
        # Select target column (Y)
        st.subheader("Select Y variable (Target column):")
        target_column = st.selectbox("Choose a Y variable", train_df.columns.tolist())

        if target_column:
            # Get categorical columns excluding target column
            categorical_columns = train_df.select_dtypes(include=['number']).columns.tolist()
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)

            # User chooses from categorical columns only
            st.subheader("Select X variables (Features):")
            selected_x_columns = st.multiselect("Choose X variables", categorical_columns)

            # Imputation option
            imputation_choice = st.selectbox("How would you like to handle missing values?", 
                                             ["Drop rows with NaN values", "Fill NaN with mode"])

            # Button to train the model
            if st.button("Train Model"):
                if selected_x_columns:
                    st.session_state['target_column'] = target_column
                    
                    # Store trained DataFrame in session state for later use
                    st.session_state['selected_x_columns'] = selected_x_columns
                    st.session_state['imputation_choice'] = imputation_choice

                    # Model training process
                    train_classification_model(train_df, target_column, selected_x_columns, imputation_choice)

                    # Now redirect to prediction page after training
                    st.session_state['model_trained'] = True
                    st.session_state['model_type'] = "Classification"
                    st.success("Model trained successfully! Navigate to the prediction page.")

                    st.rerun()  # Refresh to show the updated state

def train_classification_model(train_df, target_column, selected_x_columns, imputation_choice):
    # Separate features and the target variable
    X = train_df[selected_x_columns]
    y = train_df[target_column]

    # Handle missing values based on user selection
    if imputation_choice == "Drop rows with NaN values":
        X = X.dropna()
        y = y.loc[X.index]  # Make sure y aligns with X after dropping

    elif imputation_choice == "Fill NaN with mean":
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Fit the Logistic Regression model
    model = GradientBoostingClassifier()
    model.fit(X, y)

    # Save the model using joblib
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

def classification_predict_page():
    st.title("Make Predictions")

    # Upload CSV file for prediction
    prediction_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"], key="prediction_file")

    if prediction_file is not None and 'model_trained' in st.session_state and st.session_state['model_trained']:
        # Load the prediction CSV
        predict_df = pd.read_csv(prediction_file)
        st.subheader("Top 5 rows of the uploaded test CSV file:")
        st.write(predict_df.head())

        # Getting the relevant details
        target_column = st.session_state['target_column']
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
            model = joblib.load('logistic_regression_model.pkl')  # Load the trained model
            label_encoder = joblib.load('label_encoder.pkl')  # Load the trained model
            predictions = model.predict(predict_df[selected_x_columns])
            original_labels = label_encoder.inverse_transform(predictions)

            # Add predictions to the prediction DataFrame
            predict_df[target_column] = original_labels
            
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

