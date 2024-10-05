import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from preprocessing.preprocessing import tag_columns, converting_dtypes, target_preprocessing, imputation, dates_preprocessing, \
                string_col_preprocessing_train, string_col_preprocessing_test, num_col_preprocessing_train, num_col_preprocessing_test

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
        st.session_state['predictor_column'] = predictor_column

        if predictor_column:
            threshold = 0.5 * len(train_df)  # 50% of the total number of rows
            target_col = predictor_column
            
            # Drop columns where the number of NaN values is greater than the threshold
            df_cleaned = train_df.dropna(thresh=threshold, axis=1)
            df_cleaned = df_cleaned.dropna(subset=[target_col])
            
            column_tags = tag_columns(df_cleaned, target_col)
            st.session_state['column_tags'] = column_tags
            df_cleaned = converting_dtypes(df_cleaned, column_tags)
            df_cleaned = target_preprocessing(df_cleaned, target_col)
            df_cleaned = imputation(df_cleaned, column_tags)
            df_cleaned, column_tags = dates_preprocessing(df_cleaned, column_tags)
            df_encoded, significant_cat_cols= string_col_preprocessing_train(df_cleaned, column_tags,target_col)
            df_encoded, significant_predictors = num_col_preprocessing_train(df_encoded, column_tags, target_col)  
            df_final = df_encoded[significant_cat_cols+significant_predictors+[target_col]]

            # Button to train the model
            if st.button("Train Model"):
                st.session_state['predictor_column'] = predictor_column
                
                # Store trained DataFrame in session state for later use
                st.session_state['significant_cat_cols']=significant_cat_cols
                st.session_state['significant_predictors']=significant_predictors

                # Model training process
                train_model(df_final, predictor_column)

                # Now redirect to prediction page after training
                st.session_state['model_trained'] = True
                st.session_state['model_type'] = "Regression"
                st.success("Model trained successfully! Navigate to the prediction page.")

                if st.button("Predict File"):
                    st.rerun()  # Refresh to show the updated state

def train_model(train_df, predictor_column):
    X = train_df.drop(predictor_column, axis=1)  # Replace with your target column name
    y = train_df[predictor_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hyperparameter_grids = {
        "Linear Regression": {},
        "Ridge Regression": {
            'alpha': [0.1, 1.0, 10.0]
        },
        "Lasso Regression": {
            'alpha': [0.1, 1.0, 10.0]
        },
        "Decision Tree": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "SVR": {
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.2, 0.5]
        },
        "KNN": {
            'n_neighbors': [3, 5, 7, 9]
        },
        "LightGBM": {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [-1, 10, 20]
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    }

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        # "LightGBM": LGBMRegressor(),
        # "XGBoost": XGBRegressor(eval_metric='rmse')  # Use RMSE for evaluation
    }

    best_models = {}
    best_rmse = {}

    for model_name, model in models.items():
        param_grid = hyperparameter_grids[model_name]

        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        final_rmse = np.sqrt(-grid_search.best_score_)  # Convert negative MSE to RMSE

        best_models[model_name] = best_model
        best_rmse[model_name] = final_rmse
        st.write(f"Model: {model_name}, Best Hyperparameters: {grid_search.best_params_}, Final RMSE: {final_rmse}")

     # Identify the best model based on RMSE
    best_overall_model_name = min(best_rmse, key=best_rmse.get)
    best_overall_rmse = best_rmse[best_overall_model_name]

    st.write("Comparison of RMSE across models:")
    for model, rmse in best_rmse.items():
        st.write(f"{model}: RMSE = {rmse}")

    st.write(f"Best Overall Model: {best_overall_model_name} with RMSE: {best_overall_rmse}")

    # Save the model using joblib
    joblib.dump(best_models[best_overall_model_name], f"{best_overall_model_name}.pkl")
    st.session_state['model_name'] = best_overall_model_name
    st.write(f"Saved best model '{best_overall_model_name}' to disk.") 

def regression_predict_page():
    st.title("Make Predictions")

    # Upload CSV file for prediction
    prediction_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"], key="prediction_file")
    # st.write(st.session_state)
    if prediction_file is not None and 'model_trained' in st.session_state and st.session_state['model_trained']:
        # Load the prediction CSV
        predict_df = pd.read_csv(prediction_file)
        st.subheader("Top 5 rows of the uploaded test CSV file:")
        st.write(predict_df.head())
        target_col =  st.session_state['predictor_column']
        column_tags = st.session_state['column_tags']

        predict_df = converting_dtypes(predict_df, column_tags)
        predict_df = imputation(predict_df, column_tags)
        predict_df, column_tags =  dates_preprocessing(predict_df, column_tags)
        predict_df = string_col_preprocessing_test(predict_df, column_tags)
        predict_df = num_col_preprocessing_test(predict_df, column_tags)
        significant_cat_cols = st.session_state['significant_cat_cols']
        significant_predictors = st.session_state['significant_predictors']
        predict_df_final = predict_df[significant_cat_cols+significant_predictors]

        final_model = st.session_state['model_name']
        # Make predictions
        model = joblib.load(final_model+'.pkl')  # Load the trained model
        predictions = model.predict(predict_df_final)
        
        # Add predictions to the prediction DataFrame
        predict_df_final[target_col] = predictions
        
        # Provide an option to download the modified prediction DataFrame
        st.subheader("Predictions:")
        st.write(predict_df_final.head())
        
        # Save the result to a new CSV file
        output_csv = 'predictions.csv'
        predict_df_final.to_csv(output_csv, index=False)
        st.success(f"Predictions added to the CSV file. You can download it below.")

        # Download link for the results
        with open(output_csv, 'rb') as f:
            st.download_button("Download predictions CSV", f, file_name=output_csv)
