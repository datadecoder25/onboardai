import streamlit as st
import pandas as pd
import joblib
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def forecasting_page():
    st.title("Forecasting Model")

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
        st.subheader("Select date column:")
        date_column = st.selectbox("Choose date column", train_df.columns.tolist())
        st.session_state['date_column'] = date_column

        st.subheader("Select predictor/forecast column:")
        value_column = st.selectbox("Choose predictor/forecast column", train_df.columns.tolist())
        st.session_state['value_column'] = value_column

        st.subheader("Enter the no of months you want to forecast:")
        forecast_periods = st.number_input('Enter no of forecast periods: ')
        forecast_periods = int(forecast_periods)

        if date_column and value_column and forecast_periods>0:
            if st.button("Train Model"):
                st.write("Trying out different models to best fit your data. Hold on...")
                progress_bar = st.progress(0)
                data = train_df #, parse_dates=[date_column], index_col=date_column)
                data.set_index(date_column, inplace=True)
                data.index = pd.to_datetime(data.index)
                data[value_column].fillna(method='ffill', inplace=True)
                sarima_output_df = pd.DataFrame()
                try:
                    # Initialize SARIMA model
                    sarima_model = auto_arima(data[value_column], 
                                            seasonal=True, 
                                            m=12,
                                            trace=True, 
                                            error_action='warn', 
                                            suppress_warnings=True, 
                                            stepwise=True)
                    
                    # SARIMA Forecasting
                    # n_periods = 12  # Number of periods to forecast
                    sarima_forecast, sarima_conf_int = sarima_model.predict(n_periods=forecast_periods, return_conf_int=True)
                    
                    # Create a DataFrame for the forecast
                    sarima_forecast_df = pd.DataFrame(sarima_forecast).reset_index().rename({"index":"date",0: "forecast"}, axis=1)
                    sarima_conf_df = pd.DataFrame({
                        'lower_bound': sarima_conf_int[:, 0],
                        'upper_bound': sarima_conf_int[:, 1]
                    })
                    
                    sarima_output_df = pd.concat([sarima_forecast_df,sarima_conf_df], axis=1)
                except:
                    st.write("Trying different models...")

                prophet_forecast_df = pd.DataFrame()
                try:
                    prophet_data = data.copy()
                    prophet_data = prophet_data.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
                    prophet_model = Prophet()
                    prophet_model.fit(prophet_data)
                    
                    # Forecasting with Prophet
                    # future = prophet_model.make_future_dataframe(periods=n_periods, freq='M')
                    future = pd.DataFrame(sarima_forecast.index, columns=['ds'])
                    prophet_forecast = prophet_model.predict(future)

                    prophet_forecast_df = prophet_forecast[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={"ds":"date","yhat":"forecast","yhat_lower":"lower_bound","yhat_upper":"upper_bound"})
                except:
                    st.write("found problem in the data...")

                try:
                    # Assume you have actual values for validation (e.g., last n_periods in historical data)
                    actual_values = data[value_column][-forecast_periods:]
                    
                    # SARIMA Performance
                    sarima_mape = mean_absolute_percentage_error(actual_values, sarima_forecast)
                    
                    # Prophet Performance
                    prophet_mape = mean_absolute_percentage_error(actual_values, prophet_forecast['yhat'][-forecast_periods:])
                    

                    if sarima_mape<=prophet_mape:
                        st.session_state['forecast_df']= sarima_output_df
                    else:
                        st.session_state['forecast_df']= prophet_forecast_df
                except:
                    if len(prophet_forecast_df)>0:
                        st.session_state['forecast_df']= prophet_forecast_df
                    else:
                        st.session_state['forecast_df']= sarima_output_df

                st.session_state['model_trained'] = True
                st.session_state['model_type'] = "Forecasting"
                progress_bar.progress(1.0)
                st.success("Model trained successfully! Navigate to the prediction page.")

                if st.button("Predict File"):
                    st.rerun()  # Refresh to show the updated state 

def forecast_predict_page():
    st.title("Make Predictions")

    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        # Load the prediction CSV
        predict_df = st.session_state['forecast_df']

        st.subheader("Forecast:")
        st.write(predict_df.head())
        
        # Save the result to a new CSV file
        output_csv = 'predictions.csv'
        predict_df.to_csv(output_csv, index=False)
        st.success(f"Predictions added to the CSV file. You can download it below.")

        # Download link for the results
        with open(output_csv, 'rb') as f:
            st.download_button("Download predictions CSV", f, file_name=output_csv)
