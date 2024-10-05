import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# Function to check the percentage of digits in a string
def is_digit_heavy_string(value):
    if not isinstance(value, str):
        return False
    digit_count = sum(c.isdigit() for c in value)
    return digit_count / len(value) >= 0.9 if len(value) > 0 else False

def is_date_string(value):
    try:
        pd.to_datetime(value)
        return True
    except (ValueError, TypeError):
        return False

def custom_is_alphanumeric(value):
    if not isinstance(value, str):
        return False
    
    letters = sum(c.isalpha() for c in value)
    digits = sum(c.isdigit() for c in value)
    
    # Apply the rules
    if len(value) > 4:
        return letters >= 2 and digits >= 2
    else:
        return letters >= 1 and digits >= 1

# Function to tag columns
def tag_columns(df, target_col):
    column_tags = {'number':[],'string':[],'unknown':[],'date':[],'alphanumeric':[], 'predictor':target_col}
    
    for column in df.loc[:, df.columns != target_col]:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_tags['number'].append(column) 
        elif pd.api.types.is_string_dtype(df[column]):
            # Check if at least 90% of the strings in the column are "digit-heavy"
            if df[column].apply(is_digit_heavy_string).mean() >= 0.9:
                column_tags['number'].append(column) 
            elif df[column].apply(is_date_string).mean() >= 0.9:
                column_tags['date'].append(column)
            elif df[column].apply(custom_is_alphanumeric).mean() >= 0.9:
                column_tags['alphanumeric'].append(column)
            else:
                column_tags['string'].append(column) 
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            column_tags['date'].append(column) 
        else:
            column_tags['unknown'].append(column)
    return column_tags

def converting_dtypes(df_cleaned, column_tags):
    # Get tags for the columns
    
    for col in column_tags['number']:
        df_cleaned[col] = df_cleaned[col].astype(float)
    
    for col in column_tags['string']:
        df_cleaned[col] = df_cleaned[col].astype(str)
    
    for col in column_tags['date']:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col], format='%Y_%m_%d')

    return df_cleaned

def target_preprocessing(df_cleaned, target_col):
    if pd.api.types.is_numeric_dtype(df_cleaned[target_col]):
        print("Target is Numeric")
        return df_cleaned
    elif pd.api.types.is_string_dtype(df_cleaned[target_col]):
        df_cleaned[target_col] = df_cleaned[target_col].replace({'$': '', ',': '','%':''}, regex=True)
        if df_cleaned[target_col].astype(str).apply(is_digit_heavy_string).mean() >= 0.9:
            print("Target is not Numeric... converting it into numeric...")
            df_cleaned[target_col] = df_cleaned[target_col].astype(float)
            return df_cleaned
        else:
            print(f'{target_col} cant be converted to numeric column')
            return pd.DataFrame()
    else:
        print(f'{target_col} cant be converted to numeric column') 
        return pd.DataFrame()

def imputation(df_cleaned, column_tags):
    for col in column_tags['string']:
        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    for col in column_tags['date']:
        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    for col in column_tags['number']:
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

    return df_cleaned

def dates_preprocessing(df_cleaned, column_tags):
    new_column_tags = column_tags.copy()
    today = pd.to_datetime('today')
    for col in new_column_tags['date']:
        # Calculate the difference in days
        df_cleaned['days_difference_'+col] = (today - df_cleaned[col]).dt.days
        new_column_tags['number'].append('days_difference_'+col)
        
    df_cleaned = df_cleaned.drop(new_column_tags['date'], axis=1)
    return df_cleaned, new_column_tags

def string_col_preprocessing_train(df_cleaned, column_tags, target_col):
    # Optionally: Trim whitespace
    df_cleaned[column_tags['string']] = df_cleaned[column_tags['string']].apply(lambda x: x.str.strip())
    
    cardinality = {col: df_cleaned[col].nunique() for col in column_tags['string']}
    low_cardinality_cols = [col for col in column_tags['string'] if cardinality[col] < 10]
    df_encoded = pd.get_dummies(df_cleaned, columns=low_cardinality_cols, drop_first=True)
    
    new_column_names = [col for col in df_encoded.columns if col.startswith(tuple(low_cardinality_cols))]
    original_to_new = {col: [new_col for new_col in new_column_names if new_col.startswith(col)] for col in low_cardinality_cols}
    
    significant_cat_cols = []
    for col in new_column_names:
        grouped = df_encoded.groupby(col)[target_col].mean()
        f_val, p_val = stats.f_oneway(*[group[target_col].values for name, group in df_encoded.groupby(col)])
        if p_val<=0.05:
           significant_cat_cols.append(col) 
    
    return df_encoded, significant_cat_cols

def string_col_preprocessing_test(df_cleaned, column_tags):
    # Optionally: Trim whitespace
    df_cleaned[column_tags['string']] = df_cleaned[column_tags['string']].apply(lambda x: x.str.strip())
    df_encoded = pd.get_dummies(df_cleaned, columns=column_tags['string'])

    return df_encoded

def num_col_preprocessing_train(df_encoded, column_tags, target_col):
    today = pd.to_datetime('today')
    #checking important numeric columns
    # for col in column_tags['number']:
    #     df_encoded[col].fillna(df_encoded[col].mean(), inplace=True)  # You can also use median or a specific value
    
    numeric_cols = column_tags['number'].copy()
    for col in column_tags['number']:
        if 'year' in col.lower():
            df_encoded['year_difference_'+col] = df_encoded[col].apply(lambda x: (today.year - x))
            numeric_cols.append('year_difference_'+col)
            numeric_cols.remove(col)
    
    # Calculate the correlation matrix
    correlation_matrix = df_encoded[numeric_cols+[target_col]].corr()
    
    # Extract the correlation with the target variable
    target_correlation = correlation_matrix[target_col].abs().sort_values(ascending=False)
    # Set a correlation threshold
    correlation_threshold = 0.2
    
    # Filter significant columns based on the threshold
    significant_numeric_columns = target_correlation[target_correlation > correlation_threshold].index.tolist()
    significant_numeric_columns = [c for c in significant_numeric_columns if c!= target_col]
    # print(f"Significant Numeric Columns: {significant_numeric_columns}")
    
    # Prepare X (features) and y (target variable)
    X = df_encoded[significant_numeric_columns]
    y = df_encoded[target_col]
    
    # Add a constant to the model (intercept)
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Store the significant columns based on p-values from the model summary
    p_values = model.pvalues
    
    # Filter significant predictors (excluding the constant)
    significant_predictors = p_values[p_values <= 0.05].index.tolist()
    if 'const' in significant_predictors:
        significant_predictors.remove('const')  # Remove the constant term

    return df_encoded, significant_predictors

def num_col_preprocessing_test(df_encoded, column_tags):
    today = pd.to_datetime('today')
    for col in column_tags['number']:
        if 'year' in col.lower():
            df_encoded['year_difference_'+col] = df_encoded[col].apply(lambda x: (today.year - x))
    return df_encoded

# def main():
#     model_dict = {}
#     # Load the dataset
#     file_path = '/Users/sprosad/Downloads/others/new_dea/onboardai/data/regression/house_price_train.csv'  # Update with your file path
#     df = pd.read_csv(file_path)
#     threshold = 0.5 * len(df)  # 70% of the total number of rows
#     target_col = 'SalePrice'
    
#     # Drop columns where the number of NaN values is greater than the threshold
#     df_cleaned = df.dropna(thresh=threshold, axis=1)
#     df_cleaned = df_cleaned.dropna(subset=[target_col])
    
#     column_tags = tag_columns(df_cleaned, target_col)
#     df_cleaned = converting_dtypes(df_cleaned, column_tags)
#     df_cleaned = target_preprocessing(df_cleaned, target_col)
#     df_cleaned = imputation(df_cleaned, column_tags)
#     df_cleaned, column_tags = dates_preprocessing(df_cleaned, column_tags)
#     df_encoded, significant_cat_cols = string_col_preprocessing_train(df_cleaned, column_tags,target_col)
#     df_encoded, significant_predictors = num_col_preprocessing_train(df_encoded, column_tags, target_col)  
#     df_final = df_encoded[significant_cat_cols+significant_predictors+[target_col]]
#     model_dict['significant_cat_cols']=significant_cat_cols
#     model_dict['significant_predictors']=significant_predictors
    
#     print("DONE...!!!")
#     return df_final, model_dict