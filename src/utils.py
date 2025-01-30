import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """
    Load data from CSV file
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def remove_outliers(df, column, n_std=3):
    """
    Remove outliers using standard deviation method
    """
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] <= mean + (n_std * std)) & 
             (df[column] >= mean - (n_std * std))]

def calculate_efficiency_metrics(df):
    """
    Calculate various efficiency metrics
    """
    metrics = {}
    
    # Cost per seat kilometer
    metrics['cost_per_seat_km'] = df['Price_USD'] / (df['Capacity'] * df['Range_km'])
    
    # Fuel efficiency (liters per seat per hour)
    metrics['fuel_per_seat_hour'] = df['Fuel_Consumption_Lph'] / df['Capacity']
    
    # Maintenance cost per flight hour per seat
    metrics['maintenance_per_seat_hour'] = df['Hourly_Maintenance_Cost'] / df['Capacity']
    
    return metrics

def encode_categorical_features(df, categorical_columns):
    """
    Encode categorical features using LabelEncoder
    """
    encoded_df = df.copy()
    encoders = {}
    
    for column in categorical_columns:
        if column in df.columns:
            le = LabelEncoder()
            encoded_df[column] = le.fit_transform(df[column])
            encoders[column] = le
    
    return encoded_df, encoders

def scale_numerical_features(df, numerical_columns):
    """
    Scale numerical features using StandardScaler
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df_scaled, scaler

def calculate_age_based_depreciation(df):
    """
    Calculate depreciation based on age
    """
    # Assuming 5% annual depreciation
    annual_depreciation_rate = 0.05
    df['depreciation_factor'] = (1 - annual_depreciation_rate) ** df['Age']
    df['estimated_original_price'] = df['Price_USD'] / df['depreciation_factor']
    return df

def generate_summary_statistics(df):
    """
    Generate summary statistics for numerical columns
    """
    summary = {}
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_columns:
        summary[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skew': df[col].skew()
        }
    
    return summary

def calculate_correlations(df):
    """
    Calculate correlations between numerical features
    """
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    return df[numerical_columns].corr()

def categorize_aircraft(capacity):
    """
    Categorize aircraft based on capacity
    """
    if capacity <= 4:
        return 'Small'
    elif capacity <= 100:
        return 'Medium'
    elif capacity <= 250:
        return 'Large'
    else:
        return 'Very Large'
