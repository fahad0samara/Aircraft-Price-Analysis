#!/usr/bin/env python
# coding: utf-8

# # Aircraft Price Analysis
# 
# This notebook contains a comprehensive analysis of aircraft prices and their determining factors.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ## 1. Data Loading and Cleaning

def load_data():
    """Load the raw aircraft price dataset."""
    df = pd.read_csv('../data/airplane_price_dataset.csv')
    return df

def clean_data(df):
    """Clean and preprocess the data."""
    df_clean = df.copy()
    
    # Rename columns to English
    column_names = {
        'Üretim Yılı': 'Production_Year',
        'Motor Sayısı': 'Number_of_Engines',
        'Motor Türü': 'Engine_Type',
        'Kapasite': 'Capacity',
        'Menzil (km)': 'Range_km',
        'Yakıt Tüketimi (L/saat)': 'Fuel_Consumption_Lph',
        'Saatlik Bakım Maliyeti ($)': 'Hourly_Maintenance_Cost',
        'Yaş': 'Age',
        'Satış Bölgesi': 'Sales_Region',
        'Fiyat ($)': 'Price_USD'
    }
    df_clean = df_clean.rename(columns=column_names)
    
    # Remove outliers using IQR method
    numerical_cols = ['Price_USD', 'Capacity', 'Range_km', 'Fuel_Consumption_Lph', 'Hourly_Maintenance_Cost']
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

# ## 2. Feature Engineering

def engineer_features(df):
    """Create new features for analysis."""
    df = df.copy()
    
    # Cost per seat
    df['Cost_Per_Seat'] = df['Price_USD'] / df['Capacity']
    
    # Maintenance cost per seat
    df['Maintenance_Cost_Per_Seat'] = df['Hourly_Maintenance_Cost'] / df['Capacity']
    
    # Fuel efficiency
    df['Fuel_Efficiency'] = df['Fuel_Consumption_Lph'] / df['Capacity']
    
    # Range efficiency
    df['Range_Efficiency'] = df['Range_km'] / df['Fuel_Consumption_Lph']
    
    # Price per range km
    df['Price_Per_Range'] = df['Price_USD'] / df['Range_km']
    
    # Operational cost index
    df['Operational_Cost_Index'] = (df['Hourly_Maintenance_Cost'] + 
                                  df['Fuel_Consumption_Lph'] * 100) / df['Capacity']
    
    # Age factor
    df['Age_Factor'] = np.exp(-0.05 * df['Age'])
    
    # Categories
    df['Size_Category'] = pd.qcut(df['Capacity'], q=4, 
                                labels=['Small', 'Medium', 'Large', 'Extra Large'])
    df['Range_Category'] = pd.qcut(df['Range_km'], q=4, 
                                 labels=['Short', 'Medium', 'Long', 'Ultra Long'])
    
    return df

# ## 3. Visualization Functions

def plot_price_distribution(df):
    """Plot price distribution by aircraft model."""
    fig = px.box(df, x='Model', y='Price_USD', 
                 title='Price Distribution by Aircraft Model',
                 color='Model')
    return fig

def plot_price_vs_age(df):
    """Plot price vs age relationship."""
    fig = px.scatter(df, x='Age', y='Price_USD',
                    color='Model', size='Capacity',
                    title='Price vs Age by Aircraft Model',
                    hover_data=['Production_Year'])
    return fig

def plot_efficiency_metrics(df):
    """Plot efficiency metrics."""
    fig1 = px.scatter(df, x='Capacity', y='Fuel_Consumption_Lph',
                     color='Model', title='Fuel Consumption vs Capacity',
                     hover_data=['Range_km'])
    
    fig2 = px.scatter(df, x='Age', y='Hourly_Maintenance_Cost',
                     color='Model', title='Maintenance Cost vs Age',
                     hover_data=['Production_Year'])
    
    return fig1, fig2

def plot_market_analysis(df):
    """Plot market analysis visualizations."""
    fig1 = px.pie(df, names='Sales_Region',
                  title='Aircraft Distribution by Region')
    
    fig2 = px.histogram(df, x='Age', color='Model',
                       title='Age Distribution by Model')
    
    return fig1, fig2

# ## 4. Price Prediction Model

def train_prediction_model(df):
    """Train the price prediction model."""
    numeric_features = [
        'Production_Year', 'Number_of_Engines', 'Capacity', 'Range_km',
        'Fuel_Consumption_Lph', 'Hourly_Maintenance_Cost', 'Age',
        'Cost_Per_Seat', 'Maintenance_Cost_Per_Seat', 'Fuel_Efficiency',
        'Range_Efficiency', 'Price_Per_Range', 'Operational_Cost_Index',
        'Age_Factor'
    ]
    
    categorical_features = ['Model', 'Sales_Region', 'Size_Category', 'Range_Category']
    
    # Prepare features
    X = df[numeric_features].copy()
    X_cat = pd.get_dummies(df[categorical_features])
    X = pd.concat([X, X_cat], axis=1)
    y = df['Price_USD']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_features]),
        columns=numeric_features,
        index=X_train.index
    )
    X_train_scaled = pd.concat([X_train_scaled, X_train.drop(columns=numeric_features)], axis=1)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, numeric_features, X_train_scaled.columns

def predict_price(model, scaler, features, input_data):
    """Make price predictions using the trained model."""
    # Scale numeric features
    input_scaled = scaler.transform(input_data[numeric_features])
    input_scaled = pd.DataFrame(input_scaled, columns=numeric_features)
    
    # Add categorical features
    input_cat = pd.get_dummies(input_data[categorical_features])
    input_final = pd.concat([input_scaled, input_cat], axis=1)
    
    # Ensure all features are present
    for col in features:
        if col not in input_final.columns:
            input_final[col] = 0
    
    # Make prediction
    prediction = model.predict(input_final[features])[0]
    return prediction

# ## 5. Main Analysis

def main():
    # Load and prepare data
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df = engineer_features(df_clean)
    
    # Train model
    model, scaler, numeric_features, features = train_prediction_model(df)
    
    # Example prediction
    example_data = pd.DataFrame({
        'Production_Year': [2020],
        'Number_of_Engines': [2],
        'Capacity': [50],
        'Range_km': [3000],
        'Fuel_Consumption_Lph': [8.42],
        'Hourly_Maintenance_Cost': [2782],
        'Age': [5],
        'Model': ['Bombardier CRJ200'],
        'Sales_Region': ['Asia']
    })
    
    prediction = predict_price(model, scaler, features, example_data)
    print(f"Predicted price: ${prediction:,.2f}")

if __name__ == "__main__":
    main()
