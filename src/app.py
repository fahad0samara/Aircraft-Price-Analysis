import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils import *
from config import *

# Page config
st.set_page_config(
    page_title="Aircraft Price Analysis",
    page_icon="✈️",
    layout="wide"
)

# Data cleaning function
def clean_data(df):
    # Create a copy of the dataframe
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
    
    # Remove outliers using IQR method for numerical columns
    numerical_cols = ['Price_USD', 'Capacity', 'Range_km', 'Fuel_Consumption_Lph', 'Hourly_Maintenance_Cost']
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    # Feature Engineering
    # 1. Cost per seat (Price per passenger capacity)
    df_clean['Cost_Per_Seat'] = df_clean['Price_USD'] / df_clean['Capacity']
    
    # 2. Maintenance cost per seat
    df_clean['Maintenance_Cost_Per_Seat'] = df_clean['Hourly_Maintenance_Cost'] / df_clean['Capacity']
    
    # 3. Fuel efficiency (Fuel consumption per passenger)
    df_clean['Fuel_Efficiency'] = df_clean['Fuel_Consumption_Lph'] / df_clean['Capacity']
    
    # 4. Range efficiency (Range per fuel consumption)
    df_clean['Range_Efficiency'] = df_clean['Range_km'] / df_clean['Fuel_Consumption_Lph']
    
    # 5. Price per range km
    df_clean['Price_Per_Range'] = df_clean['Price_USD'] / df_clean['Range_km']
    
    # 6. Operational cost index
    df_clean['Operational_Cost_Index'] = (df_clean['Hourly_Maintenance_Cost'] + 
                                        df_clean['Fuel_Consumption_Lph'] * 100) / df_clean['Capacity']
    
    # 7. Age factor (exponential decay)
    df_clean['Age_Factor'] = np.exp(-0.05 * df_clean['Age'])
    
    # 8. Size category
    df_clean['Size_Category'] = pd.qcut(df_clean['Capacity'], q=4, labels=['Small', 'Medium', 'Large', 'Extra Large'])
    
    # 9. Range category
    df_clean['Range_Category'] = pd.qcut(df_clean['Range_km'], q=4, labels=['Short', 'Medium', 'Long', 'Ultra Long'])
    
    return df_clean

def train_model(df):
    # Create feature matrix
    numeric_features = [
        'Production_Year',
        'Number_of_Engines',
        'Capacity',
        'Range_km',
        'Fuel_Consumption_Lph',
        'Hourly_Maintenance_Cost',
        'Age',
        'Cost_Per_Seat',
        'Maintenance_Cost_Per_Seat',
        'Fuel_Efficiency',
        'Range_Efficiency',
        'Price_Per_Range',
        'Operational_Cost_Index',
        'Age_Factor'
    ]
    
    categorical_features = ['Model', 'Sales_Region', 'Size_Category', 'Range_Category']
    
    # Create feature matrix
    X = df[numeric_features].copy()
    
    # Create categorical features DataFrame
    X_cat = pd.DataFrame()
    for feature in categorical_features:
        dummies = pd.get_dummies(df[feature], prefix=feature)
        X_cat = pd.concat([X_cat, dummies], axis=1)
    
    # Combine numeric and categorical features
    X = pd.concat([X, X_cat], axis=1)
    
    # Target variable
    y = df['Price_USD']
    
    # Ensure X and y have the same index
    X = X.loc[y.index]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the numeric features
    scaler = StandardScaler()
    X_train_numeric = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_features]),
        columns=numeric_features,
        index=X_train.index
    )
    
    # Combine scaled numeric features with categorical features
    X_train_categorical = X_train.drop(columns=numeric_features)
    X_train_scaled = pd.concat([X_train_numeric, X_train_categorical], axis=1)
    
    # Train the model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, numeric_features, categorical_features, X_train_scaled.columns

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    try:
        # Use absolute path to the data file
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'airplane_price_dataset.csv')
        
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            return None
            
        df = pd.read_csv(data_path)
        df_clean = clean_data(df)
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title("✈️ Aircraft Price Analysis Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analysis of aircraft prices and their determining factors.
    Explore different aspects of the data through various visualizations and analysis tools.
    """)

    # Load data
    df = load_and_prepare_data()
    
    if df is None:
        st.error("Could not load the data. Please check if the data file exists in the correct location.")
        return
        
    # Sidebar
    st.sidebar.header("Filters")
    selected_model = st.sidebar.multiselect(
        "Select Aircraft Models",
        options=df['Model'].unique(),
        default=df['Model'].unique()[:2]
    )
    
    selected_region = st.sidebar.multiselect(
        "Select Sales Regions",
        options=df['Sales_Region'].unique(),
        default=df['Sales_Region'].unique()[:2]
    )
    
    # Filter data based on selection
    mask = (df['Model'].isin(selected_model)) & (df['Sales_Region'].isin(selected_region))
    filtered_df = df[mask]

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Efficiency Metrics", "Market Analysis", "Price Prediction"])
    
    with tab1:
        st.header("Price Analysis")
        
        # Price distribution
        fig_price = px.box(filtered_df, x='Model', y='Price_USD', 
                         title='Price Distribution by Aircraft Model',
                         color='Model')
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Price vs Age scatter plot
        fig_age = px.scatter(filtered_df, x='Age', y='Price_USD',
                           color='Model', size='Capacity',
                           title='Price vs Age by Aircraft Model',
                           hover_data=['Production_Year'])
        st.plotly_chart(fig_age, use_container_width=True)

    with tab2:
        st.header("Efficiency Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fuel efficiency
            fig_fuel = px.scatter(filtered_df, 
                                x='Capacity', 
                                y='Fuel_Consumption_Lph',
                                color='Model',
                                title='Fuel Consumption vs Capacity',
                                hover_data=['Range_km'])
            st.plotly_chart(fig_fuel, use_container_width=True)
        
        with col2:
            # Maintenance cost
            fig_maint = px.scatter(filtered_df,
                                 x='Age',
                                 y='Hourly_Maintenance_Cost',
                                 color='Model',
                                 title='Maintenance Cost vs Age',
                                 hover_data=['Production_Year'])
            st.plotly_chart(fig_maint, use_container_width=True)

    with tab3:
        st.header("Market Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional distribution
            fig_region = px.pie(filtered_df, 
                              names='Sales_Region',
                              title='Aircraft Distribution by Region')
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            # Age distribution
            fig_age_dist = px.histogram(filtered_df,
                                      x='Age',
                                      color='Model',
                                      title='Age Distribution by Model')
            st.plotly_chart(fig_age_dist, use_container_width=True)

    with tab4:
        st.header("Price Prediction Model")
        
        # Show price ranges for each model
        st.subheader("Current Price Ranges by Model")
        price_ranges = df.groupby('Model').agg({
            'Price_USD': ['min', 'mean', 'max']
        })['Price_USD'].round(2)
        
        st.dataframe(price_ranges)
        
        # Create prediction interface
        st.subheader("Predict Aircraft Price")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_name = st.selectbox("Aircraft Model", df['Model'].unique())
            production_year = st.number_input("Production Year", 
                                            min_value=1980, 
                                            max_value=2025, 
                                            value=2020)
            engines = st.number_input("Number of Engines", 
                                    min_value=1, 
                                    max_value=4, 
                                    value=2)
        
        with col2:
            # Get typical capacity for selected model
            typical_capacity = df[df['Model'] == model_name]['Capacity'].median()
            capacity = st.number_input("Capacity (passengers)", 
                                     min_value=1, 
                                     max_value=500, 
                                     value=int(typical_capacity))
            
            # Get typical range for selected model
            typical_range = df[df['Model'] == model_name]['Range_km'].median()
            range_km = st.number_input("Range (km)", 
                                     min_value=1000, 
                                     max_value=20000, 
                                     value=int(typical_range))
            
            # Get typical fuel consumption for selected model
            typical_fuel = df[df['Model'] == model_name]['Fuel_Consumption_Lph'].median()
            fuel_consumption = st.number_input("Fuel Consumption (L/hour)", 
                                             min_value=1.0, 
                                             max_value=50.0, 
                                             value=float(typical_fuel))
        
        with col3:
            # Get typical maintenance cost for selected model
            typical_maintenance = df[df['Model'] == model_name]['Hourly_Maintenance_Cost'].median()
            maintenance_cost = st.number_input("Hourly Maintenance Cost ($)", 
                                             min_value=100, 
                                             max_value=5000, 
                                             value=int(typical_maintenance))
            region = st.selectbox("Sales Region", df['Sales_Region'].unique())
        
        if st.button("Predict Price"):
            try:
                # Train the model
                model, scaler, numeric_features, categorical_features, feature_names = train_model(df)
                
                # Prepare input data
                input_data = pd.DataFrame({
                    'Production_Year': [production_year],
                    'Number_of_Engines': [engines],
                    'Capacity': [capacity],
                    'Range_km': [range_km],
                    'Fuel_Consumption_Lph': [fuel_consumption],
                    'Hourly_Maintenance_Cost': [maintenance_cost],
                    'Age': [2025 - production_year],
                    'Cost_Per_Seat': [maintenance_cost / capacity],
                    'Maintenance_Cost_Per_Seat': [maintenance_cost / capacity],
                    'Fuel_Efficiency': [fuel_consumption / capacity],
                    'Range_Efficiency': [range_km / fuel_consumption],
                    'Price_Per_Range': [0],  # Will be updated after prediction
                    'Operational_Cost_Index': [(maintenance_cost + fuel_consumption * 100) / capacity],
                    'Age_Factor': [np.exp(-0.05 * (2025 - production_year))],
                    'Model': [model_name],
                    'Sales_Region': [region]
                })
                
                # Add size and range categories
                # Calculate quartiles for capacity and range
                capacity_quartiles = df['Capacity'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
                range_quartiles = df['Range_km'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
                
                # Create bins with quartiles
                def get_category(value, edges, labels):
                    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                        if lower <= value <= upper:
                            return labels[i]
                    return labels[-1]  # Return last category if above all edges
                
                # Assign categories
                input_data['Size_Category'] = get_category(
                    capacity,
                    capacity_quartiles,
                    ['Small', 'Medium', 'Large', 'Extra Large']
                )
                
                input_data['Range_Category'] = get_category(
                    range_km,
                    range_quartiles,
                    ['Short', 'Medium', 'Long', 'Ultra Long']
                )

                # Create one-hot encoded features
                X_input = pd.DataFrame()
                
                # Process numeric features
                X_input_numeric = input_data[numeric_features].copy()
                X_input_numeric_scaled = pd.DataFrame(
                    scaler.transform(X_input_numeric),
                    columns=numeric_features
                )
                
                # Process categorical features
                X_input_cat = pd.DataFrame()
                for feature in categorical_features:
                    dummies = pd.get_dummies(input_data[feature], prefix=feature)
                    X_input_cat = pd.concat([X_input_cat, dummies], axis=1)
                
                # Combine numeric and categorical features
                X_input = pd.concat([X_input_numeric_scaled, X_input_cat], axis=1)
                
                # Ensure all necessary columns are present
                for col in feature_names:
                    if col not in X_input.columns:
                        X_input[col] = 0
                
                # Ensure correct column order
                X_input = X_input[feature_names]
                
                # Make prediction
                prediction = model.predict(X_input)[0]
                
                # Get price range for selected model
                model_price_range = price_ranges.loc[model_name]
                
                # Calculate confidence score based on similarity to training data
                confidence_score = min(100, 100 * (1 - abs(prediction - model_price_range['mean']) / model_price_range['mean']))
                
                # Display results
                st.success(f"Predicted Price: ${prediction:,.2f}")
                
                st.info(f"""
                Price Range for {model_name}:
                - Minimum: ${model_price_range['min']:,.2f}
                - Average: ${model_price_range['mean']:,.2f}
                - Maximum: ${model_price_range['max']:,.2f}
                
                Confidence Score: {confidence_score:.1f}%
                """)
                
                # Warning if prediction is outside typical range
                if prediction < model_price_range['min'] or prediction > model_price_range['max']:
                    st.warning("""
                    ⚠️ The predicted price is outside the typical range for this model. 
                    This might be due to unusual combination of features or market conditions.
                    Consider adjusting the input parameters to be closer to typical values for this model.
                    """)
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(feature_importance, x='Feature', y='Importance',
                                      title='Top 10 Most Important Features')
                st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.info("Please try adjusting the input values or contact support if the error persists.")
            
if __name__ == "__main__":
    main()
