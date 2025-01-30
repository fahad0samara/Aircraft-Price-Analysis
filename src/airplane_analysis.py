import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('airplane_price_dataset.csv')

# Data Cleaning and Feature Engineering
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
    
    # Check and remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Check for missing values
    print("\nMissing Values:")
    print(df_clean.isnull().sum())
    
    # Feature Engineering
    # 1. Cost per seat (Price per passenger capacity)
    df_clean['Cost_Per_Seat'] = df_clean['Price_USD'] / df_clean['Capacity']
    
    # 2. Maintenance cost per seat
    df_clean['Maintenance_Cost_Per_Seat'] = df_clean['Hourly_Maintenance_Cost'] / df_clean['Capacity']
    
    # 3. Fuel efficiency (Fuel consumption per passenger)
    df_clean['Fuel_Efficiency'] = df_clean['Fuel_Consumption_Lph'] / df_clean['Capacity']
    
    # 4. Range efficiency (Range per fuel consumption)
    df_clean['Range_Efficiency'] = df_clean['Range_km'] / df_clean['Fuel_Consumption_Lph']
    
    # 5. Categorize aircraft by size
    def categorize_size(capacity):
        if capacity <= 4:
            return 'Small'
        elif capacity <= 100:
            return 'Medium'
        elif capacity <= 250:
            return 'Large'
        else:
            return 'Very Large'
    
    df_clean['Aircraft_Size'] = df_clean['Capacity'].apply(categorize_size)
    
    # 6. Age categories
    def categorize_age(age):
        if age <= 5:
            return 'New'
        elif age <= 15:
            return 'Mid-Age'
        else:
            return 'Old'
    
    df_clean['Age_Category'] = df_clean['Age'].apply(categorize_age)
    
    # Remove outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    numerical_columns = ['Price_USD', 'Fuel_Consumption_Lph', 'Hourly_Maintenance_Cost']
    for col in numerical_columns:
        df_clean = remove_outliers(df_clean, col)
    
    return df_clean

# Machine Learning Model
def train_price_prediction_model(df):
    # Prepare features
    # Select features for the model
    features = ['Production_Year', 'Number_of_Engines', 'Capacity', 'Range_km', 
                'Fuel_Consumption_Lph', 'Hourly_Maintenance_Cost', 'Age']
    
    # Categorical features to encode
    categorical_features = ['Model', 'Engine_Type', 'Sales_Region', 'Aircraft_Size', 'Age_Category']
    
    # Create X (features) and y (target)
    X = df[features].copy()
    
    # Add encoded categorical features
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(df[feature])
    
    y = df['Price_USD']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance in Price Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return rf_model, scaler

def advanced_analysis(df):
    # 1. Price trends over production years
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Production_Year', y='Price_USD', hue='Aircraft_Size')
    plt.title('Price Trends Over Production Years by Aircraft Size')
    plt.ylabel('Price (USD)')
    plt.xlabel('Production Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_trends.png')
    plt.close()
    
    # 2. Maintenance cost analysis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Age', y='Maintenance_Cost_Per_Seat', 
                    hue='Aircraft_Size', size='Capacity', sizes=(50, 400))
    plt.title('Maintenance Cost per Seat vs Age')
    plt.tight_layout()
    plt.savefig('maintenance_analysis.png')
    plt.close()
    
    # 3. Fuel efficiency analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Aircraft_Size', y='Fuel_Efficiency')
    plt.title('Fuel Efficiency by Aircraft Size')
    plt.ylabel('Fuel Consumption per Passenger (L/hour/passenger)')
    plt.tight_layout()
    plt.savefig('fuel_efficiency.png')
    plt.close()
    
    # 4. Range efficiency by model
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Range_Efficiency')
    plt.title('Range Efficiency by Aircraft Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('range_efficiency.png')
    plt.close()

def main():
    # Read and clean data
    print("Cleaning and preparing data...")
    df_clean = clean_data(df)
    
    # Save cleaned data
    df_clean.to_csv('cleaned_airplane_data.csv', index=False)
    print("\nCleaned data saved to 'cleaned_airplane_data.csv'")
    
    # Perform advanced analysis
    print("\nPerforming advanced analysis...")
    advanced_analysis(df_clean)
    
    # Train and evaluate the model
    print("\nTraining price prediction model...")
    model, scaler = train_price_prediction_model(df_clean)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")
    print("The following files have been created:")
    print("- cleaned_airplane_data.csv (cleaned dataset)")
    print("- price_trends.png (price trends over years)")
    print("- maintenance_analysis.png (maintenance cost analysis)")
    print("- fuel_efficiency.png (fuel efficiency analysis)")
    print("- range_efficiency.png (range efficiency by model)")
    print("- feature_importance.png (importance of features in price prediction)")

if __name__ == "__main__":
    main()
