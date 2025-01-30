# Data paths
ORIGINAL_DATA_PATH = '../airplane_price_dataset.csv'
CLEANED_DATA_PATH = '../cleaned_airplane_data.csv'

# Model parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Feature engineering parameters
OUTLIER_THRESHOLD = 3  # Number of standard deviations for outlier removal
AGE_CATEGORIES = {
    'New': 5,        # 0-5 years
    'Mid-Age': 15,   # 6-15 years
    'Old': float('inf')  # >15 years
}

AIRCRAFT_SIZE_CATEGORIES = {
    'Small': 4,      # ≤4 passengers
    'Medium': 100,   # ≤100 passengers
    'Large': 250,    # ≤250 passengers
    'Very Large': float('inf')  # >250 passengers
}

# Visualization settings
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (12, 6)
DPI = 300

# Feature lists
NUMERICAL_FEATURES = [
    'Production_Year',
    'Number_of_Engines',
    'Capacity',
    'Range_km',
    'Fuel_Consumption_Lph',
    'Hourly_Maintenance_Cost',
    'Age'
]

CATEGORICAL_FEATURES = [
    'Model',
    'Engine_Type',
    'Sales_Region',
    'Aircraft_Size',
    'Age_Category'
]

# Column name mappings
COLUMN_MAPPINGS = {
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
