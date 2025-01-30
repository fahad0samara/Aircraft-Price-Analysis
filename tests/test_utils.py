import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import *

class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Model': ['Boeing 737', 'Airbus A320', 'Cessna 172'],
            'Production_Year': [2020, 2019, 2018],
            'Capacity': [180, 150, 4],
            'Age': [3, 4, 5],
            'Price_USD': [1000000, 900000, 200000],
            'Fuel_Consumption_Lph': [100, 90, 10],
            'Range_km': [5000, 4800, 1200],
            'Hourly_Maintenance_Cost': [1000, 900, 100]
        })

    def test_remove_outliers(self):
        """Test outlier removal function"""
        df_with_outlier = self.test_data.copy()
        df_with_outlier.loc[3] = ['Test', 2020, 1000, 3, 99999999, 100, 5000, 1000]
        
        cleaned_df = remove_outliers(df_with_outlier, 'Price_USD')
        self.assertEqual(len(cleaned_df), len(self.test_data))

    def test_calculate_efficiency_metrics(self):
        """Test efficiency metrics calculation"""
        metrics = calculate_efficiency_metrics(self.test_data)
        
        self.assertIn('cost_per_seat_km', metrics)
        self.assertIn('fuel_per_seat_hour', metrics)
        self.assertIn('maintenance_per_seat_hour', metrics)

    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        categorical_columns = ['Model']
        encoded_df, encoders = encode_categorical_features(self.test_data, categorical_columns)
        
        self.assertTrue(encoded_df['Model'].dtype in [np.int32, np.int64])
        self.assertIn('Model', encoders)

    def test_categorize_aircraft(self):
        """Test aircraft categorization"""
        self.assertEqual(categorize_aircraft(4), 'Small')
        self.assertEqual(categorize_aircraft(50), 'Medium')
        self.assertEqual(categorize_aircraft(180), 'Large')
        self.assertEqual(categorize_aircraft(300), 'Very Large')

if __name__ == '__main__':
    unittest.main()
