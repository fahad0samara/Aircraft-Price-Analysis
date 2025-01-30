# Aircraft Price Analysis Project

## Overview
This project analyzes aircraft pricing data to understand various factors affecting aircraft prices and builds a machine learning model for price prediction.

## Dataset
The dataset (`airplane_price_dataset.csv`) contains information about various aircraft including:
- Model types (Airbus, Boeing, Bombardier, Cessna)
- Technical specifications (engines, capacity, range)
- Operating costs (fuel consumption, maintenance)
- Market data (age, sales region, price)

## Features
- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis
- Price prediction model using Random Forest
- Comprehensive visualizations

## Project Structure
```
├── data/
│   ├── airplane_price_dataset.csv    # Original dataset
│   └── cleaned_airplane_data.csv     # Cleaned and processed data
├── src/
│   ├── airplane_analysis.py          # Main analysis script
│   ├── utils.py                      # Utility functions
│   └── config.py                     # Configuration settings
├── visualizations/                   # Generated plots
│   ├── price_trends.png
│   ├── maintenance_analysis.png
│   ├── fuel_efficiency.png
│   ├── range_efficiency.png
│   └── feature_importance.png
├── tests/                           # Test files
│   └── test_utils.py
├── requirements.txt                 # Project dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## Setup and Installation
1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the main analysis:
```python
python src/airplane_analysis.py
```

## Results
The analysis includes:
- Price prediction model with 97.6% accuracy (R² score)
- Comprehensive visualizations of aircraft characteristics
- Detailed analysis of maintenance costs and fuel efficiency
- Price trends across different aircraft categories

## Visualizations
- `price_trends.png`: Price trends over production years by aircraft size
- `maintenance_analysis.png`: Maintenance costs analysis
- `fuel_efficiency.png`: Fuel efficiency comparisons
- `range_efficiency.png`: Range efficiency by model
- `feature_importance.png`: Feature importance in price prediction

## License
MIT License
