# Real Estate Data Processing Pipeline

A comprehensive Python-based real estate data processing pipeline for data ingestion, cleaning, transformation, and market analysis.

## Project Overview

This project implements a complete data science pipeline for real estate market analysis, featuring automated data cleaning, statistical analysis, and visualization capabilities.

## Features

- **Data Ingestion**: Load real estate data from CSV files
- **Data Cleaning**: Handle missing values, outliers, and invalid entries
- **Statistical Analysis**: Generate comprehensive market insights
- **Data Visualization**: Create charts and graphs for market trends
- **Export Capabilities**: Save cleaned datasets for further analysis

## Project Structure

```
├── src/
│   ├── real_estate_dataset.py    # Main RealEstateDataset class
│   └── __init__.py
├── notebooks/
│   └── real_estate_analysis.ipynb # Jupyter notebook for analysis
├── data/
│   ├── raw/                      # Raw data files
│   └── cleaned/                  # Processed data files
├── demo.py                       # Command-line demo script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd "Python Task-1"
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface
```bash
python demo.py
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/real_estate_analysis.ipynb
```

### Python API
```python
from src.real_estate_dataset import RealEstateDataset

# Create dataset instance
dataset = RealEstateDataset()

# Load and process data
dataset.load_data('data/raw/housing_data.csv')
dataset.clean_data()
insights = dataset.describe_data()

# Save cleaned data
dataset.save_cleaned_data('data/cleaned/housing_data_cleaned.csv')
```

## Dataset Information

The pipeline processes real estate data with the following features:
- Property ID, Location, Type (House/Condo/Apartment)
- Price, Bedrooms, Bathrooms, Size (sqft)
- Year Built, Sales Status, Days on Market
- Additional features: HOA fees, parking, amenities

## Data Processing Pipeline

1. **Data Loading**: Read CSV files and validate structure
2. **Data Cleaning**: 
   - Handle missing values (median for numerical, mode for categorical)
   - Remove invalid entries (negative prices, invalid sizes)
   - Standardize data types and formats
3. **Data Analysis**: Generate descriptive statistics and insights
4. **Visualization**: Create comprehensive charts and graphs
5. **Export**: Save cleaned datasets in CSV format

## Results

- **Data Quality**: 99.6% retention rate after cleaning
- **Insights**: Property type distribution, price analysis by location and type
- **Visualizations**: Price distributions, correlation matrices, market trends

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## License

This project is available for educational purposes.

## Author

Real Estate Data Analysis Team
