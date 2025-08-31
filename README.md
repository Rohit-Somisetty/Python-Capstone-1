# Real Estate Data Processing Pipeline

A comprehensive Python-based real estate data processing pipeline for data ingestion, cleaning, transformation, and market analysis.

## Project Overview

This project implements a complete data science pipeline for real estate market analysis, featuring automated data cleaning, statistical analysis, and visualization capabilities.


## Features

- **Data Ingestion**: Load real estate data from CSV files
- **Data Cleaning**: Handle missing values, outliers, and invalid entries
- **Statistical Analysis**: Generate comprehensive market insights
- **Advanced Analysis**: Calculate price per square foot, property age, and more
- **Data Visualization**: Create charts and graphs for market trends, including price distributions, top locations, and price trends
- **Export Capabilities**: Save cleaned datasets for further analysis
## Advanced Analysis & Visualization

This project includes two extensible classes for advanced data analysis and visualization:

### Analysis Class
- Takes a cleaned DataFrame as input
- Methods:
   - `calculate_price_per_sqft()`: Adds a `Price_per_sqft` column
   - `calculate_property_age()`: Adds a `Property_Age` column

### Visualization Class (inherits from Analysis)
- Generates visualizations using matplotlib/seaborn
- Methods:
   - `plot_price_distribution(property_type=None)`: Boxplot of price per sqft by property type (polymorphic for any type)
   - `plot_top_locations(top_n=5)`: Barplot of top N locations by property count
   - `plot_price_trends()`: Line chart of average price by year and scatter plot of price per sqft vs. size

#### Example Usage
```python
analysis = Analysis(cleaned_data)
df_with_price = analysis.calculate_price_per_sqft()
df_with_age = analysis.calculate_property_age()

viz = Visualization(df_with_age)
viz.plot_price_distribution()  # All property types
viz.plot_price_distribution(property_type='Apartment')  # Specific type
viz.plot_top_locations(top_n=5)
viz.plot_price_trends()
```

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
- **Insights**: Property type distribution, price analysis by location and type, price per sqft, property age, top locations
- **Visualizations**: Price distributions, boxplots, barplots, line charts, scatter plots, correlation matrices, market trends

