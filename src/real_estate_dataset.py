"""
Real Estate Dataset Class for Data Processing Pipeline

This module contains the RealEstateDataset class that handles data ingestion,
cleaning, and basic exploration of real estate data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class RealEstateDataset:
    """
    A class for handling real estate data processing including loading,
    cleaning, and analyzing real estate datasets.
    """
    
    def __init__(self):
        """Initialize the RealEstateDataset instance."""
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.filepath: Optional[str] = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Read the dataset and initialize a pandas DataFrame.
        
        Args:
            filepath (str): Path to the CSV file containing real estate data
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            self.filepath = filepath
            self.data = pd.read_csv(filepath)
            # Keep a copy of original data for comparison
            self.original_data = self.data.copy()
            
            print(f"✅ Data loaded successfully from {filepath}")
            print(f"📊 Dataset shape: {self.data.shape}")
            print("\n🔍 First 5 rows:")
            print(self.data.head())
            print("\n📋 Column names:")
            print(self.data.columns.tolist())
            print("\n🏷️  Data types:")
            print(self.data.dtypes)
            
            return self.data
            
        except FileNotFoundError:
            print(f"❌ Error: File '{filepath}' not found.")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Handle missing and invalid data.
        
        This method:
        - Fills missing numerical values with mean/median
        - Handles missing categorical values
        - Removes invalid entries (negative prices, etc.)
        - Ensures proper data types
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if self.data is None:
            raise ValueError("❌ No data loaded. Please call load_data() first.")
        
        print("🧹 Starting data cleaning process...")
        
        # Store initial state
        initial_rows = len(self.data)
        
        # 1. Handle missing values
        print("\n📊 Missing values before cleaning:")
        missing_before = self.data.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # Numerical columns - fill with median
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.data[col].isnull().sum() > 0:
                median_val = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_val)
                print(f"   ✓ Filled {col} missing values with median: {median_val:.2f}")
        
        # Categorical columns - fill with mode or drop
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                if self.data[col].isnull().sum() / len(self.data) > 0.5:
                    # If more than 50% missing, drop the column
                    self.data = self.data.drop(columns=[col])
                    print(f"   ✓ Dropped column '{col}' (>50% missing)")
                else:
                    # Fill with mode
                    mode_val = self.data[col].mode()[0] if not self.data[col].mode().empty else 'Unknown'
                    self.data[col] = self.data[col].fillna(mode_val)
                    print(f"   ✓ Filled {col} missing values with mode: '{mode_val}'")
        
        # 2. Handle invalid entries
        print("\n🔍 Checking for invalid entries...")
        
        # Remove negative prices
        if 'Price' in self.data.columns:
            negative_prices = self.data['Price'] < 0
            if negative_prices.sum() > 0:
                self.data = self.data[~negative_prices]
                print(f"   ✓ Removed {negative_prices.sum()} rows with negative prices")
        
        # Remove zero or negative bedrooms/bathrooms (if they exist)
        for col in ['Bedrooms', 'Bathrooms']:
            if col in self.data.columns:
                invalid_rooms = self.data[col] < 0
                if invalid_rooms.sum() > 0:
                    self.data = self.data[~invalid_rooms]
                    print(f"   ✓ Removed {invalid_rooms.sum()} rows with negative {col}")
        
        # Remove zero or negative size
        if 'Size_sqft' in self.data.columns:
            invalid_size = self.data['Size_sqft'] <= 0
            if invalid_size.sum() > 0:
                self.data = self.data[~invalid_size]
                print(f"   ✓ Removed {invalid_size.sum()} rows with invalid size")
        
        # 3. Fix data types
        print("\n🔧 Fixing data types...")
        
        # Convert date columns
        date_columns = ['Date_Added']
        for col in date_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                    print(f"   ✓ Converted {col} to datetime")
                except:
                    print(f"   ⚠️  Could not convert {col} to datetime")
        
        # Ensure binary columns are 0/1
        binary_columns = ['Sold', 'Garage', 'Pool']
        for col in binary_columns:
            if col in self.data.columns:
                unique_vals = self.data[col].unique()
                if not all(val in [0, 1, np.nan] for val in unique_vals):
                    # Convert Yes/No or True/False to 1/0
                    self.data[col] = self.data[col].map({
                        'Yes': 1, 'No': 0, 'True': 1, 'False': 0,
                        'Y': 1, 'N': 0, True: 1, False: 0
                    }).fillna(self.data[col])
                    print(f"   ✓ Standardized binary column: {col}")
        
        # 4. Remove duplicates
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            self.data = self.data.drop_duplicates()
            print(f"   ✓ Removed {duplicates} duplicate rows")
        
        # 5. Reset index
        self.data = self.data.reset_index(drop=True)
        
        # Summary
        final_rows = len(self.data)
        rows_removed = initial_rows - final_rows
        
        print(f"\n✅ Data cleaning completed!")
        print(f"📊 Initial rows: {initial_rows}")
        print(f"📊 Final rows: {final_rows}")
        print(f"📊 Rows removed: {rows_removed}")
        
        print("\n📊 Missing values after cleaning:")
        missing_after = self.data.isnull().sum()
        remaining_missing = missing_after[missing_after > 0]
        if len(remaining_missing) == 0:
            print("   ✅ No missing values remaining!")
        else:
            print(remaining_missing)
        
        return self.data
    
    def describe_data(self) -> Dict[str, Any]:
        """
        Print basic statistics and exploratory insights.
        
        Returns:
            Dict: Dictionary containing various statistics and insights
        """
        if self.data is None:
            raise ValueError("❌ No data loaded. Please call load_data() first.")
        
        print("📈 REAL ESTATE DATA ANALYSIS REPORT")
        print("=" * 50)
        
        # Basic info
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   • Total properties: {len(self.data):,}")
        print(f"   • Number of features: {len(self.data.columns)}")
        print(f"   • Data types: {self.data.dtypes.value_counts().to_dict()}")
        
        # Descriptive statistics for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n📊 NUMERICAL STATISTICS")
            print(self.data[numerical_cols].describe())
        
        insights = {}
        
        # Property type analysis
        if 'Type' in self.data.columns:
            print(f"\n🏠 PROPERTY TYPE DISTRIBUTION")
            type_counts = self.data['Type'].value_counts()
            type_percentages = self.data['Type'].value_counts(normalize=True) * 100
            
            for prop_type, count in type_counts.items():
                percentage = type_percentages[prop_type]
                print(f"   • {prop_type}: {count:,} ({percentage:.1f}%)")
            
            insights['property_types'] = type_counts.to_dict()
        
        # Price analysis by property type
        if 'Price' in self.data.columns and 'Type' in self.data.columns:
            print(f"\n💰 AVERAGE PRICES BY PROPERTY TYPE")
            avg_prices = self.data.groupby('Type')['Price'].agg(['mean', 'median', 'count'])
            
            for prop_type in avg_prices.index:
                mean_price = avg_prices.loc[prop_type, 'mean']
                median_price = avg_prices.loc[prop_type, 'median']
                count = avg_prices.loc[prop_type, 'count']
                print(f"   • {prop_type}:")
                print(f"     - Average: ${mean_price:,.0f}")
                print(f"     - Median: ${median_price:,.0f}")
                print(f"     - Count: {count:,}")
            
            insights['avg_prices_by_type'] = avg_prices['mean'].to_dict()
        
        # Size analysis by location
        if 'Size_sqft' in self.data.columns and 'Location' in self.data.columns:
            print(f"\n📏 AVERAGE SIZE BY LOCATION (Top 10)")
            avg_size = self.data.groupby('Location')['Size_sqft'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            for i, (location, row) in enumerate(avg_size.head(10).iterrows()):
                mean_size = row['mean']
                count = row['count']
                print(f"   {i+1:2}. {location}: {mean_size:,.0f} sqft (n={count:,})")
            
            insights['avg_size_by_location'] = avg_size['mean'].head(10).to_dict()
        
        # Sales analysis
        if 'Sold' in self.data.columns:
            print(f"\n🏷️  SALES STATUS")
            sold_counts = self.data['Sold'].value_counts()
            sold_rate = (sold_counts.get(1, 0) / len(self.data)) * 100
            print(f"   • Sold: {sold_counts.get(1, 0):,} ({sold_rate:.1f}%)")
            print(f"   • Not Sold: {sold_counts.get(0, 0):,} ({100-sold_rate:.1f}%)")
            
            insights['sold_rate'] = sold_rate
        
        # Days on market analysis
        if 'Days_on_Market' in self.data.columns:
            print(f"\n⏱️  DAYS ON MARKET")
            dom_stats = self.data['Days_on_Market'].describe()
            print(f"   • Average: {dom_stats['mean']:.1f} days")
            print(f"   • Median: {dom_stats['50%']:.1f} days")
            print(f"   • Max: {dom_stats['max']:.0f} days")
            
            insights['days_on_market'] = {
                'mean': dom_stats['mean'],
                'median': dom_stats['50%'],
                'max': dom_stats['max']
            }
        
        # Price range analysis
        if 'Price' in self.data.columns:
            print(f"\n💵 PRICE DISTRIBUTION")
            price_stats = self.data['Price'].describe()
            print(f"   • Average: ${price_stats['mean']:,.0f}")
            print(f"   • Median: ${price_stats['50%']:,.0f}")
            print(f"   • Min: ${price_stats['min']:,.0f}")
            print(f"   • Max: ${price_stats['max']:,.0f}")
            
            # Price ranges
            price_ranges = pd.cut(self.data['Price'], 
                                bins=[0, 200000, 400000, 600000, 800000, float('inf')],
                                labels=['<$200K', '$200K-$400K', '$400K-$600K', '$600K-$800K', '>$800K'])
            
            print(f"\n💰 PRICE RANGES")
            price_range_counts = price_ranges.value_counts().sort_index()
            for price_range, count in price_range_counts.items():
                percentage = (count / len(self.data)) * 100
                print(f"   • {price_range}: {count:,} ({percentage:.1f}%)")
            
            insights['price_stats'] = {
                'mean': price_stats['mean'],
                'median': price_stats['50%'],
                'min': price_stats['min'],
                'max': price_stats['max']
            }
        
        # Correlation analysis (for numerical columns)
        if len(numerical_cols) > 1:
            print(f"\n🔗 TOP CORRELATIONS WITH PRICE")
            if 'Price' in numerical_cols:
                correlations = self.data[numerical_cols].corr()['Price'].abs().sort_values(ascending=False)
                # Exclude Price itself and show top 5
                top_correlations = correlations.drop('Price').head(5)
                
                for feature, corr in top_correlations.items():
                    print(f"   • {feature}: {corr:.3f}")
                
                insights['price_correlations'] = top_correlations.to_dict()
        
        print(f"\n" + "=" * 50)
        print("✅ Data analysis completed!")
        
        return insights
    
    def save_cleaned_data(self, output_path: str) -> None:
        """
        Save the cleaned dataset to a CSV file.
        
        Args:
            output_path (str): Path where to save the cleaned data
        """
        if self.data is None:
            raise ValueError("❌ No data to save. Please load and clean data first.")
        
        try:
            self.data.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to: {output_path}")
            print(f"📊 Saved {len(self.data)} rows and {len(self.data.columns)} columns")
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current dataset.
        
        Returns:
            Dict: Summary statistics and information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "data_types": self.data.dtypes.to_dict()
        }
    
    def create_visualizations(self) -> None:
        """
        Create basic visualizations for the real estate data.
        """
        if self.data is None:
            raise ValueError("❌ No data loaded. Please call load_data() first.")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Real Estate Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        if 'Price' in self.data.columns:
            axes[0, 0].hist(self.data['Price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Price Distribution')
            axes[0, 0].set_xlabel('Price ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].ticklabel_format(style='plain', axis='x')
        
        # 2. Property type counts
        if 'Type' in self.data.columns:
            type_counts = self.data['Type'].value_counts()
            axes[0, 1].bar(type_counts.index, type_counts.values, color=['lightcoral', 'lightgreen', 'lightyellow'])
            axes[0, 1].set_title('Property Type Distribution')
            axes[0, 1].set_xlabel('Property Type')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Price vs Size scatter plot
        if 'Price' in self.data.columns and 'Size_sqft' in self.data.columns:
            axes[1, 0].scatter(self.data['Size_sqft'], self.data['Price'], alpha=0.6, color='purple')
            axes[1, 0].set_title('Price vs Size')
            axes[1, 0].set_xlabel('Size (sqft)')
            axes[1, 0].set_ylabel('Price ($)')
        
        # 4. Average price by location (top 10)
        if 'Price' in self.data.columns and 'Location' in self.data.columns:
            avg_price_by_location = self.data.groupby('Location')['Price'].mean().sort_values(ascending=False).head(10)
            axes[1, 1].bar(range(len(avg_price_by_location)), avg_price_by_location.values, color='orange')
            axes[1, 1].set_title('Average Price by Location (Top 10)')
            axes[1, 1].set_xlabel('Location')
            axes[1, 1].set_ylabel('Average Price ($)')
            axes[1, 1].set_xticks(range(len(avg_price_by_location)))
            axes[1, 1].set_xticklabels(avg_price_by_location.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Visualizations created successfully!")


# Example usage
if __name__ == "__main__":
    # Create an instance of the RealEstateDataset
    dataset = RealEstateDataset()
    
    # Example workflow (uncomment to run)
    # dataset.load_data('data/raw/housing_data.csv')
    # dataset.clean_data()
    # insights = dataset.describe_data()
    # dataset.save_cleaned_data('data/cleaned/housing_data_cleaned.csv')
    # dataset.create_visualizations()
    
    print("🏠 RealEstateDataset class ready for use!")
