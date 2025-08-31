#!/usr/bin/env python3
"""
Real Estate Data Processing Pipeline - Demo Script

This script demonstrates the usage of the RealEstateDataset class
for processing real estate data.

Author: Real Estate Data Processing Team
Date: August 2025
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_estate_dataset import RealEstateDataset


def main():
    """Main function to demonstrate the real estate data processing pipeline."""
    print("🏠 Real Estate Data Processing Pipeline")
    print("=" * 50)
    
    # Create an instance of RealEstateDataset
    dataset = RealEstateDataset()
    
    # Define file paths
    data_path = os.path.join('data', 'raw', 'housing_data.csv')
    output_path = os.path.join('data', 'cleaned', 'housing_data_cleaned.csv')
    
    try:
        # Step 1: Load data
        print("\n🔄 Step 1: Loading data...")
        dataset.load_data(data_path)
        
        # Step 2: Clean data
        print("\n🔄 Step 2: Cleaning data...")
        dataset.clean_data()
        
        # Step 3: Generate insights
        print("\n🔄 Step 3: Generating insights...")
        insights = dataset.describe_data()
        
        # Step 4: Save cleaned data
        print("\n🔄 Step 4: Saving cleaned data...")
        dataset.save_cleaned_data(output_path)
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Processed {len(dataset.data)} properties")
        print(f"💾 Cleaned data saved to: {output_path}")
        
        return dataset, insights
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None, None


if __name__ == "__main__":
    dataset, insights = main()
    
    if dataset is not None:
        print("\n🎉 Processing completed successfully!")
        print("\nTo use this pipeline in your projects:")
        print("1. Import the RealEstateDataset class")
        print("2. Create an instance: dataset = RealEstateDataset()")
        print("3. Load your data: dataset.load_data('your_file.csv')")
        print("4. Clean the data: dataset.clean_data()")
        print("5. Analyze the data: dataset.describe_data()")
        print("6. Save results: dataset.save_cleaned_data('cleaned_file.csv')")
    else:
        print("\n❌ Processing failed. Please check your data file and try again.")
