#!/usr/bin/env python3
"""
Data Explorer Script for WeatherMLOps
Standalone script to analyze reference and production data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """Load reference and production data"""
    ref_path = "data/reference_data.csv"
    prod_path = "data/production_data.csv"
    
    ref_data = None
    prod_data = None
    
    if os.path.exists(ref_path):
        ref_data = pd.read_csv(ref_path)
        print(f"✅ Reference data loaded: {len(ref_data)} records")
    else:
        print("❌ Reference data not found")
    
    if os.path.exists(prod_path):
        prod_data = pd.read_csv(prod_path)
        print(f"✅ Production data loaded: {len(prod_data)} records")
    else:
        print("❌ Production data not found")
    
    return ref_data, prod_data

def analyze_data_quality(df, name):
    """Analyze data quality for a dataset"""
    print(f"\n🔍 {name} Data Quality Report")
    print("=" * 50)
    
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    print(f"Unique Locations: {df[['lat', 'lon']].drop_duplicates().shape[0]}")
    print(f"Date Range: Month {df['month'].min()} - {df['month'].max()}")
    
    print("\nColumn Information:")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(col_info.to_string(index=False))
    
    print(f"\n{name} Statistical Summary:")
    print(df.describe())

def compare_datasets(ref_df, prod_df):
    """Compare reference and production datasets"""
    print("\n📊 Dataset Comparison")
    print("=" * 50)
    
    # Basic statistics comparison
    print(f"Reference Data Points: {len(ref_df)}")
    print(f"Production Data Points: {len(prod_df)}")
    print(f"Reference Avg Temp: {ref_df['temperature'].mean():.2f}°C")
    print(f"Production Avg Temp: {prod_df['temperature'].mean():.2f}°C")
    print(f"Reference Temp Std: {ref_df['temperature'].std():.2f}°C")
    print(f"Production Temp Std: {prod_df['temperature'].std():.2f}°C")
    
    # Statistical comparison
    print("\nStatistical Comparison:")
    t_stat, p_value = stats.ttest_ind(ref_df['temperature'], prod_df['temperature'])
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Significance: Significant Difference ⚠️")
    else:
        print("Significance: No Significant Difference ✅")
    
    # Temperature difference
    temp_diff = abs(ref_df['temperature'].mean() - prod_df['temperature'].mean())
    print(f"Temperature Difference: {temp_diff:.2f}°C")
    
    if temp_diff > 5.0:
        print("🔴 Large temperature difference between datasets")
    elif temp_diff > 2.0:
        print("🟡 Moderate temperature difference between datasets")
    else:
        print("✅ Small temperature difference between datasets")

def generate_plots(ref_df, prod_df):
    """Generate comparison plots"""
    print("\n📈 Generating comparison plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('WeatherMLOps Data Analysis', fontsize=16, fontweight='bold')
    
    # Temperature distribution comparison
    axes[0, 0].hist(ref_df['temperature'], alpha=0.7, bins=30, label='Reference Data', color='#667eea')
    axes[0, 0].hist(prod_df['temperature'], alpha=0.7, bins=30, label='Production Data', color='#764ba2')
    axes[0, 0].set_title('Temperature Distribution Comparison')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature by hour
    hourly_ref = ref_df.groupby('hour')['temperature'].mean()
    hourly_prod = prod_df.groupby('hour')['temperature'].mean()
    axes[0, 1].plot(hourly_ref.index, hourly_ref.values, 'o-', label='Reference Data', color='#667eea', linewidth=2)
    axes[0, 1].plot(hourly_prod.index, hourly_prod.values, 's-', label='Production Data', color='#764ba2', linewidth=2)
    axes[0, 1].set_title('Average Temperature by Hour')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature by month
    monthly_ref = ref_df.groupby('month')['temperature'].mean()
    monthly_prod = prod_df.groupby('month')['temperature'].mean()
    axes[1, 0].bar(monthly_ref.index - 0.2, monthly_ref.values, 0.4, label='Reference Data', color='#667eea', alpha=0.8)
    axes[1, 0].bar(monthly_prod.index + 0.2, monthly_prod.values, 0.4, label='Production Data', color='#764ba2', alpha=0.8)
    axes[1, 0].set_title('Average Temperature by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    temp_data = [ref_df['temperature'], prod_df['temperature']]
    axes[1, 1].boxplot(temp_data, labels=['Reference', 'Production'], patch_artist=True)
    axes[1, 1].set_title('Temperature Distribution Box Plot')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Color the boxes
    colors = ['#667eea', '#764ba2']
    for patch, color in zip(axes[1, 1].artists, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "data_analysis_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Analysis plot saved as: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function"""
    print("🌤️ WeatherMLOps Data Explorer")
    print("=" * 50)
    
    # Load data
    ref_data, prod_data = load_data()
    
    if ref_data is None and prod_data is None:
        print("❌ No data files found. Please ensure data files exist in the data/ directory.")
        return
    
    # Analyze individual datasets
    if ref_data is not None:
        analyze_data_quality(ref_data, "Reference")
    
    if prod_data is not None:
        analyze_data_quality(prod_data, "Production")
    
    # Compare datasets if both are available
    if ref_data is not None and prod_data is not None:
        compare_datasets(ref_data, prod_data)
        
        # Generate plots
        try:
            generate_plots(ref_data, prod_data)
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")
            print("This might be due to missing matplotlib or running in a headless environment.")
    
    print("\n✅ Data analysis complete!")
    print("\n💡 Recommendations:")
    print("- Check for significant temperature differences between datasets")
    print("- Monitor for data drift using the drift detection features")
    print("- Ensure data quality by checking for missing values and duplicates")
    print("- Use the Streamlit app for interactive data exploration")

if __name__ == "__main__":
    main() 