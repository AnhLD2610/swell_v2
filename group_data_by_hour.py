import pandas as pd
import numpy as np

def group_data_by_hour(input_file, output_file):
    """
    Group arbitrage data by hour and calculate mean values
    """
    print(f"📊 Loading data from: {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"📋 Original data: {len(df)} rows")
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create hour column (remove minutes and seconds)
    df['hour'] = df['datetime'].dt.floor('H')  # Floor to hour
    
    print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"🕐 Hour range: {df['hour'].min()} to {df['hour'].max()}")
    
    # Group by hour and calculate mean
    print("🔄 Grouping data by hour...")
    
    # Select numeric columns to aggregate
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Group by hour and calculate mean
    hourly_data = df.groupby('hour').agg({
        **{col: 'mean' for col in numeric_columns},
        'datetime': 'first'  # Keep first datetime for reference
    }).reset_index()
    
    # Rename datetime column to represent the hour
    hourly_data['datetime'] = hourly_data['hour']
    hourly_data = hourly_data.drop('hour', axis=1)
    
    print(f"✅ Grouped data: {len(hourly_data)} rows (hours)")
    
    # Show sample of original vs grouped data
    print("\n📈 Sample comparison:")
    print("Original data (first 5 rows):")
    print(df[['datetime', 'arb_profit', 'price_HYPE_HyperEVM', 'Amount_HYPE_HyperEVM']].head())
    
    print("\nGrouped data (first 5 rows):")
    print(hourly_data[['datetime', 'arb_profit', 'price_HYPE_HyperEVM', 'Amount_HYPE_HyperEVM']].head())
    
    # Save to new file
    print(f"\n💾 Saving hourly data to: {output_file}")
    hourly_data.to_csv(output_file, index=False)
    
    # Show statistics
    print("\n📊 Data Summary:")
    print(f"Original rows: {len(df)}")
    print(f"Hourly rows: {len(hourly_data)}")
    print(f"Compression ratio: {len(df) / len(hourly_data):.1f}x")
    
    # Show target statistics
    print(f"\n🎯 Target (arb_profit) statistics:")
    print(f"Original - Min: {df['arb_profit'].min():.4f}, Max: {df['arb_profit'].max():.4f}, Mean: {df['arb_profit'].mean():.4f}")
    print(f"Hourly   - Min: {hourly_data['arb_profit'].min():.4f}, Max: {hourly_data['arb_profit'].max():.4f}, Mean: {hourly_data['arb_profit'].mean():.4f}")
    
    return hourly_data

def main():
    input_file = '2025_03_22_to_2025_05_25_ARB_EVM_CORE_DATA.csv'
    output_file = 'final_data_no_clean.csv'
    
    print("🚀 Grouping Arbitrage Data by Hour")
    print("=" * 50)
    
    # Process data
    hourly_data = group_data_by_hour(input_file, output_file)
    
    print("\n✅ Process completed successfully!")
    print(f"📄 New hourly data saved as: {output_file}")

if __name__ == "__main__":
    main() 