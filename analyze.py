import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def read_csv_file(file_path=None):
    #Read a CSV file
    if file_path is None:
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            file_path = input("Enter the path to the CSV file: ").strip()
    
    # Validate 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    if not file_path.lower().endswith('.csv'):
        print(f"Warning: File '{file_path}' doesn't have a .csv extension")
    
    # Read file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV file: {file_path}")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file '{file_path}' is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file '{file_path}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file '{file_path}': {e}")


def load_all_csvs(base_path="archive"):
    #Load CSV files: stocks, companies, and index.

    stocks_path = os.path.join(base_path, "sp500_stocks.csv")
    companies_path = os.path.join(base_path, "sp500_companies.csv")
    index_path = os.path.join(base_path, "sp500_index.csv")
    
    print("Loading CSV files...")
    print("="*50)
    
    stocks_df = pd.read_csv(stocks_path)
    print(f"Loaded stocks data: {stocks_df.shape[0]} rows, {stocks_df.shape[1]} columns")
    
    companies_df = pd.read_csv(companies_path)
    print(f"Loaded companies data: {companies_df.shape[0]} rows, {companies_df.shape[1]} columns")
    
    index_df = pd.read_csv(index_path)
    print(f"Loaded index data: {index_df.shape[0]} rows, {index_df.shape[1]} columns")
    
    return stocks_df, companies_df, index_df


def clean_and_pivot_stock_data(stocks_df):
    #Clean stock data and pivot into price matrix P_{t,i}.
    
    print("\n" + "="*50)
    print("Cleaning and structuring stock data...")
    print("="*50)
    
    # Make a copy 
    df = stocks_df.copy()
    
    initial_rows = len(df)
    print(f"Initial rows: {initial_rows:,}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Converted Date column to datetime")
    
    # Drop rows where Adj Close is missing (NaN, empty, or empty string)
    # Check for NaN, None, and empty strings
    before_drop = len(df)
    
    # Replace empty strings with NaN
    df['Adj Close'] = df['Adj Close'].replace('', np.nan)
    # Drop rows with NaN in Adj Close
    df = df.dropna(subset=['Adj Close'])
    
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
    df = df.dropna(subset=['Adj Close'])
    
    rows_dropped = before_drop - len(df)
    print(f"Dropped {rows_dropped:,} rows with missing Adj Close values")
    print(f"Remaining rows: {len(df):,}")
    
    unique_symbols = df['Symbol'].nunique()
    unique_dates = df['Date'].nunique()
    print(f"Unique symbols: {unique_symbols}")
    print(f"Unique dates: {unique_dates}")
    
    # Pivot the data into a matrix
    # Rows (index) → Date
    # Columns → Symbol
    # Values → Adj Close
    print("\nPivoting data into price matrix...")
    price_matrix = df.pivot_table(
        index='Date',
        columns='Symbol',
        values='Adj Close',
        aggfunc='first'  # Use first value if duplicates exist
    )
    
    print(f"\nPrice matrix P_{{t,i}} created:")
    print(f"  Shape: {price_matrix.shape[0]} rows (dates) × {price_matrix.shape[1]} columns (symbols)")
    print(f"  Date range: {price_matrix.index.min()} to {price_matrix.index.max()}")
    print(f"  Symbols: {price_matrix.columns.tolist()[:10]}..." if len(price_matrix.columns) > 10 else f"  Symbols: {price_matrix.columns.tolist()}")
    
    return price_matrix

def convert_to_log_returns(price_matrix):
    #Convert price matrix to log returns.
    
    print("\n" + "="*50)
    print("Converting price matrix to log returns...")
    print("="*50)
    
    # Compute log returns
    
    log_returns = np.log(price_matrix).diff()
    
    # Drop first row
    rows_before = len(log_returns)
    log_returns = log_returns.iloc[1:].copy()
    rows_dropped = rows_before - len(log_returns)
    
    print(f"Computed log returns: r_{{t,i}} = log(P_{{t,i}} / P_{{t-1,i}})")
    print(f"Dropped {rows_dropped} row(s) with NaN values from differencing")
    print(f"Final shape: {log_returns.shape[0]} rows (dates) × {log_returns.shape[1]} columns (symbols)")
    
    return log_returns

def display_summary_of_log_returns(log_returns):

    print("\n" + "="*50)
    print("Log Returns Matrix R_{t,i} Summary:")
    print("="*50)
    print(f"\nFirst few rows and columns:")
    print(log_returns.iloc[:5, :5])
    print(f"\nData types:")
    print(log_returns.dtypes.value_counts())

    print(f"\nMissing values per column (first 10):")
    missing_counts = log_returns.isna().sum()
    print(missing_counts.head(10))
    print(f"\nTotal missing values: {log_returns.isna().sum().sum():,}")
    print(f"Percentage of missing values: {log_returns.isna().sum().sum() / (log_returns.shape[0] * log_returns.shape[1]) * 100:.2f}%")

    print("\n" + "="*50)
    print("Log Returns Matrix R_{t,i} is ready for further analysis!")

def get_mean_returns(log_returns):
    #Calculate mean returns
    
    print("\n" + "="*50)
    print("Calculating mean returns for each symbol...")
    print("="*50)
    
    # Calculate mean returns for each symbol
    mean_returns = log_returns.mean()
    
    print(f"\nSummary statistics:")
    print(f"  Number of symbols: {len(mean_returns)}")
    print(f"  Mean return (across all symbols): {mean_returns.mean():.6f}")
    print(f"  Median return: {mean_returns.median():.6f}")
    print(f"  Min return: {mean_returns.min():.6f} ({mean_returns.idxmin()})")
    print(f"  Max return: {mean_returns.max():.6f} ({mean_returns.idxmax()})")
    print(f"  Std dev of mean returns: {mean_returns.std():.6f}")
    
    print(f"\nTop 10 symbols by mean return:")
    top_10 = mean_returns.nlargest(10)
    for symbol, ret in top_10.items():
        print(f"  {symbol}: {ret:.6f}")
    
    print(f"\nBottom 10 symbols by mean return:")
    bottom_10 = mean_returns.nsmallest(10)
    for symbol, ret in bottom_10.items():
        print(f"  {symbol}: {ret:.6f}")

    return mean_returns

    


if __name__ == "__main__":

    stocks_df, companies_df, index_df = load_all_csvs()
    
    price_matrix = clean_and_pivot_stock_data(stocks_df)
    
    # Display summary of the price matrix
    print("\n" + "="*50)
    print("Price Matrix P_{t,i} Summary:")
    print("="*50)
    print(f"\nFirst few rows and columns:")
    print(price_matrix.iloc[:5, :5])
    print(f"\nData types:")
    print(price_matrix.dtypes.value_counts())
    print(f"\nMissing values per column (first 10):")
    missing_counts = price_matrix.isna().sum()
    print(missing_counts.head(10))
    print(f"\nTotal missing values: {price_matrix.isna().sum().sum():,}")
    print(f"Percentage of missing values: {price_matrix.isna().sum().sum() / (price_matrix.shape[0] * price_matrix.shape[1]) * 100:.2f}%")
    

    print("\n" + "="*50)
    print("Price matrix P_{t,i} is ready for further analysis!")
    print("="*50)
    
    log_returns = convert_to_log_returns(price_matrix)
    
    # Display summary of log returns matrix
    display_summary_of_log_returns(log_returns)
    
    print("\n" + "="*50)
    print("Data Processing Complete!")
    print("="*50)
    print(f"✓ Price matrix P_{{t,i}}: {price_matrix.shape[0]} dates × {price_matrix.shape[1]} symbols")
    print(f"✓ Log returns matrix R_{{t,i}}: {log_returns.shape[0]} dates × {log_returns.shape[1]} symbols")
    print(f"✓ R ∈ R^{{{log_returns.shape[0]}×{log_returns.shape[1]}}} - ready for quantitative analysis!")
    print("="*50)
    get_mean_returns(log_returns)
    print("\n" + "="*50)
    print("Mean returns for each symbol are ready for further analysis!")
    print("="*50)

