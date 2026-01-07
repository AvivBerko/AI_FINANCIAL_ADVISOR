import pandas as pd
import numpy as np


def calculate_star_ratings(df):
    """
    Takes a raw ETF DataFrame, adds Star Ratings, and returns the modified DataFrame.
    ETFs without sufficient data will have NaN ratings but won't be dropped.
    """
    print("... Calculating Morningstar-style ratings...")

    # Filter ETFs with needed data
    df_filtered = df.dropna(
        subset=['1 Year Return', '3 Year Return', '5 Year Return', 'Volatility (Annual STD)', 'Category']).copy()
    
    print(f"   {len(df_filtered)} out of {len(df)} ETFs have sufficient data for rating")

    # Calculate Sharpe-like ratios
    df_filtered['Sharpe_1yr'] = df_filtered['1 Year Return'] / df_filtered['Volatility (Annual STD)']
    df_filtered['Sharpe_3yr'] = df_filtered['3 Year Return'] / df_filtered['Volatility (Annual STD)']
    df_filtered['Sharpe_5yr'] = df_filtered['5 Year Return'] / df_filtered['Volatility (Annual STD)']

    # Weighted composite score
    df_filtered['composite_score'] = (0.2 * df_filtered['Sharpe_1yr'] +
                                      0.3 * df_filtered['Sharpe_3yr'] +
                                      0.5 * df_filtered['Sharpe_5yr'])

    # Assign stars by percentile within Category
    def assign_stars(group):
        group = group.copy()
        group['rank'] = group['composite_score'].rank(pct=True)
        conditions = [
            group['rank'] >= 0.9,
            (group['rank'] >= 0.675) & (group['rank'] < 0.9),
            (group['rank'] >= 0.325) & (group['rank'] < 0.675),
            (group['rank'] >= 0.1) & (group['rank'] < 0.325),
            group['rank'] < 0.1,
        ]
        choices = [5, 4, 3, 2, 1]
        group['custom_star_rating'] = np.select(conditions, choices)
        return group

    df_rated = df_filtered.groupby('Category', group_keys=False).apply(assign_stars)

    # Merge back to original data to preserve all ETFs
    # Keep only the rating columns from df_rated
    rating_cols = ['composite_score','custom_star_rating']
    df_final = df.merge(
        df_rated[['Fund Symbol'] + rating_cols],
        on='Fund Symbol',
        how='left'
    )
    
    print(f"   ✓ Ratings merged back to all {len(df_final)} ETFs")
    return df_final


if __name__ == "__main__":
    """
    Standalone mode: Read raw ETF data, calculate ratings, and save to CSV.
    Usage: python src/morning_star_rating.py
    """
    import os
    from pathlib import Path
    
    print("=" * 60)
    print("MORNINGSTAR-STYLE STAR RATING CALCULATOR")
    print("=" * 60)
    
    # Define file paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / 'data' / 'raw' / 'combined_etf_data.csv'
    output_file = project_root / 'data' / 'raw' / 'combined_etf_with_morning_star.csv'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"\nError: Input file not found: {input_file}")
        print("   Please run the ETF fetcher first:")
        print("   python src/etf_fetcher.py")
        exit(1)
    
    # Load data
    print(f"\nLoading ETF data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} ETFs")
    
    # Calculate ratings
    print("\nCalculating Morningstar-style star ratings...")
    df_with_ratings = calculate_star_ratings(df)
    
    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_with_ratings.to_csv(output_file, index=False)
    
    print(f"\nSUCCESS!")
    print(f"   Output saved to: {output_file}")
    print(f"   Total ETFs: {len(df_with_ratings)}")
    
    # Show rating distribution
    if 'custom_star_rating' in df_with_ratings.columns:
        rating_counts = df_with_ratings['custom_star_rating'].value_counts().sort_index()
        print(f"\nStar Rating Distribution:")
        for stars, count in rating_counts.items():
            if pd.notna(stars):
                print(f"   {'*' * int(stars)} ({int(stars)} stars): {count} ETFs")
        
        na_count = df_with_ratings['custom_star_rating'].isna().sum()
        if na_count > 0:
            print(f"   No rating (insufficient data): {na_count} ETFs")
    
    print("\n" + "=" * 60)