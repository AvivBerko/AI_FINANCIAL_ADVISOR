"""
ETF Recommendation System - Main Pipeline
==========================================

This script runs the complete end-to-end pipeline:
1. Fetch ETF data (or use cached)
2. Generate synthetic investor data
3. Prepare training data
4. Train ML models
5. Evaluate and save results

Usage:
    python pipeline.py [--skip-fetch] [--skip-training]
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(__file__))

print("=" * 60)
print("ETF RECOMMENDATION SYSTEM - FULL PIPELINE")
print("=" * 60)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def step_1_fetch_etf_data(skip=True):
    """Fetch ETF data from APIs"""
    print("\n📥 STEP 1: Fetching ETF Data")
    print("-" * 60)
    
    if skip:
        print("⏭️  Skipped (using cached data)")
        return
    
    # Check if raw data exists
    raw_file = 'data/raw/combined_etf_data.csv'
    if os.path.exists(raw_file):
        print(f"✅ Using cached raw ETF data: {raw_file}")
        return
    
    print("⚠️  No cached data found. Running ETF fetcher...")
    print("   This will fetch data from yfinance and Finnhub APIs.")
    
    # Run the ETF fetcher script
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/etf_fetcher.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Raw ETF data fetched successfully")
    else:
        print(f"❌ Error fetching ETF data:")
        print(result.stderr)
        raise Exception("ETF fetcher failed")


def step_1_5_calculate_ratings():
    """Calculate Morningstar-style star ratings"""
    print("\n⭐ STEP 1.5: Calculating Morningstar Star Ratings")
    print("-" * 60)
    
    # Check if final file with ratings already exists
    final_file = 'data/raw/combined_etf_with_morning_star.csv'
    if os.path.exists(final_file):
        print(f"✅ Ratings already calculated: {final_file}")
        return
    
    # Load raw ETF data
    raw_file = 'data/raw/combined_etf_data.csv'
    if not os.path.exists(raw_file):
        print(f"❌ Raw ETF data not found: {raw_file}")
        print("   Please run Step 1 first.")
        raise FileNotFoundError(f"Missing {raw_file}")
    
    print(f"Loading raw ETF data from {raw_file}...")
    import pandas as pd
    from src.morning_star_rating import calculate_star_ratings
    
    df = pd.read_csv(raw_file)
    print(f"Loaded {len(df)} ETFs")
    
    # Calculate ratings
    print("Calculating Morningstar-style star ratings...")
    try:
        df_with_ratings = calculate_star_ratings(df)
        print(f"✅ Star ratings calculated for {len(df_with_ratings)} ETFs")
        
        # Save the final dataset
        df_with_ratings.to_csv(final_file, index=False)
        print(f"✅ Dataset with ratings saved to: {final_file}")
    except Exception as e:
        print(f"❌ Error calculating ratings: {e}")
        raise


def step_2_generate_investors():
    """Generate synthetic investor data"""
    print("\n👥 STEP 2: Generating Synthetic Investors")
    print("-" * 60)
    
    # Check if investor data already exists
    import glob
    existing_files = glob.glob('data/processed/synthetic_investor_data_*.csv')
    if existing_files:
        print(f"✅ Using existing investor data: {existing_files[0]}")
        return
    
    print("Generating 5,000 investors with balanced risk profiles...")
    
    # Run the data generation script
    import subprocess
    result = subprocess.run(
        [sys.executable, 'src/data_generation.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Investor data generated successfully")
    else:
        print(f"❌ Error generating investor data:")
        print(result.stderr)
        raise Exception("Investor data generation failed")


def step_3_prepare_training_data():
    """Prepare training data (run Phase 2 notebook logic)"""
    print("\n🔧 STEP 3: Preparing Training Data")
    print("-" * 60)
    
    # Check if training data already exists
    import glob
    training_files = glob.glob('data/training/X_train.csv')
    if training_files:
        print(f"✅ Training data already exists in data/training/")
        return
    
    print("⚠️  Training data not found. You need to run:")
    print("   notebooks/phase2_data_preparation.ipynb")
    print("\n   This notebook will:")
    print("   - Load ETF data and investor data")
    print("   - Create training/test splits")
    print("   - Save to data/training/")
    print("\n   You can run it with:")
    print("   jupyter nbconvert --execute --to notebook notebooks/phase2_data_preparation.ipynb")


def step_4_train_models(skip=False):
    """Train ML models (run Phase 3 notebook logic)"""
    print("\n🤖 STEP 4: Training ML Models")
    print("-" * 60)
    
    if skip:
        print("⏭️  Skipped (using existing models)")
        return
    
    # Check if models already exist
    import glob
    model_files = glob.glob('models/*.pkl')
    if model_files:
        print(f"✅ Models already exist in models/ ({len(model_files)} files)")
        return
    
    print("⚠️  Models not found. You need to run:")
    print("   notebooks/phase3_model_training.ipynb")
    print("\n   This notebook will:")
    print("   - Load training data from data/training/")
    print("   - Train multiple ML models")
    print("   - Save models to models/")
    print("   - Generate evaluation reports")
    print("\n   You can run it with:")
    print("   jupyter nbconvert --execute --to notebook notebooks/phase3_model_training.ipynb")


def step_5_summary():
    """Print summary"""
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    
    print("\n📁 Generated Files:") 
    print("   - data/raw/combined_etf_with_morning_star.csv")
    print("   - data/processed/synthetic_investor_data_*.csv")
    print("   - data/training/X_train.csv, y_train_*.csv")
    print("   - models/*.pkl")
    print("   - evaluation_reports/*.png, *.json")
    
    print("\n🎯 Next Steps:")
    print("   1. Run notebooks/phase2_data_preparation.ipynb to create training data")
    print("   2. Run notebooks/phase3_model_training.ipynb to train models")
    print("   3. Review evaluation_reports/model_metrics.json")
    print("   4. Check evaluation_reports/*.png for visualizations")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run ETF Recommendation System Pipeline')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip ETF data fetching')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    args = parser.parse_args()
    
    try:
        step_1_fetch_etf_data(skip=args.skip_fetch)
        step_1_5_calculate_ratings()
        step_2_generate_investors()
        step_3_prepare_training_data()
        step_4_train_models(skip=args.skip_training)
        step_5_summary()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
