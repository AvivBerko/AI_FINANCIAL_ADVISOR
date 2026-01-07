import yfinance as yf
import finnhub
import pandas as pd
import numpy as np
import time
import datetime
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


pd.set_option('display.max_columns', None)

class FinnhubETFDataHandler:
    def __init__(self, finnhub_api_key):
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)

    def get_finnhub_daily_prices(self, symbol):
        try:
            quote = self.finnhub_client.quote(symbol)

            data = {
                "Symbol": symbol,
                "Current Price": quote.get("c"),
                "Previous Close": quote.get("pc"),
                "Day High": quote.get("h"),
                "Day Low": quote.get("l"),
                "Open": quote.get("o")
            }

            # Avoid rate-limiting
            time.sleep(1)

            return data

        except Exception as e:
            print(f"Error retrieving data for {symbol}: {e}")
            return None


class YFinanceETFDataHandler:

    def __init__(self):
        pass

    # Helper methods
    def period_return(self, series, years):
        if series.empty:
            return None
        try:
            days = years * 252
            if len(series) < days:
                return None
            return (series.iloc[-1] / series.iloc[-days]) - 1
        except Exception:
            return None

    def max_drawdown(self, series):
        if series.empty:
            return None
        roll_max = series.cummax()
        drawdown = (series / roll_max) - 1.0
        return drawdown.min()

    def annual_volatility(self, returns):
        if returns.empty:
            return None
        return returns.std() * np.sqrt(252)

    # Main method to fetch ETF data
    def fetch_etf_data(self, ticker_symbol):
        etf = yf.Ticker(ticker_symbol)
        info = etf.info

        hist = etf.history(period="10y")
        hist['daily_return'] = hist['Close'].pct_change()

        volatility_annual = self.annual_volatility(hist['daily_return'].dropna())
        mdd = self.max_drawdown(hist['Close'])

        # Get geographic allocation
        geo_allocation = self.fetch_geographic_allocation(etf)

        data = {
            "Fund Symbol": ticker_symbol,
            "Fund Name": info.get("longName", None),
            "Fund Manager": info.get("fundFamily", None),
            "Fifty Two Week High": info.get("fiftyTwoWeekHigh", None),
            "Fifty Two Week Low": info.get("fiftyTwoWeekLow", None),
            "Volume": info.get("volume", None),
            "Average Trading Volume": info.get("averageVolume", None),
            "Market Cap": info.get("marketCap", None),
            "Category": info.get("category", None),
            "Assets Under Management (AUM)": info.get("totalAssets", None),
            "Dividend Yield": info.get("dividendYield", None),
            "1 Year Return": self.period_return(hist['Close'], 1),
            "3 Year Return": self.period_return(hist['Close'], 3),
            "5 Year Return": self.period_return(hist['Close'], 5),
            "Volatility (Annual STD)": volatility_annual,
            "Maximum Drawdown": mdd,
            "Price to Earnings (PE)": info.get("trailingPE", None),
            "Price to Book (PB)": info.get("priceToBook", None),
            "Summary": info.get("longBusinessSummary", None)
        }

        data.update(geo_allocation)
        return data

    def fetch_fund_operations(self, ticker_symbol):
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.funds_data

        try:
            expense_ratio = data.fund_operations.loc['Annual Report Expense Ratio', ticker_symbol]
            category_avg_expense_ratio = data.fund_operations.loc['Annual Report Expense Ratio', 'Category Average']

            def flatten_prefix(base_key, dct):
                return {f"{base_key}_{k}": v for k, v in (dct or {}).items()}

            return {
                "expense_ratio": float(expense_ratio),
                "category_avg_expense_ratio": float(category_avg_expense_ratio),
                **flatten_prefix("asset_classes", data.asset_classes),
                **flatten_prefix("sector_weightings", data.sector_weightings)
            }
        except Exception as e:
            print(f"Error fetching fund operations for {ticker_symbol}: {e}")
            return None

    def fetch_geographic_allocation(self, ticker):
        """
        Retrieves geographic allocations if available.
        Returns a dict with country/region and their percentage holdings.
        """
        geo_data = {}
        try:
            geo_holdings = ticker.geo_holdings
            if geo_holdings is not None and len(geo_holdings) > 0:
                for g in geo_holdings:
                    country_name = g['country'].replace(' ', '_').lower()
                    geo_data[f'geo_{country_name}'] = g['percentage']
        except Exception:
            pass
        return geo_data

class ETFDataAggregator:
    def __init__(self, finnhub_api_key):
        self.finnhub_handler = FinnhubETFDataHandler(finnhub_api_key)
        self.yfinance_handler = YFinanceETFDataHandler()

    def fetch_complete_etf_data(self, ticker):
        yf_data = self.yfinance_handler.fetch_etf_data(ticker)
        fund_ops = self.yfinance_handler.fetch_fund_operations(ticker)
        fh_data = self.finnhub_handler.get_finnhub_daily_prices(ticker)

        complete_data = yf_data or {}
        if fund_ops:
            complete_data.update(fund_ops)
        if fh_data:
            complete_data.update(fh_data)

        return complete_data



def main():
    print("Running main function")

    # Load API key and verify it exists
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_api_key:
        print(f"✓ Finnhub API key loaded (length: {len(finnhub_api_key)})")
    else:
        print("✗ ERROR: FINNHUB_API_KEY not found in environment variables")
        print("Please ensure your .env file contains: FINNHUB_API_KEY=your_key_here")
        return
    
    aggregator = ETFDataAggregator(finnhub_api_key=finnhub_api_key)
    etf_tickers = [
    "VOO",
    "IVV",
    "SPY",
    "VTI",
    "QQQ",
    "VTV",
    "BND",
    "AGG",
    "IWF",
    "VXUS",
    "VGT",
    "VWO",
    "VIG",
    "IJH",
    "SPYM",
    "XLK",
    "VO",
    "IJR",
    "ITOT",
    "RSP",
    "BNDX",
    "IBIT",
    "SCHD",
    "QQQM",
    "IWM",
    "VB",
    "EFA",
    "VYM",
    "IWD",
    "IVW",
    "IAU",
    "SGOV",
    "SCHX",
    "VCIT",
    "VT",
    "SCHF",
    "VEU",
    "SCHG",
    "XLF",
    "IXUS",
    "TLT",
    "QUAL",
    "IVE",
    "VV",
    "IWR",
    "IWB",
    "SPYG",
    "IEF",
    "BIL",
    "VTEB",
    "BSV",
    "MUB",
    "JEPI",
    "DIA",
    "XLV",
    "VCSH",
    "DFAC",
    "MBB",
    "SCHB",
    "SMH",
    "VGIT",
    "DGRO",
    "VONG",
    "JPST",
    "VNQ",
    "IUSB",
    "LQD",
    "GOVT",
    "MGK",
    "SPDW",
    "VBR",
    "JEPQ",
    "TQQQ",
    "SPYV",
    "SLV",
    "DYNF",
    "OEF",
    "VGK",
    "XLE",
    "EFV",
    "BIV",
    "XLC",
    "IUSG",
    "CGDV",
    "VGSH",
    "USHY",
    "JAAA",
    "VXF",
    "XLI",
    "ACWI",
    "IUSV",
    "GLDM",
    "GDX",
    "SHY",
    "IDEV",
]



    combined_data_list = []
    for ticker in etf_tickers:
        print(f"Fetching combined data for {ticker}...")
        data = aggregator.fetch_complete_etf_data(ticker)
        if data:
            combined_data_list.append(data)
        else:
            print(f"Warning: No data for {ticker}")

    df = pd.DataFrame(combined_data_list)
    print(f"\n📊 Fetched data for {len(df)} ETFs")
    
    # Save raw ETF data
    output_path = Path(__file__).parent.parent / 'data' / 'raw' / 'combined_etf_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Raw ETF dataset exported to:")
    print(f"   {output_path}")
    
    return output_path


if __name__ == "__main__":
    main()
