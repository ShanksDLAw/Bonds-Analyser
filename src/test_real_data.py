import logging
from data.country_fetcher import CountryDataManager
import json
from typing import Dict
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def format_metrics(metrics: Dict) -> pd.DataFrame:
    """Format metrics into a readable DataFrame"""
    rows = []
    
    # Process each category
    for category, data in metrics.items():
        for metric, value in data.items():
            rows.append({
                'Category': category.capitalize(),
                'Metric': metric.replace('_', ' ').title(),
                'Value': value
            })
    
    # Create DataFrame and sort
    df = pd.DataFrame(rows)
    df = df.sort_values(['Category', 'Metric'])
    
    # Format values
    def format_value(x):
        if isinstance(x, (int, float)):
            if abs(x) > 1000:
                return f"{x:,.0f}"
            return f"{x:.2f}"
        return str(x)
    
    df['Value'] = df['Value'].apply(format_value)
    return df

def main():
    """Test real data fetching for multiple countries"""
    countries = [
        'USA',  # United States
        'GBR',  # United Kingdom
        'DEU',  # Germany
        'JPN',  # Japan
        'CHN',  # China
        'BRA',  # Brazil
        'IND',  # India
        'ZAF'   # South Africa
    ]
    
    manager = CountryDataManager()
    
    print("\nFetching data for multiple countries...\n")
    
    for country_code in countries:
        try:
            print(f"\n{'='*80}")
            print(f"Data for {country_code}")
            print(f"{'='*80}")
            
            metrics = manager.get_comprehensive_metrics(country_code)
            
            if not any(metrics.values()):
                print(f"No data available for {country_code}")
                continue
            
            # Format and display metrics
            df = format_metrics(metrics)
            print("\n", df.to_string(index=False), "\n")
            
        except Exception as e:
            logger.error(f"Error processing {country_code}: {e}")
            continue

if __name__ == "__main__":
    main() 