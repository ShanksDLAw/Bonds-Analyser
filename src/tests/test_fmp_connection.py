import os
from dotenv import load_dotenv
import requests
import sys
import json
from typing import Dict, List, Tuple

def test_endpoint(base_url: str, endpoint: str, api_key: str) -> Tuple[bool, str]:
    """Test a specific FMP API endpoint"""
    url = f"{base_url}/{endpoint}?apikey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True, "Success"
        elif response.status_code == 403:
            return False, "Access Denied (403)"
        else:
            return False, f"Error: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_fmp_connection():
    """Test FMP API connection and endpoint availability"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        print("❌ Error: FMP_API_KEY not found in .env file")
        sys.exit(1)
        
    # Base URL
    base_url = "https://financialmodelingprep.com/api/v3"
    
    # List of endpoints to test
    endpoints = {
        "Stock List": "stock/list",
        "Company Search": "search?query=AAPL&limit=1",
        "Company Profile": "profile/AAPL",
        "Company Quote": "quote/AAPL",
        "Key Metrics": "key-metrics/AAPL?limit=1",
        "Key Metrics TTM": "key-metrics-ttm/AAPL",
        "Balance Sheet": "balance-sheet-statement/AAPL?period=quarter&limit=1",
        "Ratios TTM": "ratios-ttm/AAPL",
        "Treasury Rates": "treasury",
    }
    
    print("\n🔍 Testing FMP API endpoints...")
    print("=" * 50)
    
    results = {}
    available_endpoints = []
    
    for name, endpoint in endpoints.items():
        success, message = test_endpoint(base_url, endpoint, api_key)
        results[name] = {"success": success, "message": message}
        if success:
            available_endpoints.append(name)
        status = "✅" if success else "❌"
        print(f"{status} {name}: {message}")
    
    print("\n📊 Summary")
    print("=" * 50)
    print(f"Total endpoints tested: {len(endpoints)}")
    print(f"Available endpoints: {len(available_endpoints)}")
    print(f"Restricted endpoints: {len(endpoints) - len(available_endpoints)}")
    
    if available_endpoints:
        print("\n✅ Available endpoints:")
        for endpoint in available_endpoints:
            print(f"- {endpoint}")
    
    # Save results to a file
    with open('fmp_api_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n💾 Test results saved to fmp_api_test_results.json")
    
    return results

if __name__ == "__main__":
    test_fmp_connection() 