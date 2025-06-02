# src/tests/test_fetcher.py
import pytest
import os
import sys
from src.data.country_fetcher import CountryDataManager, IMFClient, WorldBankClient, TreasuryClient
from unittest.mock import patch, Mock

# Add the project root to the Python path when running the file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)

from src.config import Config
from src.tests.fetcher import BondRiskFetcher

def test_fetcher_initialization():
    """Test that the BondRiskFetcher initializes correctly with API keys"""
    fetcher = BondRiskFetcher()
    assert fetcher.fmp_key is not None, "FMP API key is missing"
    assert fetcher.fred_key is not None, "FRED API key is missing"
    assert fetcher.cache is not None, "Cache manager was not initialized"

def test_company_search():
    """Test that the company search function works correctly"""
    fetcher = BondRiskFetcher()
    symbol = fetcher._search_company_symbol("Apple")
    assert symbol == "AAPL", f"Expected AAPL but got {symbol}"
    
    # Test with a non-existent company
    symbol = fetcher._search_company_symbol("NonExistentCompany12345")
    assert symbol is None, "Should return None for non-existent companies"

@pytest.mark.skip(reason="Requires API calls, should be run manually")
def test_get_company_data():
    """Test the full data fetching workflow (skipped by default to avoid API calls)"""
    fetcher = BondRiskFetcher()
    data = fetcher.get_company_data("Apple")
    assert data.stock_symbol == "AAPL"
    assert data.company_name == "Apple Inc."
    assert data.market_cap > 0
    assert data.risk_free_rate > 0

def test_country_manager_initialization():
    manager = CountryDataManager()
    assert isinstance(manager.imf, IMFClient)
    assert isinstance(manager.world_bank, WorldBankClient)
    assert isinstance(manager.treasury, TreasuryClient)

@pytest.fixture
def country_manager():
    """Initialize CountryDataManager instance"""
    return CountryDataManager()

def test_imf_client():
    """Test IMF client initialization and data fetching"""
    client = IMFClient()
    data = client.get_country_data('USA')
    assert isinstance(data, dict)
    assert len(data) > 0

def test_world_bank_client():
    """Test World Bank client initialization and data fetching"""
    client = WorldBankClient()
    data = client.get_all_indicators('USA')
    assert isinstance(data, dict)
    assert len(data) > 0

def test_treasury_client():
    """Test Treasury client initialization and data fetching"""
    client = TreasuryClient()
    rates = client.get_interest_rates()
    assert isinstance(rates, dict)
    assert len(rates) > 0

@patch('src.data.country_fetcher.requests.get')
def test_country_data_manager(mock_get, country_manager, mock_country_data):
    """Test comprehensive country data fetching"""
    # Mock API responses
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_country_data
    mock_get.return_value = mock_response

    metrics = country_manager.get_comprehensive_metrics("USA")
    
    assert "macro" in metrics
    assert "market" in metrics
    assert "risk" in metrics
    
    # Verify macro indicators
    assert metrics["macro"]["gdp_growth"] == mock_country_data["macro"]["gdp_growth"]
    assert metrics["macro"]["inflation"] == mock_country_data["macro"]["inflation"]
    
    # Verify market data
    assert metrics["market"]["bond_10y"] == mock_country_data["market"]["bond_10y"]
    assert metrics["market"]["policy_rate"] == mock_country_data["market"]["policy_rate"]
    
    # Verify risk metrics
    assert metrics["risk"]["market_volatility"] == mock_country_data["risk"]["market_volatility"]
    assert metrics["risk"]["sovereign_spread"] == mock_country_data["risk"]["sovereign_spread"]

def test_error_handling(country_manager):
    """Test error handling in data fetching"""
    # Test with invalid country code
    metrics = country_manager.get_comprehensive_metrics("INVALID")
    
    assert "macro" in metrics
    assert "market" in metrics
    assert "risk" in metrics
    
    # Verify default values are used
    assert metrics["macro"]["gdp_growth"] == 0.0
    assert metrics["macro"]["inflation"] == 0.0
    assert metrics["market"]["bond_10y"] == 3.5
    assert metrics["risk"]["market_volatility"] == 0.15

if __name__ == "__main__":
    # Run the tests when the file is executed directly
    print("Running tests for BondRiskFetcher...")
    test_fetcher_initialization()
    print("✓ Fetcher initialization test passed")
    
    test_company_search()
    print("✓ Company search test passed")
    
    # Uncomment to run the API test (will make actual API calls)
    # test_get_company_data()
    # print("✓ Get company data test passed")
    
    print("All tests passed successfully!")