import pytest
from typing import Dict
from datetime import datetime

@pytest.fixture
def mock_country_data() -> Dict:
    """Mock country data for testing"""
    return {
        "macro": {
            "gdp_growth": 2.5,
            "inflation": 3.0,
            "debt_to_gdp": 60.0,
            "current_account": -2.0
        },
        "market": {
            "interest_rates": {
                "10-Year": 3.5,
                "5-Year": 3.0,
                "2-Year": 2.5,
                "1-Year": 2.0
            },
            "equity_index": 15000,
            "bond_10y": 3.5,
            "policy_rate": 4.0
        },
        "risk": {
            "fx_volatility": 0.1,
            "market_volatility": 0.15,
            "sovereign_spread": 0.02
        }
    }

@pytest.fixture
def mock_bond_data() -> Dict:
    """Mock bond data for testing"""
    return {
        "face_value": 1000,
        "coupon_rate": 5.0,  # 5%
        "maturity_date": (datetime.now().replace(year=datetime.now().year + 10)).strftime("%Y-%m-%d"),
        "last_price": 980,
        "credit_rating": "BBB",
        "yield_to_maturity": 0.06  # 6%
    }

@pytest.fixture
def mock_market_data() -> Dict:
    """Mock market data for testing"""
    return {
        "risk_free_rate": 0.035,  # 3.5%
        "market_volatility": 0.15,  # 15%
        "credit_spreads": {
            "AAA": 0.005,
            "AA": 0.01,
            "A": 0.015,
            "BBB": 0.025,
            "BB": 0.04,
            "B": 0.06
        }
    }

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    ) 