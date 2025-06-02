import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.risk.country_risk import CountryRiskAnalyzer, CountryRiskMetrics
from src.data.country_fetcher import CountryDataManager, IMFClient, WorldBankClient, TreasuryClient

@pytest.fixture
def mock_country_data():
    return {
        "macro": {
            "gdp_growth": 2.5,
            "inflation": 3.0,
            "debt_to_gdp": 60.0,
            "current_account": -2.0
        },
        "market": {
            "policy_rate": 4.0,
            "govt_bond_10y": 5.0,
            "fx_rate": 1.2,
            "equity_index": 1000.0
        },
        "risk": {
            "sovereign_spread": 0.02,
            "fx_volatility": 0.10,
            "market_volatility": 0.15
        }
    }

@pytest.fixture
def mock_metrics():
    return CountryRiskMetrics(
        sovereign_spread=0.02,
        local_volatility=0.15,
        currency_vol=0.10,
        inflation_rate=0.03,
        policy_rate=0.04,
        debt_to_gdp=0.60
    )

@pytest.fixture
def country_analyzer():
    """Initialize CountryRiskAnalyzer instance"""
    return CountryRiskAnalyzer()

@pytest.fixture
def country_manager():
    """Initialize CountryDataManager instance"""
    return CountryDataManager()

def test_country_risk_metrics_creation():
    """Test creation of CountryRiskMetrics"""
    metrics = CountryRiskMetrics(
        sovereign_spread=0.02,
        local_volatility=0.15,
        currency_vol=0.10,
        inflation_rate=0.03,
        policy_rate=0.04,
        debt_to_gdp=0.60
    )
    
    assert metrics.sovereign_spread == 0.02
    assert metrics.local_volatility == 0.15
    assert metrics.currency_vol == 0.10
    assert metrics.inflation_rate == 0.03
    assert metrics.policy_rate == 0.04
    assert metrics.debt_to_gdp == 0.60

def test_survival_probability_adjustment(country_analyzer, mock_metrics):
    """Test survival probability adjustment based on country metrics"""
    base_prob = 0.95
    time_horizon = 5.0
    
    adjusted_prob = country_analyzer.adjust_survival_probability(
        base_prob, mock_metrics, time_horizon
    )
    
    assert 0 <= adjusted_prob <= 1.0
    assert adjusted_prob < base_prob  # Should be lower due to country risk

def test_local_hull_white_params(country_analyzer, mock_metrics):
    """Test Hull-White parameter adjustments for local market"""
    reversion, long_term, vol = country_analyzer.local_hull_white_params(mock_metrics)
    
    assert reversion > 0
    assert long_term > mock_metrics.sovereign_spread
    assert vol > 0
    assert vol > 0.02  # Should be higher than base vol

def test_simulate_local_rates(country_analyzer):
    """Test local interest rate simulation"""
    country_code = "BR"  # Brazil as example
    initial_rate = 0.05
    time_steps = 252
    num_paths = 100
    
    rates = country_analyzer.simulate_local_rates(
        country_code,
        initial_rate,
        time_steps,
        num_paths
    )
    
    assert rates.shape == (num_paths, time_steps)
    assert np.all(rates >= 0)  # Rates should be non-negative
    assert np.all(np.isfinite(rates))  # No NaN or inf values

def test_value_at_risk_local(country_analyzer, mock_metrics):
    """Test local market VaR calculation"""
    returns = np.random.normal(0, 0.02, 1000)  # Sample returns
    confidence_level = 0.99
    
    var, es = country_analyzer.value_at_risk_local(
        returns,
        mock_metrics,
        confidence_level
    )
    
    assert var > 0
    assert es > var  # ES should be larger than VaR
    assert np.isfinite(var) and np.isfinite(es)

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

def test_risk_factors(country_analyzer):
    """Test risk factors calculation"""
    risk_factors = country_analyzer.get_risk_factors("USA")
    
    # Verify structure
    assert isinstance(risk_factors, dict)
    assert all(key in risk_factors for key in [
        'gdp_growth', 'inflation', 'debt_to_gdp',
        'current_account', 'market_volatility',
        'fx_volatility', 'sovereign_spread'
    ])
    
    # Verify value ranges
    assert -20 <= risk_factors['gdp_growth'] <= 20
    assert -5 <= risk_factors['inflation'] <= 20
    assert 0 <= risk_factors['debt_to_gdp'] <= 300
    assert -20 <= risk_factors['current_account'] <= 20
    assert 0 <= risk_factors['market_volatility'] <= 1.0
    assert 0 <= risk_factors['fx_volatility'] <= 1.0
    assert 0 <= risk_factors['sovereign_spread'] <= 10.0

def test_error_handling(country_analyzer):
    """Test error handling in risk analysis"""
    # Test with invalid country code
    risk_factors = country_analyzer.get_risk_factors("INVALID")
    
    # Verify default values are used
    assert risk_factors['gdp_growth'] == 0.0
    assert risk_factors['inflation'] == 0.0
    assert risk_factors['debt_to_gdp'] == 0.0
    assert risk_factors['current_account'] == 0.0
    assert risk_factors['market_volatility'] == 0.15
    assert risk_factors['fx_volatility'] == 0.1
    assert risk_factors['sovereign_spread'] == 3.46

def test_market_risk_calculations(country_manager, mock_country_data):
    """Test market risk metric calculations"""
    spread = country_manager._calculate_spread(mock_country_data["market"])
    fx_vol = country_manager._calculate_fx_vol(mock_country_data["market"])
    market_vol = country_manager._calculate_market_vol(mock_country_data["market"])
    
    assert spread >= 0
    assert 0 < fx_vol < 1
    assert 0 < market_vol < 1

def test_risk_analyzer_initialization(country_analyzer):
    """Test risk analyzer initialization"""
    assert isinstance(country_analyzer.country_manager, CountryDataManager)

def test_integration_workflow(country_analyzer):
    """Test full country risk analysis workflow"""
    country_code = "BR"
    initial_rate = 0.05
    base_prob = 0.95
    time_horizon = 5.0
    
    # Get country metrics
    metrics = country_analyzer.get_country_metrics(country_code)
    
    # Adjust survival probability
    adj_prob = country_analyzer.adjust_survival_probability(
        base_prob, metrics, time_horizon
    )
    
    # Simulate local rates
    rates = country_analyzer.simulate_local_rates(
        country_code,
        initial_rate,
        time_steps=252,
        num_paths=1000
    )
    
    # Calculate VaR
    returns = np.diff(rates, axis=1) / rates[:, :-1]  # Calculate returns
    var, es = country_analyzer.value_at_risk_local(
        returns[:, -1],  # Use final period returns
        metrics
    )
    
    assert 0 <= adj_prob <= 1
    assert rates.shape == (1000, 252)
    assert var > 0 and es > 0