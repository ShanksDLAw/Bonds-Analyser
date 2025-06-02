"""Tests for the survival analysis model."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List

from src.risk.model_survival import SurvivalModel, SurvivalAnalyzer, SurvivalMetrics
from src.models.company import CompanyFinancials
from src.data.country_fetcher import CountryDataFetcher

@pytest.fixture
def survival_model():
    """Create a survival model instance."""
    return SurvivalModel()

@pytest.fixture
def survival_analyzer():
    """Create a survival analyzer instance."""
    return SurvivalAnalyzer()

@pytest.fixture
def sample_bond_data():
    """Create sample bond data."""
    return {
        'yield_to_maturity': 0.05,
        'duration': 5.0,
        'rating': 'BBB',
        'face_value': 1000000,
        'maturity_date': '2030-01-01'
    }

@pytest.fixture
def sample_portfolio():
    """Create sample portfolio data."""
    return [
        {'exposure': 1000000, 'rating': 'AAA'},
        {'exposure': 2000000, 'rating': 'BBB'},
        {'exposure': 500000, 'rating': 'BB'}
    ]

def test_survival_metrics():
    """Test survival metrics dataclass."""
    metrics = SurvivalMetrics(
        probability_of_default=0.02,
        loss_given_default=0.45,
        exposure_at_default=1000000,
        expected_loss=9000,
        time_to_default=8.5
    )
    
    assert metrics.probability_of_default == 0.02
    assert metrics.loss_given_default == 0.45
    assert metrics.exposure_at_default == 1000000
    assert metrics.expected_loss == 9000
    assert metrics.time_to_default == 8.5

def test_default_probability_calculation(survival_model, sample_bond_data):
    """Test default probability calculation."""
    pd = survival_model.calculate_default_probability(sample_bond_data)
    
    assert isinstance(pd, float)
    assert 0 <= pd <= 1
    
def test_yield_spread_calculation(survival_model, sample_bond_data):
    """Test yield spread calculation."""
    market_data = {'treasury_10y': 0.03}
    spread = survival_model._calculate_yield_spread(sample_bond_data, market_data)
    
    assert spread == pytest.approx(0.02)
    assert spread >= 0

def test_macro_score_calculation(survival_model):
    """Test macro score calculation."""
    macro_data = {
        'gdp_growth': 2.5,
        'inflation_rate': 2.1,
        'debt_to_gdp': 60.0
    }
    score = survival_model._calculate_macro_score(macro_data)
    
    assert 0 <= score <= 1

def test_expected_loss_calculation(survival_analyzer, sample_bond_data):
    """Test expected loss calculation."""
    metrics = survival_analyzer.calculate_expected_loss(
        exposure=1000000,
        rating='BBB',
        time_horizon=1.0
    )
    
    assert isinstance(metrics, SurvivalMetrics)
    assert 0 <= metrics.probability_of_default <= 1
    assert 0 <= metrics.loss_given_default <= 1
    assert metrics.exposure_at_default > 0
    assert metrics.expected_loss >= 0
    assert metrics.time_to_default > 0

def test_credit_var_analysis(survival_analyzer, sample_portfolio):
    """Test credit VaR analysis."""
    results = survival_analyzer.credit_var_analysis(
        portfolio=sample_portfolio,
        confidence_level=0.99,
        time_horizon=1.0
    )
    
    assert 'credit_var' in results
    assert 'expected_shortfall' in results
    assert 'expected_loss' in results
    assert 'loss_volatility' in results
    assert all(v >= 0 for v in results.values())

def test_stress_test_portfolio(survival_analyzer, sample_portfolio):
    """Test portfolio stress testing."""
    stress_scenarios = {
        'severe_recession': {
            'pd_shock': 2.0,  # 200% increase in PD
            'lgd_shock': 0.5   # 50% increase in LGD
        },
        'moderate_stress': {
            'pd_shock': 0.5,   # 50% increase in PD
            'lgd_shock': 0.2    # 20% increase in LGD
        }
    }
    
    results = survival_analyzer.stress_test_portfolio(
        portfolio=sample_portfolio,
        stress_scenarios=stress_scenarios
    )
    
    assert len(results) == len(stress_scenarios)
    for scenario in results.values():
        assert 'total_loss' in scenario
        assert 'max_loss' in scenario
        assert 'avg_loss' in scenario
        assert scenario['max_loss'] <= scenario['total_loss']
        assert scenario['avg_loss'] <= scenario['total_loss']

def test_error_handling(survival_analyzer):
    """Test error handling in survival analysis."""
    # Test with invalid portfolio data
    invalid_portfolio = [{'exposure': -1000, 'rating': 'INVALID'}]
    results = survival_analyzer.credit_var_analysis(invalid_portfolio)
    
    assert results['credit_var'] == 0.0
    assert results['expected_shortfall'] == 0.0
    
    # Test with invalid stress scenarios
    results = survival_analyzer.stress_test_portfolio(
        portfolio=[],
        stress_scenarios={'invalid': {'invalid_shock': 1.0}}
    )
    
    assert isinstance(results, dict)
    assert len(results) == 0

def test_transition_rates(survival_model):
    """Test credit rating transition rates."""
    rates = survival_model.get_transition_rates()
    
    assert isinstance(rates, dict)
    assert 'AAA' in rates
    assert 'BBB' in rates
    for from_rating, transitions in rates.items():
        assert isinstance(transitions, dict)
        assert all(0 <= prob <= 1 for prob in transitions.values())

def test_recovery_rates(survival_model):
    """Test recovery rates by rating."""
    rates = survival_model.get_recovery_rates()
    
    assert isinstance(rates, dict)
    assert all(0 <= rate <= 1 for rate in rates.values())
    assert rates['AAA'] > rates['CCC']  # Higher ratings should have higher recovery 