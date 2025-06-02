import pytest
import numpy as np
from datetime import datetime
from src.risk.metrics import BondMetrics
from src.risk.simulations import RiskSimulator
from src.risk.model_survival import SurvivalModel
from src.data.country_fetcher import CountryDataManager

class TestBondRiskAnalysis:
    """Test suite for comprehensive bond risk analysis"""
    
    @pytest.fixture
    def metrics(self):
        """Initialize BondMetrics instance"""
        return BondMetrics()
    
    @pytest.fixture
    def simulator(self):
        """Initialize RiskSimulator instance"""
        return RiskSimulator()
    
    @pytest.fixture
    def survival_model(self):
        """Initialize SurvivalModel instance"""
        return SurvivalModel()
    
    @pytest.fixture
    def country_manager(self):
        """Initialize CountryDataManager instance"""
        return CountryDataManager()
    
    def test_modified_duration(self, metrics, mock_bond_data):
        """Test modified duration calculation"""
        # Generate cash flows
        periods = int(2 * 3)  # 3 years
        times = [(i+1)/2 for i in range(periods)]
        coupon = (mock_bond_data['coupon_rate']/200) * mock_bond_data['face_value']
        cash_flows = [coupon] * periods
        cash_flows[-1] += mock_bond_data['face_value']
        
        ytm = mock_bond_data['yield_to_maturity']
        mod_dur = metrics.modified_duration(cash_flows, times, ytm)
        
        # Duration should be reasonable for a 3-year bond
        assert 2.5 <= mod_dur <= 3.0
    
    def test_convexity(self, metrics, mock_bond_data):
        """Test convexity calculation"""
        # Generate cash flows
        periods = int(2 * 3)  # 3 years
        times = [(i+1)/2 for i in range(periods)]
        coupon = (mock_bond_data['coupon_rate']/200) * mock_bond_data['face_value']
        cash_flows = [coupon] * periods
        cash_flows[-1] += mock_bond_data['face_value']
        
        ytm = mock_bond_data['yield_to_maturity']
        convexity = metrics.convexity(cash_flows, times, ytm)
        
        # Convexity should be positive and reasonable
        assert convexity > 0
        assert convexity < 20
    
    def test_credit_spread(self, metrics, mock_bond_data, mock_country_data):
        """Test credit spread calculation"""
        ytm = mock_bond_data['yield_to_maturity']
        rating = mock_bond_data['credit_rating']
        
        spread = metrics.credit_spread(ytm, rating)
        
        # Spread should be positive and reasonable
        assert spread >= 0
        assert spread <= 0.05  # 500 bps max for investment grade
    
    def test_risk_metrics(self, metrics, mock_bond_data, mock_country_data):
        """Test comprehensive risk metrics calculation"""
        risk_metrics = metrics.calculate_risk_metrics(mock_bond_data)
        
        # Verify all key metrics are present
        assert 'ytm' in risk_metrics
        assert 'modified_duration' in risk_metrics
        assert 'convexity' in risk_metrics
        assert 'credit_spread' in risk_metrics
        assert 'key_rate_durations' in risk_metrics
        assert 'market_volatility' in risk_metrics
        assert 'risk_free_rate' in risk_metrics
        
        # Verify values are reasonable
        assert 0 <= risk_metrics['ytm'] <= 0.15  # 0-15%
        assert 0 <= risk_metrics['modified_duration'] <= 30  # 0-30 years
        assert 0 <= risk_metrics['convexity'] <= 1000
        assert 0 <= risk_metrics['credit_spread'] <= 0.10  # 0-1000 bps
        assert isinstance(risk_metrics['key_rate_durations'], dict)
        assert 0 <= risk_metrics['market_volatility'] <= 1.0  # 0-100%
        assert 0 <= risk_metrics['risk_free_rate'] <= 0.10  # 0-10%