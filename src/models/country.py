"""Country risk analysis module for Bond Risk Analyzer."""

from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
import json

@dataclass
class CountryRisk:
    """Container for country risk metrics"""
    # Basic Info
    country_code: str
    country_name: str
    region: str
    income_group: str
    
    # Economic Indicators
    gdp_growth: float = 0.0
    inflation_rate: float = 0.0
    unemployment_rate: float = 0.0
    debt_to_gdp: float = 0.0
    current_account_balance: float = 0.0
    foreign_reserves: float = 0.0
    
    # Sovereign Ratings
    sp_rating: str = "NR"
    moodys_rating: str = "NR"
    fitch_rating: str = "NR"
    
    # Bond Market Metrics
    govt_10y_yield: float = 0.0
    yield_spread_vs_us: float = 0.0
    bond_market_liquidity: float = 0.0
    
    # Risk Scores (0-1, higher is riskier)
    economic_risk: float = 0.5
    political_risk: float = 0.5
    financial_risk: float = 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'country_code': self.country_code,
            'country_name': self.country_name,
            'region': self.region,
            'income_group': self.income_group,
            'economic_indicators': {
                'gdp_growth': self.gdp_growth,
                'inflation_rate': self.inflation_rate,
                'unemployment_rate': self.unemployment_rate,
                'debt_to_gdp': self.debt_to_gdp,
                'current_account_balance': self.current_account_balance,
                'foreign_reserves': self.foreign_reserves
            },
            'sovereign_ratings': {
                'sp': self.sp_rating,
                'moodys': self.moodys_rating,
                'fitch': self.fitch_rating
            },
            'bond_metrics': {
                'govt_10y_yield': self.govt_10y_yield,
                'yield_spread_vs_us': self.yield_spread_vs_us,
                'bond_market_liquidity': self.bond_market_liquidity
            },
            'risk_scores': {
                'economic_risk': self.economic_risk,
                'political_risk': self.political_risk,
                'financial_risk': self.financial_risk
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CountryRisk':
        """Create from dictionary"""
        return cls(
            country_code=data['country_code'],
            country_name=data['country_name'],
            region=data['region'],
            income_group=data['income_group'],
            gdp_growth=data['economic_indicators']['gdp_growth'],
            inflation_rate=data['economic_indicators']['inflation_rate'],
            unemployment_rate=data['economic_indicators']['unemployment_rate'],
            debt_to_gdp=data['economic_indicators']['debt_to_gdp'],
            current_account_balance=data['economic_indicators']['current_account_balance'],
            foreign_reserves=data['economic_indicators']['foreign_reserves'],
            sp_rating=data['sovereign_ratings']['sp'],
            moodys_rating=data['sovereign_ratings']['moodys'],
            fitch_rating=data['sovereign_ratings']['fitch'],
            govt_10y_yield=data['bond_metrics']['govt_10y_yield'],
            yield_spread_vs_us=data['bond_metrics']['yield_spread_vs_us'],
            bond_market_liquidity=data['bond_metrics']['bond_market_liquidity'],
            economic_risk=data['risk_scores']['economic_risk'],
            political_risk=data['risk_scores']['political_risk'],
            financial_risk=data['risk_scores']['financial_risk']
        )
    
    def calculate_overall_risk(self) -> float:
        """Calculate overall country risk score"""
        weights = {
            'economic': 0.4,
            'political': 0.3,
            'financial': 0.3
        }
        
        return (
            self.economic_risk * weights['economic'] +
            self.political_risk * weights['political'] +
            self.financial_risk * weights['financial']
        )
    
    def get_investment_recommendation(self) -> str:
        """Get investment recommendation based on risk analysis"""
        overall_risk = self.calculate_overall_risk()
        
        if overall_risk < 0.3:
            return "Strong Buy - Low risk sovereign bonds with stable outlook"
        elif overall_risk < 0.5:
            return "Buy - Moderate risk with acceptable return potential"
        elif overall_risk < 0.7:
            return "Hold - High risk, consider reducing exposure"
        else:
            return "Sell/Avoid - Very high risk, significant default potential"
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CountryRisk':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data) 