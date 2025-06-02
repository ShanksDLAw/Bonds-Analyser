"""Macro data routing module."""

from typing import Dict, Optional
import logging
from .worldbank_loader import WorldBankClient

logger = logging.getLogger(__name__)

class MacroRouter:
    """Routes macro data requests to appropriate data sources"""
    
    def __init__(self):
        """Initialize router"""
        self.wb_client = WorldBankClient()
    
    def get_country_data(self, country_code: str, year: Optional[int] = None) -> Dict:
        """Get comprehensive country data"""
        try:
            # Get World Bank data
            wb_data = self.wb_client.get_country_data(country_code, year)
            
            # Add historical data for key metrics
            if wb_data:
                for indicator in ['GDP_GROWTH', 'INFLATION', 'DEBT_TO_GDP']:
                    historical = self.wb_client.get_historical_data(
                        country_code,
                        indicator,
                        year - 5 if year else 2018,
                        year if year else 2023
                    )
                    if not historical.empty:
                        wb_data[f"{indicator}_HISTORY"] = historical.to_dict()['value']
            
            return wb_data
            
        except Exception as e:
            logger.error(f"Error getting country data: {str(e)}")
            return {}
    
    def get_macro_data(self, country_code: str) -> Dict:
        """Get comprehensive macro data for a country"""
        try:
            # Try to get data from World Bank
            wb_data = self.wb_client.get_country_data(country_code)
            
            # Format and return data
            return {
                'gdp_growth': wb_data.get('gdp_growth', None),
                'inflation': wb_data.get('inflation', None),
                'unemployment': wb_data.get('unemployment', None),
                'current_account': wb_data.get('current_account', None),
                'fiscal_balance': wb_data.get('fiscal_balance', None)
            }
            
        except Exception as e:
            logger.error(f"Error getting macro data for {country_code}: {e}")
            # Return default values if data fetch fails
            return {
                'gdp_growth': 2.0,
                'inflation': 2.5,
                'unemployment': 5.0,
                'current_account': -2.0,
                'fiscal_balance': -3.0
            }
    
    def get_risk_free_rate(self, maturity: str = '10Y') -> float:
        """Get risk-free rate for specified maturity"""
        # Default values based on typical US Treasury yields
        default_rates = {
            '3M': 0.03,
            '6M': 0.035,
            '1Y': 0.04,
            '2Y': 0.045,
            '5Y': 0.05,
            '10Y': 0.055,
            '30Y': 0.06
        }
        
        return default_rates.get(maturity, 0.055)  # Default to 10Y rate
    
    def get_market_environment(self, country_code: str) -> Dict:
        """Get market environment data"""
        try:
            # Get base data
            base_data = self.get_macro_data(country_code)
            
            # Add market-specific metrics
            market_data = {
                'volatility': 0.15,  # Default market volatility
                'liquidity_score': 0.8,  # Default liquidity score
                'market_sentiment': 'neutral',
                'risk_appetite': 'moderate'
            }
            
            # Adjust based on macro conditions
            if base_data['gdp_growth'] > 3.0:
                market_data['market_sentiment'] = 'positive'
                market_data['risk_appetite'] = 'high'
            elif base_data['gdp_growth'] < 1.0:
                market_data['market_sentiment'] = 'negative'
                market_data['risk_appetite'] = 'low'
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market environment for {country_code}: {e}")
            return {
                'volatility': 0.15,
                'liquidity_score': 0.8,
                'market_sentiment': 'neutral',
                'risk_appetite': 'moderate'
            }
    
    def get_sovereign_metrics(self, country_code: str) -> Dict:
        """Get sovereign risk metrics"""
        try:
            # Get base macro data
            macro_data = self.get_macro_data(country_code)
            
            # Calculate sovereign risk score
            risk_score = self._calculate_sovereign_risk(macro_data)
            
            return {
                'sovereign_risk_score': risk_score,
                'credit_rating': self._get_implied_rating(risk_score),
                'default_probability': self._calculate_default_prob(risk_score),
                'macro_stability': self._calculate_macro_stability(macro_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting sovereign metrics for {country_code}: {e}")
            return {
                'sovereign_risk_score': 0.5,
                'credit_rating': 'BBB',
                'default_probability': 0.02,
                'macro_stability': 0.7
            }
    
    def _calculate_sovereign_risk(self, macro_data: Dict) -> float:
        """Calculate sovereign risk score from macro data"""
        weights = {
            'gdp_growth': -0.3,  # Negative weight (higher growth = lower risk)
            'inflation': 0.2,
            'unemployment': 0.2,
            'current_account': -0.15,
            'fiscal_balance': -0.15
        }
        
        score = 0.5  # Base score
        
        for metric, weight in weights.items():
            if metric in macro_data and macro_data[metric] is not None:
                # Normalize the metric value
                if metric == 'gdp_growth':
                    norm_value = 1 - (macro_data[metric] + 2) / 8  # Range: -2% to 6%
                elif metric == 'inflation':
                    norm_value = macro_data[metric] / 10  # Range: 0% to 10%
                elif metric == 'unemployment':
                    norm_value = macro_data[metric] / 20  # Range: 0% to 20%
                elif metric in ['current_account', 'fiscal_balance']:
                    norm_value = (macro_data[metric] + 10) / 20  # Range: -10% to 10%
                else:
                    continue
                
                # Add weighted contribution
                score += weight * max(0, min(1, norm_value))
        
        return max(0, min(1, score))
    
    def _get_implied_rating(self, risk_score: float) -> str:
        """Get implied credit rating from risk score"""
        if risk_score < 0.2:
            return 'AAA'
        elif risk_score < 0.3:
            return 'AA'
        elif risk_score < 0.4:
            return 'A'
        elif risk_score < 0.6:
            return 'BBB'
        elif risk_score < 0.75:
            return 'BB'
        elif risk_score < 0.9:
            return 'B'
        else:
            return 'CCC'
    
    def _calculate_default_prob(self, risk_score: float) -> float:
        """Calculate default probability from risk score"""
        # Simple exponential relationship
        return min(1.0, 0.01 * (2.718 ** (5 * risk_score)))
    
    def _calculate_macro_stability(self, macro_data: Dict) -> float:
        """Calculate macro stability score"""
        # Higher score = more stable
        stability = 1.0
        
        if 'gdp_growth' in macro_data:
            # Penalize extreme growth (too high or too low)
            stability *= 1 - abs(macro_data['gdp_growth'] - 2.5) / 10
        
        if 'inflation' in macro_data:
            # Penalize high inflation
            stability *= 1 - macro_data['inflation'] / 20
        
        if 'unemployment' in macro_data:
            # Penalize high unemployment
            stability *= 1 - macro_data['unemployment'] / 20
        
        return max(0, min(1, stability))