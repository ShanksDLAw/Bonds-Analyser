"""Survival analysis module for bond risk assessment"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
from datetime import datetime
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from src.data.country_fetcher import CountryDataFetcher
from src.models.company import CompanyFinancials
from src.config import Config

@dataclass
class SurvivalMetrics:
    """Container for survival analysis metrics"""
    probability_of_default: float
    loss_given_default: float
    exposure_at_default: float
    expected_loss: float
    rating: str
    time_horizon: float
    expected_lifetime: float
    hazard_rate: float
    survival_probability: float
    confidence_interval: tuple

    @property
    def time_to_default(self):
        return self.expected_lifetime

class SurvivalModel:
    """Bond survival analysis model using Treasury and IMF data"""
    
    def __init__(self):
        self.country_manager = CountryDataFetcher()
        
    def calculate_default_probability(self, bond_data: Dict, country_code: str = 'US') -> float:
        """Calculate probability of default using market and macro data"""
        # Get comprehensive metrics
        metrics = self.country_manager.get_comprehensive_metrics(country_code)
        
        # Extract key risk factors
        market_data = metrics['market_data']
        macro_data = metrics['macro_indicators']
        
        # Calculate market-based factors
        market_volatility = self._calculate_market_volatility(market_data)
        yield_spread = self._calculate_yield_spread(bond_data, market_data)
        
        # Calculate macro factors
        macro_score = self._calculate_macro_score(macro_data)
        
        # Combine factors into default probability
        base_prob = 0.01  # Base default rate
        market_factor = np.exp(market_volatility * 0.1 + yield_spread * 0.2)
        macro_factor = np.exp(macro_score * 0.15)
        
        return min(base_prob * market_factor * macro_factor, 1.0)
    
    def _calculate_market_volatility(self, market_data: Dict) -> float:
        """Calculate market volatility indicator"""
        try:
            rates_vol = market_data.get('interest_rates_volatility', 0.0)
            fx_vol = market_data.get('fx_volatility', 0.0)
            return (rates_vol + fx_vol) / 2
        except Exception:
            return 0.0
    
    def _calculate_yield_spread(self, bond_data: Dict, market_data: Dict) -> float:
        """Calculate yield spread over risk-free rate"""
        try:
            risk_free = float(market_data.get('treasury_10y', 0.0))
            bond_yield = float(bond_data.get('yield_to_maturity', 0.0))
            return max(bond_yield - risk_free, 0.0)
        except Exception:
            return 0.0
    
    def _calculate_macro_score(self, macro_data: Dict) -> float:
        """Calculate macroeconomic risk score"""
        try:
            gdp_growth = float(macro_data.get('gdp_growth', 0.0))
            inflation = float(macro_data.get('inflation_rate', 0.0))
            debt_gdp = float(macro_data.get('debt_to_gdp', 0.0))
            
            # Normalize and combine factors
            gdp_factor = max(min((gdp_growth + 5) / 10, 1), 0)
            inflation_factor = max(min((2 - abs(inflation - 2)) / 2, 1), 0)
            debt_factor = max(min((100 - debt_gdp) / 100, 1), 0)
            
            return (gdp_factor + inflation_factor + debt_factor) / 3
        except Exception:
            return 0.5

    def get_yield_curve(self) -> Dict[str, float]:
        """Get current Treasury yield curve"""
        try:
            rates = self.country_manager.treasury.get_interest_rates()
            curve = {}
            if rates and 'data' in rates:
                for rate in rates['data']:
                    tenor = rate.get('security_desc', '')
                    if tenor in ['3-Month', '2-Year', '5-Year', '10-Year', '30-Year']:
                        curve[tenor] = float(rate.get('avg_interest_rate_amt', 0.0))
            return curve
        except Exception as e:
            print(f"Error getting yield curve: {str(e)}")
            return {}

    def get_market_volatility(self) -> Tuple[float, float]:
        """Get market volatility indicators"""
        try:
            metrics = self.country_manager.get_comprehensive_metrics('US')
            market_data = metrics['market_data']
            
            vix = market_data.get('market_volatility', 15.0)
            move = market_data.get('rates_volatility', 70.0)
            
            return vix, move
        except Exception:
            return 15.0, 70.0  # Default values

    def get_transition_rates(self) -> Dict[str, Dict[str, float]]:
        """Get credit rating transition probabilities"""
        # This could be enhanced with real market data
        return {
            'AAA': {'AA': 0.09, 'A': 0.06, 'BBB': 0.02},
            'AA': {'AAA': 0.02, 'A': 0.22, 'BBB': 0.06},
            'A': {'AA': 0.06, 'BBB': 0.25, 'BB': 0.05},
            'BBB': {'A': 0.05, 'BB': 0.18, 'B': 0.03},
            'BB': {'BBB': 0.07, 'B': 0.11, 'CCC': 0.04},
            'B': {'BB': 0.06, 'CCC': 0.12, 'D': 0.06},
            'CCC': {'B': 0.03, 'D': 0.27}
        }

    def get_recovery_rates(self) -> Dict[str, float]:
        """Get historical recovery rates by rating"""
        return {
            'AAA': 0.95,
            'AA': 0.92,
            'A': 0.90,
            'BBB': 0.87,
            'BB': 0.80,
            'B': 0.65,
            'CCC': 0.45
        }

    def get_market_data(self) -> Tuple[float, Dict]:
        """Get market volatility and yield curve data"""
        market_data = self.country_manager.get_comprehensive_metrics('US')
        vix = market_data['market_data'].get('market_volatility', 15.0)
        curve = self.country_manager.treasury.get_interest_rates()
        return vix, curve
    
    def get_transition_matrix(self) -> Dict:
        """Get credit rating transition matrix"""
        market_data = self.country_manager.get_comprehensive_metrics('US')
        return market_data.get('credit_data', {}).get('transition_matrix', {})

class SurvivalAnalyzer:
    """Credit risk survival analyzer"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.config = Config()
        self.country_manager = CountryDataFetcher()
        
        # Model parameters
        self.time_horizon = 5  # years
        self.confidence_level = 0.95
        
        # Risk factors and weights
        self.risk_factors = {
            'leverage': 0.25,
            'profitability': 0.20,
            'liquidity': 0.15,
            'size': 0.10,
            'market': 0.15,
            'industry': 0.15
        }
    
    def analyze_survival(
        self,
        company: CompanyFinancials,
        time_horizon: Optional[int] = None
    ) -> SurvivalMetrics:
        """Perform survival analysis"""
        try:
            # Use provided time horizon or default
            horizon = time_horizon or self.time_horizon
            
            # Get company metrics
            metrics = company.get_credit_metrics()
            coverage = company.get_coverage_ratios()
            
            # Calculate hazard rate
            hazard_rate = self._calculate_hazard_rate(metrics, coverage)
            
            # Calculate survival probability
            survival_prob = np.exp(-hazard_rate * horizon)
            
            # Calculate probability of default
            prob_default = 1 - survival_prob
            
            # Calculate expected lifetime
            expected_lifetime = 1 / hazard_rate if hazard_rate > 0 else float('inf')
            
            # Calculate confidence interval
            ci = self._calculate_confidence_interval(
                survival_prob,
                hazard_rate,
                horizon
            )
            
            return SurvivalMetrics(
                probability_of_default=prob_default,
                loss_given_default=0.45,  # Standard LGD
                exposure_at_default=metrics.get('total_debt', 0),
                expected_loss=prob_default * 0.45 * metrics.get('total_debt', 0),
                rating=metrics.get('rating', 'NR'),
                time_horizon=horizon,
                expected_lifetime=expected_lifetime,
                hazard_rate=hazard_rate,
                survival_probability=survival_prob,
                confidence_interval=ci
            )
            
        except Exception as e:
            print(f"Error in survival analysis: {str(e)}")
            return SurvivalMetrics(
                probability_of_default=0.05,
                loss_given_default=0.45,
                exposure_at_default=1000000,
                expected_loss=22500,
                rating='NR',
                time_horizon=5,
                expected_lifetime=20.0,
                hazard_rate=0.05,
                survival_probability=0.95,
                confidence_interval=(0.90, 1.00)
            )
    
    def _calculate_hazard_rate(
        self,
        metrics: Dict,
        coverage: Dict
    ) -> float:
        """Calculate hazard rate based on company metrics"""
        try:
            # Get key metrics
            leverage = metrics['leverage_ratio']
            profitability = metrics['operating_margin']
            liquidity = coverage['interest_coverage']
            
            # Calculate component scores
            leverage_score = min(1.0, leverage / 5.0)  # Cap at 5x leverage
            profitability_score = max(0.0, min(1.0, (0.2 - profitability) / 0.2))
            liquidity_score = max(0.0, min(1.0, 2.0 / liquidity))
            
            # Combined score with weights
            hazard_rate = (
                self.risk_factors['leverage'] * leverage_score +
                self.risk_factors['profitability'] * profitability_score +
                self.risk_factors['liquidity'] * liquidity_score
            )
            
            # Scale to reasonable range (0-20%)
            return min(0.20, max(0.01, hazard_rate))
            
        except Exception as e:
            print(f"Error calculating hazard rate: {str(e)}")
            return 0.05  # Default 5% hazard rate
    
    def _calculate_confidence_interval(
        self,
        survival_prob: float,
        hazard_rate: float,
        horizon: int
    ) -> tuple:
        """Calculate confidence interval for survival probability"""
        try:
            # Standard error calculation
            se = np.sqrt(
                survival_prob * (1 - survival_prob) / (horizon * hazard_rate)
            )
            
            # Z-score for confidence level
            z = stats.norm.ppf((1 + self.confidence_level) / 2)
            
            # Calculate interval
            lower = max(0.0, survival_prob - z * se)
            upper = min(1.0, survival_prob + z * se)
            
            return (lower, upper)
            
        except Exception as e:
            print(f"Error calculating confidence interval: {str(e)}")
            return (
                max(0.0, survival_prob - 0.05),
                min(1.0, survival_prob + 0.05)
            )
    
    def get_survival_curve(
        self,
        company: CompanyFinancials,
        max_horizon: int = 10
    ) -> pd.DataFrame:
        """Generate survival curve data"""
        try:
            # Calculate metrics
            metrics = self.analyze_survival(company, max_horizon)
            
            # Generate time points
            times = np.linspace(0, max_horizon, 100)
            
            # Calculate survival probabilities
            survival_probs = np.exp(-metrics.hazard_rate * times)
            
            # Create DataFrame
            return pd.DataFrame({
                'time': times,
                'survival_probability': survival_probs
            })
            
        except Exception as e:
            print(f"Error generating survival curve: {str(e)}")
            return pd.DataFrame({
                'time': [0, max_horizon],
                'survival_probability': [1.0, 0.95]
            })
    
    def calculate_expected_loss(
        self,
                             exposure: float,
                             rating: str,
        time_horizon: float = 1.0
    ) -> SurvivalMetrics:
        """Calculate expected loss using survival analysis"""
        try:
            # Get market data
            market_data = self.country_manager.get_comprehensive_metrics('US')
            
            # Get rating transition probabilities
            transition_probs = market_data.get('credit_data', {}).get(
                'transition_matrix', {}
            ).get(rating, {})
            
            if not transition_probs:
                # Use default probabilities if no transition matrix
                pd = {
                    'AAA': 0.0001, 'AA': 0.0002, 'A': 0.0005,
                    'BBB': 0.002, 'BB': 0.008, 'B': 0.03,
                    'CCC': 0.10, 'CC': 0.20, 'C': 0.30, 'D': 1.0
                }.get(rating[:2], 0.05)
            else:
                pd = transition_probs.get('D', 0.05)
            
            # Adjust for time horizon
            pd = 1 - (1 - pd) ** time_horizon
            
            # Calculate loss given default based on seniority
            lgd = 0.45  # Standard LGD for senior unsecured
            
            # Calculate exposure at default (assume no amortization)
            ead = exposure
            
            # Calculate expected loss
            el = pd * lgd * ead
            
            # Estimate time to default using hazard rate
            hazard_rate = -np.log(1 - pd) / time_horizon
            ttd = 1 / hazard_rate if hazard_rate > 0 else float('inf')
            
            return SurvivalMetrics(
                probability_of_default=pd,
                loss_given_default=lgd,
                exposure_at_default=ead,
                expected_loss=el,
                rating=rating,
                time_horizon=time_horizon,
                expected_lifetime=ttd,
                hazard_rate=hazard_rate,
                survival_probability=1-pd,
                confidence_interval=(0.90, 1.00)
            )
            
        except Exception as e:
            print(f"Error in expected loss calculation: {str(e)}")
            return SurvivalMetrics(
                probability_of_default=0.05,
                loss_given_default=0.45,
                exposure_at_default=exposure,
                expected_loss=exposure * 0.05 * 0.45,
                rating=rating,
                time_horizon=time_horizon,
                expected_lifetime=20.0,
                hazard_rate=0.05,
                survival_probability=0.95,
                confidence_interval=(0.90, 1.00)
            )
    
    def credit_var_analysis(self, portfolio: List[Dict[str, float]], confidence_level: float = 0.99, time_horizon: float = 1.0) -> Dict[str, float]:
        """Calculate portfolio credit VaR using Monte Carlo simulation"""
        try:
            n_sims = 10000
            losses = np.zeros(n_sims)
            
            for sim in range(n_sims):
                portfolio_loss = 0
                for position in portfolio:
                    exposure = position['exposure']
                    rating = position['rating']
                    
                    # Get risk parameters
                    risk_metrics = self.calculate_expected_loss(
                        exposure, rating, time_horizon
                    )
                    
                    # Simulate default event
                    if np.random.random() < risk_metrics.probability_of_default:
                        loss = exposure * risk_metrics.loss_given_default
                        portfolio_loss += loss  # Fixed indentation
                    
                losses[sim] = portfolio_loss
            
            # Calculate VaR and ES
            credit_var = np.percentile(losses, confidence_level * 100)
            es = np.mean(losses[losses > credit_var])
            
            return {
                'credit_var': float(credit_var),
                'expected_shortfall': float(es),
                'expected_loss': float(np.mean(losses)),
                'loss_volatility': float(np.std(losses))
            }
                
        except Exception as e:
            print(f"Error in credit VaR analysis: {str(e)}")
            return {
                'credit_var': 0.0,
                'expected_shortfall': 0.0,
                'expected_loss': 0.0,
                'loss_volatility': 0.0
            }
    
    def stress_test_portfolio(
        self,
        portfolio: List[Dict[str, float]],
        stress_scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Run stress tests on portfolio"""
        try:
            results = {}
            
            for scenario_name, shocks in stress_scenarios.items():
                scenario_losses = []
                
                for position in portfolio:
                    exposure = position['exposure']
                    rating = position['rating']
                    
                    # Get base metrics
                    base_metrics = self.calculate_expected_loss(
                        exposure, rating
                    )
                    
                    # Apply shocks
                    stressed_pd = base_metrics.probability_of_default * (
                        1 + shocks.get('pd_shock', 0)
                    )
                    stressed_lgd = base_metrics.loss_given_default * (
                        1 + shocks.get('lgd_shock', 0)
                    )
                    
                    # Calculate stressed loss
                    stressed_loss = exposure * stressed_pd * stressed_lgd
                    scenario_losses.append(stressed_loss)
                
                results[scenario_name] = {
                    'total_loss': sum(scenario_losses),
                    'max_loss': max(scenario_losses),
                    'avg_loss': np.mean(scenario_losses)
                }
            
            return results
            
        except Exception as e:
            print(f"Error in stress testing: {str(e)}")
            return {}