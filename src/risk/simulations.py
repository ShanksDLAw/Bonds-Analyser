"""Monte Carlo simulation module for bond risk analysis"""

import numpy as np
from scipy.stats import norm, t
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.data.country_fetcher import CountryDataFetcher
from src.data.fetcher import CompanyFinancials, Bond
from src.risk.metrics import RiskAnalyzer, MetricsError
from src.config import Config, ConfigError

# Configure logging
logger = logging.getLogger(__name__)

class SimulationError(Exception):
    """Custom exception for simulation errors"""
    pass

@dataclass
class SimulationResult:
    """Container for simulation results"""
    price_distribution: np.ndarray
    yield_distribution: np.ndarray
    var_95: float
    var_99: float
    expected_shortfall: float
    stress_scenarios: Dict[str, float]
    correlation_matrix: pd.DataFrame
    risk_contributions: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation results to dictionary"""
        return {
            'price_stats': {
                'mean': float(np.mean(self.price_distribution)),
                'std': float(np.std(self.price_distribution)),
                'min': float(np.min(self.price_distribution)),
                'max': float(np.max(self.price_distribution)),
                'percentiles': {
                    '1%': float(np.percentile(self.price_distribution, 1)),
                    '5%': float(np.percentile(self.price_distribution, 5)),
                    '25%': float(np.percentile(self.price_distribution, 25)),
                    '50%': float(np.percentile(self.price_distribution, 50)),
                    '75%': float(np.percentile(self.price_distribution, 75)),
                    '95%': float(np.percentile(self.price_distribution, 95)),
                    '99%': float(np.percentile(self.price_distribution, 99))
                }
            },
            'yield_stats': {
                'mean': float(np.mean(self.yield_distribution)),
                'std': float(np.std(self.yield_distribution)),
                'min': float(np.min(self.yield_distribution)),
                'max': float(np.max(self.yield_distribution))
            },
            'risk_metrics': {
                'var_95': self.var_95,
                'var_99': self.var_99,
                'expected_shortfall': self.expected_shortfall
            },
            'stress_scenarios': self.stress_scenarios,
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'risk_contributions': self.risk_contributions
        }

@dataclass
class ScenarioParams:
    """Scenario parameters"""
    yield_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    growth_adjustment: float = 1.0

class RiskSimulator:
    """Monte Carlo simulation for bond risk analysis"""
    
    def __init__(self, n_scenarios: int = 10000, seed: Optional[int] = None):
        """Initialize simulator with number of scenarios and random seed"""
        try:
            self.n_scenarios = n_scenarios
            self.seed = seed
            if seed is not None:
                np.random.seed(seed)
                
            self.country_manager = CountryDataFetcher()
            self.risk_analyzer = RiskAnalyzer()
            
            # Default simulation parameters
            self.params = {
                'rate_vol': 0.15,  # Interest rate volatility
                'spread_vol': 0.25,  # Credit spread volatility
                'recovery_rate': 0.4,  # Recovery rate in default
                'correlation': {  # Correlation matrix
                    'rates_spreads': 0.3,
                    'rates_growth': -0.2,
                    'spreads_growth': -0.4
                }
            }
            
            # Define standard scenarios
            self.scenarios = {
                'base_case': ScenarioParams(),
                'stress_test': ScenarioParams(
                    yield_multiplier=2.0,
                    spread_multiplier=2.0,
                    volatility_multiplier=1.5,
                    growth_adjustment=0.5
                ),
                'rate_hike': ScenarioParams(
                    yield_multiplier=1.5,
                    spread_multiplier=1.2,
                    volatility_multiplier=1.3,
                    growth_adjustment=0.8
                ),
                'recession': ScenarioParams(
                    yield_multiplier=0.8,
                    spread_multiplier=2.5,
                    volatility_multiplier=2.0,
                    growth_adjustment=0.3
                ),
                'recovery': ScenarioParams(
                    yield_multiplier=1.2,
                    spread_multiplier=0.7,
                    volatility_multiplier=0.8,
                    growth_adjustment=1.5
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize RiskSimulator: {str(e)}")
            raise SimulationError(f"Initialization error: {str(e)}")
    
    def run_simulation(self, bond: Bond, company: CompanyFinancials) -> Dict[str, Any]:
        """Run Monte Carlo simulation for bond risk analysis"""
        try:
            # Get market data
            market_data = self._get_market_data()
            
            # Generate correlated risk factor paths
            risk_factors = self._generate_risk_factors(market_data)
            
            # Simulate bond prices and yields
            prices, yields = self._simulate_bond_metrics(
                bond, company, risk_factors, market_data
            )
            
            # Calculate risk metrics
            var_95 = self._calculate_var(prices, 0.95)
            var_99 = self._calculate_var(prices, 0.99)
            es = self._calculate_expected_shortfall(prices, 0.95)
            
            # Run stress scenarios
            stress_results = self._run_stress_scenarios(
                bond, company, market_data
            )
            
            # Calculate risk contributions
            contributions = self._calculate_risk_contributions(
                prices, risk_factors
            )
            
            # Calculate correlation matrix
            correlations = self._calculate_correlations(risk_factors)
            
            return {
                'price_distribution': prices.tolist() if hasattr(prices, 'tolist') else prices,
                'yield_distribution': yields.tolist() if hasattr(yields, 'tolist') else yields,
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall': es,
                'stress_scenarios': stress_results,
                'correlation_matrix': correlations.values.tolist() if hasattr(correlations, 'values') else correlations,
                'risk_contributions': contributions
            }
            
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            raise SimulationError(f"Simulation failed: {str(e)}")
    
    def _get_market_data(self) -> Dict[str, Any]:
        """Get market data for simulation"""
        try:
            # Get market data from country manager
            metrics = self.country_manager.get_comprehensive_metrics('USA')  # Use US as default
            
            return {
                'risk_free_rate': metrics['market']['bond_yield'],
                'market_volatility': metrics['market']['market_volatility'],
                'sovereign_spread': metrics['risk']['sovereign_spread'],
                'market_risk': metrics['risk']['market_risk']
            }
            
        except Exception as e:
            logger.warning(f"Error getting market data: {str(e)}")
            # Return default values
            return {
                'risk_free_rate': 0.03,
                'market_volatility': 0.15,
                'sovereign_spread': 0.001,
                'market_risk': 0.15
            }
    
    def _generate_risk_factors(self, market_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate correlated risk factor paths"""
        try:
            # Set up correlation matrix
            corr_matrix = np.array([
                [1.0, self.params['correlation']['rates_spreads'],
                 self.params['correlation']['rates_growth']],
                [self.params['correlation']['rates_spreads'], 1.0,
                 self.params['correlation']['spreads_growth']],
                [self.params['correlation']['rates_growth'],
                 self.params['correlation']['spreads_growth'], 1.0]
            ])
            
            # Generate correlated normal variables
            L = np.linalg.cholesky(corr_matrix)
            Z = np.random.standard_normal((3, self.n_scenarios))
            corr_normals = L @ Z
            
            # Transform to risk factor changes
            rate_changes = (market_data['market_volatility'] * corr_normals[0] *
                          np.sqrt(1/252))  # Daily changes
            spread_changes = (market_data['market_volatility'] * corr_normals[1] *
                            np.sqrt(1/252))
            growth_changes = (0.02 * corr_normals[2] * np.sqrt(1/252))  # 2% annual vol
            
            return {
                'rates': rate_changes,
                'spreads': spread_changes,
                'growth': growth_changes
            }
            
        except Exception as e:
            logger.error(f"Error generating risk factors: {str(e)}")
            raise SimulationError(f"Failed to generate risk factors: {str(e)}")
    
    def _simulate_bond_metrics(
        self,
        bond: Bond,
        company: CompanyFinancials,
        risk_factors: Dict[str, np.ndarray],
        market_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate bond prices and yields"""
        try:
            # Get base metrics
            base_price = bond.current_price
            base_yield = bond.ytm
            duration = bond.duration
            
            # Calculate price changes
            rate_impact = -duration * risk_factors['rates']
            spread_impact = -duration * risk_factors['spreads']
            growth_impact = duration * risk_factors['growth'] * 0.5  # Reduced sensitivity
            
            # Combine impacts
            total_impact = rate_impact + spread_impact + growth_impact
            
            # Calculate new prices and yields
            prices = base_price * (1 + total_impact)
            yields = base_yield + risk_factors['rates'] + risk_factors['spreads']
            
            return prices, yields
            
        except Exception as e:
            logger.error(f"Error simulating bond metrics: {str(e)}")
            raise SimulationError(f"Failed to simulate bond metrics: {str(e)}")
    
    def _calculate_default_intensity(self, company: CompanyFinancials) -> float:
        """Calculate annualized default intensity"""
        try:
            metrics = company.get_credit_metrics()
            coverage = company.get_coverage_ratios()
            
            # Base default intensity from credit metrics
            base_intensity = 0.01  # 1% base rate
            
            # Adjust for leverage
            leverage_factor = max(1.0, metrics.get('leverage_ratio', 3.0) / 3.0)
            
            # Adjust for coverage
            coverage_factor = max(1.0, 2.0 / coverage.get('interest_coverage', 2.0))
            
            # Combined intensity
            intensity = base_intensity * leverage_factor * coverage_factor
            
            return min(1.0, intensity)  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Error calculating default intensity: {str(e)}")
            return 0.02  # Return moderate default intensity
    
    def _calculate_var(self, prices: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk"""
        try:
            return float(np.percentile(prices, (1 - confidence) * 100))
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def _calculate_expected_shortfall(
        self,
        prices: np.ndarray,
        confidence: float
    ) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        try:
            var = self._calculate_var(prices, confidence)
            return float(np.mean(prices[prices <= var]))
        except Exception as e:
            logger.error(f"Error calculating ES: {str(e)}")
            return 0.0
    
    def _run_stress_scenarios(
        self,
        bond: Bond,
        company: CompanyFinancials,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Run predefined stress scenarios"""
        try:
            results = {}
            
            for name, params in self.scenarios.items():
                try:
                    # Apply scenario shocks
                    shocked_data = {
                        'risk_free_rate': market_data['risk_free_rate'] * params.yield_multiplier,
                        'market_volatility': market_data['market_volatility'] * params.volatility_multiplier,
                        'sovereign_spread': market_data['sovereign_spread'] * params.spread_multiplier,
                        'market_risk': market_data['market_risk'] * params.volatility_multiplier
                    }
                    
                    # Calculate stressed price
                    stressed_price = self._stress_test(bond, company, shocked_data)
                    results[name] = stressed_price
                    
                except Exception as e:
                    logger.warning(f"Error in scenario {name}: {str(e)}")
                    results[name] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Error running stress scenarios: {str(e)}")
            return {}
    
    def _stress_test(
        self,
        bond: Bond,
        company: CompanyFinancials,
        shocks: Dict[str, float]
    ) -> float:
        """Apply stress test to bond price"""
        try:
            # Get base metrics
            base_price = bond.current_price
            duration = bond.duration
            convexity = bond.convexity if hasattr(bond, 'convexity') else 0
            
            # Calculate yield change
            d_yield = (shocks['risk_free_rate'] - bond.ytm +
                      shocks['sovereign_spread'])
            
            # First-order approximation with convexity adjustment
            price_change = (-duration * d_yield +
                          0.5 * convexity * d_yield * d_yield)
            
            # Apply credit spread widening based on market stress
            credit_impact = -duration * (
                shocks['market_volatility'] - market_data['market_volatility']
            )
            
            return base_price * (1 + price_change + credit_impact)
            
        except Exception as e:
            logger.error(f"Error in stress test: {str(e)}")
            return bond.current_price  # Return unchanged price on error
    
    def _calculate_risk_contributions(
        self,
        prices: np.ndarray,
        risk_factors: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate risk factor contributions"""
        try:
            # Calculate price returns
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate factor returns
            factor_returns = {
                name: np.diff(factors)
                for name, factors in risk_factors.items()
            }
            
            # Calculate correlations
            correlations = {}
            total_risk = np.std(returns)
            
            for name, factor_return in factor_returns.items():
                corr = np.corrcoef(returns, factor_return)[0, 1]
                beta = corr * np.std(returns) / np.std(factor_return)
                contribution = beta * np.std(factor_return) / total_risk
                correlations[name] = contribution
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating risk contributions: {str(e)}")
            return {name: 0.0 for name in risk_factors.keys()}
    
    def _calculate_correlations(
        self,
        risk_factors: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Calculate correlation matrix between risk factors"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(risk_factors)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return pd.DataFrame()
    
    def run_scenario(
        self,
        company: CompanyFinancials,
        scenario_name: str,
        custom_params: Optional[Dict] = None
    ) -> Dict:
        """Run specific scenario analysis"""
        try:
            # Get scenario parameters
            if custom_params:
                params = ScenarioParams(**custom_params)
            else:
                params = self.scenarios.get(scenario_name, ScenarioParams())
            
            # Get base metrics
            base_metrics = company.get_credit_metrics()
            base_coverage = company.get_coverage_ratios()
            
            # Apply scenario adjustments
            adjusted_metrics = self._apply_scenario_adjustments(
                base_metrics,
                base_coverage,
                params
            )
            
            # Calculate impact
            impact = self._calculate_impact(base_metrics, adjusted_metrics)
            
            return {
                'scenario': scenario_name,
                'base_case': base_metrics,
                'stressed_case': adjusted_metrics,
                'impact': impact,
                'description': self.get_scenario_description(scenario_name)
            }
            
        except Exception as e:
            logger.error(f"Error running scenario {scenario_name}: {str(e)}")
            return {}
    
    def _apply_scenario_adjustments(
        self,
        base_metrics: Dict,
        base_coverage: Dict,
        params: ScenarioParams
    ) -> Dict:
        """Apply scenario adjustments to metrics"""
        try:
            adjusted = {}
            
            # Adjust yields and spreads
            for metric in ['yield', 'spread']:
                if metric in base_metrics:
                    adjusted[metric] = base_metrics[metric] * params.yield_multiplier
            
            # Adjust coverage ratios
            for ratio in base_coverage:
                adjusted[ratio] = base_coverage[ratio] * params.growth_adjustment
            
            # Adjust credit metrics
            adjusted['leverage_ratio'] = (
                base_metrics.get('leverage_ratio', 3.0) / params.growth_adjustment
            )
            
            adjusted['default_probability'] = min(
                1.0,
                base_metrics.get('default_probability', 0.01) *
                params.spread_multiplier / params.growth_adjustment
            )
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error applying scenario adjustments: {str(e)}")
            return {}
    
    def _calculate_impact(self, base: Dict, scenario: Dict) -> Dict:
        """Calculate scenario impact"""
        try:
            return {
                metric: scenario.get(metric, 0) - base.get(metric, 0)
                for metric in set(base.keys()) | set(scenario.keys())
            }
        except Exception as e:
            logger.error(f"Error calculating impact: {str(e)}")
            return {}
    
    def get_scenario_description(self, scenario_name: str) -> str:
        """Get scenario description"""
        descriptions = {
            'base_case': "Base case with current market conditions",
            'stress_test': "Severe market stress with higher rates and spreads",
            'rate_hike': "Interest rate increase scenario",
            'recession': "Economic downturn with credit deterioration",
            'recovery': "Economic recovery with improving credit conditions"
        }
        return descriptions.get(scenario_name, "Custom scenario")