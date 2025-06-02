import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from src.data.country_fetcher import CountryDataManager, IMFClient, WorldBankClient
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class CountryRiskMetrics:
    """Container for country-specific risk metrics"""
    sovereign_spread: float  # Spread over US Treasury
    local_volatility: float  # Local market volatility
    currency_vol: float  # Currency volatility vs USD
    inflation_rate: float  # Current inflation rate
    policy_rate: float  # Central bank policy rate
    debt_to_gdp: float  # Government debt to GDP ratio
    political_risk_score: float = 0.5  # Political risk score (0-1)
    market_liquidity_score: float = 0.7  # Market liquidity score (0-1)
    currency_volatility: float = 0.0  # Currency volatility
    base_interest_rate: float = 0.0  # Base interest rate

class CountryRiskAnalyzer:
    """Analyzes country risk based on various metrics"""
    
    def __init__(self):
        self.data_manager = CountryDataManager()
    
    def analyze_country_risk(self, country_code: str, metrics: Optional[Dict] = None) -> Dict:
        """
        Analyze country risk using provided or fetched metrics
        
        Args:
            country_code: ISO country code
            metrics: Optional pre-calculated metrics (for scenario analysis)
            
        Returns:
            Dictionary containing risk analysis results
        """
        # Use provided metrics or fetch new ones
        if metrics is None:
            metrics = self.data_manager.get_comprehensive_metrics(country_code)
        
        # Calculate individual risk components
        political_risk = self._calculate_political_risk(metrics)
        sovereign_risk = self._calculate_sovereign_risk(metrics)
        market_risk = self._calculate_market_risk(metrics)
        
        # Calculate combined risk score with dynamic weights
        weights = self._calculate_risk_weights(metrics)
        combined_score = (
            weights['political'] * political_risk +
            weights['sovereign'] * sovereign_risk +
            weights['market'] * market_risk
        )
        
        return {
            'political_risk': political_risk,
            'sovereign_risk': sovereign_risk,
            'market_risk': market_risk,
            'combined_risk_score': combined_score,
            'risk_weights': weights
        }
    
    def _calculate_political_risk(self, metrics: Dict) -> float:
        """Calculate political risk score (0-100)"""
        try:
            # Use debt/GDP and current account as proxies for political stability
            debt_factor = min(metrics['macro']['debt_to_gdp'] / 100, 1.0)
            ca_factor = max(-metrics['macro']['current_account'] / 20, 0)  # Negative CA increases risk
            
            # Base political risk score
            base_score = 30 + (debt_factor * 40) + (ca_factor * 30)
            
            # Add volatility adjustment
            vol_adjustment = metrics['risk']['market_volatility'] * 20
            
            return min(max(base_score + vol_adjustment, 0), 100)
        except Exception as e:
            print(f"Error calculating political risk: {e}")
            return 50.0  # Default moderate risk
    
    def _calculate_sovereign_risk(self, metrics: Dict) -> float:
        """Calculate sovereign risk score (0-100)"""
        try:
            # Economic factors
            gdp_factor = max(-metrics.get('macro', {}).get('gdp_growth', 0) / 2, 0)  # Negative growth increases risk
            inflation_factor = metrics.get('macro', {}).get('inflation', 2.5) / 10
            debt_factor = metrics.get('macro', {}).get('debt_to_gdp', 60.0) / 100
            
            # Market factors - convert to basis points and normalize
            spread_factor = min(metrics.get('risk', {}).get('sovereign_spread', 0.02) * 100, 0.25)  # Cap at 25%
            
            # Combined score with progressive weights
            base_score = (
                gdp_factor * 25 +
                inflation_factor * 25 +
                debt_factor * 25 +
                spread_factor * 100  # Convert to percentage
            )
            
            return min(max(base_score, 0), 100)
        except Exception as e:
            logger.error(f"Error calculating sovereign risk: {e}")
            return 50.0
    
    def _calculate_market_risk(self, metrics: Dict) -> float:
        """Calculate market risk score (0-100)"""
        try:
            # Market volatility components (ensure values are in decimal form)
            market_vol = min(metrics.get('risk', {}).get('market_volatility', 0.15), 1.0) * 100
            fx_vol = min(metrics.get('risk', {}).get('fx_volatility', 0.10), 1.0) * 100
            
            # Interest rate risk (normalize rates to percentage)
            rates = metrics.get('market', {}).get('interest_rates', {})
            
            # Default rate values by tenor
            default_rates = {
                '2-Year': 0.025,   # 2.5%
                '5-Year': 0.030,   # 3.0%
                '10-Year': 0.035,  # 3.5%
                '30-Year': 0.040   # 4.0%
            }
            
            # Get valid rates, using defaults for missing values
            valid_rates = []
            for tenor, default_rate in default_rates.items():
                rate = rates.get(tenor)
                if isinstance(rate, (int, float)) and not pd.isna(rate):
                    valid_rates.append(rate)
                else:
                    valid_rates.append(default_rate)
            
            # Calculate average rate level
            rate_level = np.mean(valid_rates) * 100  # Convert to percentage
            
            # Combined score
            base_score = (
                market_vol * 0.3 +  # 30% weight
                fx_vol * 0.3 +      # 30% weight
                rate_level * 0.4    # 40% weight
            )
            
            return min(max(base_score, 0), 100)
        except Exception as e:
            logger.error(f"Error calculating market risk: {e}")
            return 50.0
    
    def _calculate_risk_weights(self, metrics: Dict) -> Dict[str, float]:
        """Calculate dynamic weights for risk components based on market conditions"""
        try:
            # Base weights
            weights = {
                'political': 0.3,
                'sovereign': 0.4,
                'market': 0.3
            }
            
            # Calculate market stress (ensure values are in decimal form)
            market_stress = sum([
                min(metrics.get('risk', {}).get('market_volatility', 0.15), 1.0),
                min(metrics.get('risk', {}).get('fx_volatility', 0.10), 1.0),
                min(metrics.get('risk', {}).get('sovereign_spread', 0.02), 0.10)
            ]) / 3
            
            # In high stress scenarios, increase market weight
            if market_stress > 0.2:  # High stress threshold
                stress_adj = min(market_stress - 0.2, 0.2)
                weights['market'] += stress_adj
                weights['political'] -= stress_adj / 2
                weights['sovereign'] -= stress_adj / 2
            
            # Normalize weights
            total = sum(weights.values())
            return {k: v/total for k, v in weights.items()}
            
        except Exception as e:
            logger.error(f"Error calculating risk weights: {e}")
            return {'political': 0.3, 'sovereign': 0.4, 'market': 0.3}
    
    def get_risk_factors(self, country_code: str) -> Dict[str, float]:
        """Get detailed risk factors for dashboard"""
        metrics = self.data_manager.get_comprehensive_metrics(country_code)
        
        return {
            'gdp_growth': metrics['macro']['gdp_growth'],
            'inflation': metrics['macro']['inflation'],
            'debt_to_gdp': metrics['macro']['debt_to_gdp'],
            'current_account': metrics['macro']['current_account'],
            'market_volatility': metrics['risk']['market_volatility'],
            'fx_volatility': metrics['risk']['fx_volatility'],
            'sovereign_spread': metrics['risk']['sovereign_spread']
        }
    
    def calculate_risk_score(self, country_code: str) -> float:
        """Calculate overall risk score (0-100)"""
        risk_factors = self.get_risk_factors(country_code)
        
        # Define weights for each factor
        weights = {
            'gdp_growth': 0.15,
            'inflation': 0.15,
            'debt_to_gdp': 0.20,
            'current_account': 0.10,
            'market_volatility': 0.15,
            'fx_volatility': 0.10,
            'sovereign_spread': 0.15
        }
        
        # Normalize and weight each factor
        score = 0.0
        for factor, weight in weights.items():
            value = risk_factors[factor]
            
            # Normalize based on typical ranges
            if factor == 'gdp_growth':
                norm_value = max(0, min(100, (value + 10) * 5))  # -10 to +10 range
            elif factor == 'inflation':
                norm_value = max(0, min(100, 100 - value * 10))  # Lower is better
            elif factor == 'debt_to_gdp':
                norm_value = max(0, min(100, 100 - value / 2))  # Lower is better
            elif factor == 'current_account':
                norm_value = max(0, min(100, (value + 10) * 5))  # -10 to +10 range
            elif factor in ['market_volatility', 'fx_volatility']:
                norm_value = max(0, min(100, 100 - value * 100))  # Lower is better
            else:  # sovereign_spread
                norm_value = max(0, min(100, 100 - value * 10))  # Lower is better
            
            score += norm_value * weight
        
        return score
    
    def get_risk_category(self, score: float) -> str:
        """Get risk category based on score"""
        if score >= 80:
            return "Very Low Risk"
        elif score >= 60:
            return "Low Risk"
        elif score >= 40:
            return "Moderate Risk"
        elif score >= 20:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def get_comprehensive_analysis(self, country_code: str) -> Dict:
        """Get comprehensive risk analysis"""
        risk_factors = self.get_risk_factors(country_code)
        risk_score = self.calculate_risk_score(country_code)
        risk_category = self.get_risk_category(risk_score)
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def fetch_country_metrics(self, country_code: str) -> CountryRiskMetrics:
        """Fetch comprehensive country risk metrics with improved error handling"""
        try:
            # Get comprehensive metrics from data manager
            data = self.data_manager.get_comprehensive_metrics(country_code)
            
            # Extract relevant metrics for risk analysis
            macro = data.get('macro', {})
            market = data.get('market', {})
            risk = data.get('risk', {})
            
            # Ensure we have realistic default values
            sovereign_spread = risk.get('sovereign_spread', 0.02)  # 200bps default
            if sovereign_spread == 0:
                sovereign_spread = 0.015  # 150bps minimum for non-US
            
            return CountryRiskMetrics(
                sovereign_spread=sovereign_spread,
                local_volatility=risk.get('market_volatility', 0.15),
                currency_vol=risk.get('fx_volatility', 0.10),
                inflation_rate=macro.get('inflation', 2.5) / 100,
                policy_rate=market.get('policy_rate', 0.05),
                debt_to_gdp=macro.get('debt_to_gdp', 60.0) / 100,
                political_risk_score=risk.get('political_risk', 0.5),
                market_liquidity_score=market.get('liquidity_score', 0.7),
                currency_volatility=risk.get('fx_volatility', 0.10),
                base_interest_rate=market.get('policy_rate', 0.05)
            )
            
        except Exception as e:
            print(f"Error fetching country metrics for {country_code}: {e}")
            # Return realistic defaults for developed markets
            return CountryRiskMetrics(
                sovereign_spread=0.015,  # 150bps spread
                local_volatility=0.15,   # 15% volatility
                currency_vol=0.10,       # 10% FX volatility
                inflation_rate=0.025,    # 2.5% inflation
                policy_rate=0.05,        # 5% policy rate
                debt_to_gdp=0.60,        # 60% debt/GDP
                political_risk_score=0.3, # Low political risk
                market_liquidity_score=0.8, # High liquidity
                currency_volatility=0.10,
                base_interest_rate=0.05
            )
    
    def _calculate_currency_volatility(self, fx_rates: Dict) -> float:
        """Calculate currency volatility from exchange rate data"""
        if not fx_rates:
            return 0.10  # Default volatility if no data available
        
        # Use the latest exchange rate volatility or calculate from historical data
        # This is a simplified implementation - in practice, you'd use historical data
        # to calculate actual volatility
        return 0.10  # Placeholder implementation
    
    def get_country_metrics(self, country_code: str) -> CountryRiskMetrics:
        """Fetch and compute country-specific risk metrics"""
        # TODO: Implement API calls to get real data
        # For now using placeholder logic
        
        # Get US 10Y as baseline
        us_curve = self.data_manager.get_comprehensive_metrics('US')
        us_10y = us_curve['market_data']['interest_rates'].get('10-Year', 0.0)
        
        # Placeholder metrics (to be replaced with API data)
        metrics = CountryRiskMetrics(
            sovereign_spread=0.02,  # 200bps over UST
            local_volatility=0.15,  # 15% annual vol
            currency_vol=0.10,  # 10% FX vol
            inflation_rate=0.04,  # 4% inflation
            policy_rate=0.05,  # 5% policy rate
            debt_to_gdp=0.60,  # 60% debt/GDP
            political_risk_score=0.5,  # Moderate political risk
            market_liquidity_score=0.7,  # Good market liquidity
            currency_volatility=0.10,  # 10% currency volatility
            base_interest_rate=0.05  # 5% base rate
        )
        
        return metrics
    
    def adjust_survival_probability(self,
                                  base_prob: float,
                                  country_metrics: CountryRiskMetrics,
                                  time_horizon: float) -> float:
        """Adjust survival probability based on country risk factors"""
        # Higher spread -> lower survival probability
        spread_impact = np.exp(-country_metrics.sovereign_spread * time_horizon)
        
        # Higher debt/GDP -> lower survival probability
        debt_impact = 1.0 - (country_metrics.debt_to_gdp * 0.2)  # 20% weight
        
        # Higher inflation -> lower survival probability
        inflation_impact = np.exp(-country_metrics.inflation_rate * 0.5)  # 50% dampening
        
        # Combine impacts multiplicatively
        country_adjustment = spread_impact * debt_impact * inflation_impact
        
        return base_prob * country_adjustment
    
    def local_hull_white_params(self,
                              country_metrics: CountryRiskMetrics) -> Tuple[float, float, float]:
        """Adjust Hull-White parameters for local market conditions"""
        # Base parameters
        base_reversion = 0.1
        base_vol = 0.02
        
        # Adjust mean reversion based on policy rate stability
        policy_adjustment = 1.0 + (country_metrics.policy_rate - 0.02) * 2
        adjusted_reversion = base_reversion * policy_adjustment
        
        # Adjust volatility based on local market conditions
        vol_adjustment = country_metrics.local_volatility / 0.1  # Normalized to 10% base
        adjusted_vol = base_vol * vol_adjustment
        
        # Long-term rate includes sovereign spread
        long_term_rate = 0.03 + country_metrics.sovereign_spread  # 3% base + spread
        
        return adjusted_reversion, long_term_rate, adjusted_vol
    
    def simulate_local_rates(self,
                           country_code: str,
                           initial_rate: float,
                           time_steps: int = 252,
                           num_paths: int = 1000) -> np.ndarray:
        """Simulate interest rates in local market context"""
        # Get country risk metrics
        metrics = self.get_country_metrics(country_code)
        
        # Get adjusted Hull-White parameters
        reversion, long_term, vol = self.local_hull_white_params(metrics)
        
        # Simulate with local parameters
        dt = 1/time_steps
        rates = np.zeros((num_paths, time_steps))
        rates[:, 0] = initial_rate
        
        for t in range(1, time_steps):
            drift = reversion * (long_term - rates[:, t-1])
            diffusion = vol * np.random.normal(0, np.sqrt(dt), num_paths)
            rates[:, t] = rates[:, t-1] + drift * dt + diffusion
            
            # Add currency risk premium
            rates[:, t] += metrics.currency_vol * 0.5  # 50% of FX vol as premium
        
        return rates
    
    def value_at_risk_local(self,
                          returns: np.ndarray,
                          country_metrics: CountryRiskMetrics,
                          confidence_level: float = 0.99) -> Tuple[float, float]:
        """Calculate VaR adjusted for local market conditions"""
        # Adjust for higher moments in emerging markets
        if country_metrics.local_volatility > 0.2:  # High vol regime
            # Use Student's t distribution for fat tails
            from scipy.stats import t
            df = 4  # Degrees of freedom for fat tails
            std = np.std(returns)
            var = t.ppf(1 - confidence_level, df) * std
            es = -std * (df + t.ppf(1-confidence_level, df)**2) \
                 * t.pdf(t.ppf(1-confidence_level, df), df) \
                 / ((df-1) * (1-confidence_level))
        else:
            # Use normal distribution for developed markets
            from scipy.stats import norm
            std = np.std(returns)
            var = norm.ppf(1 - confidence_level) * std
            es = -std * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)
        
        return -var, -es  # Convert to positive numbers

    def get_sovereign_spread(self, country_code: str) -> float:
        """Calculate sovereign spread over US Treasury curve"""
        try:
            # Get US Treasury curve as risk-free benchmark
            us_data = self.data_manager.get_comprehensive_metrics('USA')
            us_curve = us_data.get('market', {}).get('interest_rates', {})
            
            # Get country's government bond yields
            country_data = self.data_manager.get_comprehensive_metrics(country_code)
            country_curve = country_data.get('market', {}).get('interest_rates', {})
            
            # Calculate spread at 10Y point (in decimal form)
            us_10y = us_curve.get('10-Year', 0.035)  # 3.5% default
            
            # Use country-specific spreads for default values
            default_spreads = {
                'USA': 0.0,
                'GBR': 0.005,  # 50bps
                'JPN': 0.002,  # 20bps
                'DEU': 0.003,  # 30bps
                'FRA': 0.004,  # 40bps
                'IND': 0.015,  # 150bps
                'CHN': 0.012   # 120bps
            }
            
            # Get default spread for the country or use 100bps
            default_spread = default_spreads.get(country_code, 0.01)
            
            # Calculate country 10Y yield with default spread if not available
            country_10y = country_curve.get('10-Year', us_10y + default_spread)
            
            # Return spread in decimal form (e.g., 0.02 for 200bps)
            spread = max(0.0, min(country_10y - us_10y, 0.15))  # Cap at 1500bps
            
            # Ensure minimum spread for non-US countries
            if country_code != 'USA' and spread < 0.001:  # Minimum 10bps for non-US
                spread = default_spread
            
            return spread
            
        except Exception as e:
            logger.error(f"Error calculating sovereign spread: {e}")
            return default_spreads.get(country_code, 0.01)  # Default to country-specific spread or 100bps