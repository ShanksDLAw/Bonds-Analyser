"""Risk metrics calculation module."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import time

from src.data.country_fetcher import CountryDataFetcher
from src.data.bond_fetcher import BondFetcher
from src.models.company import CompanyFinancials, Bond
from src.config import Config, ConfigError
from src.risk.debt_analyzer import DebtAnalyzer

logger = logging.getLogger(__name__)

class MetricsError(Exception):
    """Custom exception for metrics calculation errors"""
    pass

@dataclass
class BondMetrics:
    """Bond risk metrics data class"""
    credit_risk: float = 0.50  # 50%
    duration_risk: float = 0.30  # 30%
    liquidity_risk: float = 0.40  # 40%
    default_probability: float = 0.00  # 0%
    recovery_rate: float = 0.40  # 40%
    credit_spread: float = 0.001  # 0.1%
    rating_outlook: str = "Stable"

class RiskAnalyzer:
    """Risk analysis engine"""
    
    def __init__(self, company: CompanyFinancials):
        """Initialize analyzer"""
        self.company = company
        self.config = Config()
        self.bond_fetcher = BondFetcher()
        self.country_fetcher = CountryDataFetcher()
        self.debt_analyzer = DebtAnalyzer(company)
        
        # Load real bond data
        self._load_bonds()
    
    def _load_bonds(self):
        """Load real bond data"""
        try:
            self.bonds = self.bond_fetcher.get_company_bonds(self.company.ticker)
            if not self.bonds:
                logger.info(f"No bonds found for {self.company.ticker}, using debt analysis")
        except Exception as e:
            logger.error(f"Error loading bonds: {str(e)}")
            self.bonds = []
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            # Get company metrics
            coverage_ratios = self.company.get_coverage_ratios()
            credit_metrics = self.company.get_credit_metrics()
            debt_analysis = self.debt_analyzer.analyze_debt_capacity()
            
            # Calculate core metrics
            default_prob = self._calculate_default_probability(
                coverage_ratios,
                credit_metrics,
                debt_analysis
            )
            
            yield_spread = self._calculate_yield_spread(
                default_prob,
                credit_metrics.get('credit_score', 50),
                debt_analysis
            )
            
            risk_score = self._calculate_risk_score(
                default_prob,
                yield_spread,
                coverage_ratios,
                credit_metrics,
                debt_analysis
            )
            
            return {
                'default_probability': default_prob,
                'yield_spread': yield_spread,
                'risk_score': risk_score,
                'coverage_ratios': coverage_ratios,
                'credit_metrics': credit_metrics,
                'debt_analysis': debt_analysis,
                'bond_metrics': self._get_bond_metrics() if self.bonds else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return None
    
    def _calculate_default_probability(
        self,
        coverage_ratios: Dict[str, float],
        credit_metrics: Dict[str, Any],
        debt_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate default probability using multiple factors"""
        try:
            # Get key metrics
            interest_coverage = coverage_ratios.get('interest_coverage', 3.5)
            debt_service = coverage_ratios.get('debt_service_coverage', 2.8)
            leverage = credit_metrics.get('leverage_ratio', 3.0)
            operating_margin = credit_metrics.get('operating_margin', 0.15)
            
            # Include debt analysis if available
            if debt_analysis:
                debt_metrics = debt_analysis.get('current_metrics', {})
                debt_to_ebitda = debt_metrics.get('debt_to_ebitda', leverage)
                debt_service = debt_metrics.get('debt_service_coverage', debt_service)
                leverage = max(leverage, debt_to_ebitda)
            
            # Calculate component probabilities
            coverage_prob = max(0.0, min(1.0, (3.0 - interest_coverage) / 3.0))
            service_prob = max(0.0, min(1.0, (2.5 - debt_service) / 2.5))
            leverage_prob = max(0.0, min(1.0, leverage / 10.0))
            margin_prob = max(0.0, min(1.0, (0.15 - operating_margin) / 0.15))
            
            # Weighted average
            weights = {
                'coverage': 0.35,
                'service': 0.25,
                'leverage': 0.25,
                'margin': 0.15
            }
            
            default_prob = (
                coverage_prob * weights['coverage'] +
                service_prob * weights['service'] +
                leverage_prob * weights['leverage'] +
                margin_prob * weights['margin']
            )
            
            # Adjust for debt capacity if available
            if debt_analysis:
                capacity = debt_analysis.get('debt_capacity', {})
                if capacity.get('current_debt', 0) > capacity.get('max_sustainable_debt', float('inf')):
                    default_prob *= 1.2  # Increase probability if exceeding capacity
            
            return min(1.0, max(0.0, default_prob))
            
        except Exception as e:
            logger.error(f"Error calculating default probability: {str(e)}")
            return 0.10  # Default moderate risk
    
    def _calculate_yield_spread(
        self,
        default_prob: float,
        credit_score: int,
        debt_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate yield spread over risk-free rate"""
        try:
            # Base spread components
            default_spread = default_prob * 2000  # Up to 2000bps for default risk
            credit_spread = (100 - credit_score) * 20  # Up to 2000bps for credit risk
            
            # Market conditions adjustment
            market_premium = 100  # Base market premium
            
            # Adjust for debt metrics if available
            if debt_analysis:
                metrics = debt_analysis.get('current_metrics', {})
                benchmarks = debt_analysis.get('industry_benchmarks', {})
                
                # Add premium for high leverage
                if metrics.get('debt_to_ebitda', 0) > benchmarks.get('debt_to_ebitda', {}).get('high_risk', 6.0):
                    market_premium += 50
                
                # Add premium for low coverage
                if metrics.get('interest_coverage', float('inf')) < benchmarks.get('interest_coverage', {}).get('high_risk', 1.5):
                    market_premium += 50
            
            # Calculate total spread
            spread = default_spread + credit_spread + market_premium
            
            # Add liquidity premium for lower-rated bonds
            if credit_score < 50:
                spread *= 1.2  # 20% premium for lower ratings
            
            return max(0.0, spread)
            
        except Exception as e:
            logger.error(f"Error calculating yield spread: {str(e)}")
            return 300  # Default 300bps spread
    
    def _calculate_risk_score(
        self,
        default_prob: float,
        yield_spread: float,
        coverage_ratios: Dict[str, float],
        credit_metrics: Dict[str, Any],
        debt_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate combined risk score (0-100)"""
        try:
            # Component scores
            default_score = 100 * (1 - default_prob)
            spread_score = 100 * (1 - min(1.0, yield_spread / 2000))
            coverage_score = 100 * min(1.0, coverage_ratios.get('interest_coverage', 3.5) / 5.0)
            leverage_score = 100 * (1 - min(1.0, credit_metrics.get('leverage_ratio', 3.0) / 10.0))
            
            # Include debt analysis if available
            if debt_analysis:
                metrics = debt_analysis.get('current_metrics', {})
                capacity = debt_analysis.get('debt_capacity', {})
                
                # Adjust leverage score
                debt_to_ebitda = metrics.get('debt_to_ebitda', 0)
                leverage_score = min(leverage_score, 100 * (1 - min(1.0, debt_to_ebitda / 8.0)))
                
                # Add capacity score
                current_debt = capacity.get('current_debt', 0)
                max_debt = capacity.get('max_sustainable_debt', float('inf'))
                if max_debt > 0:
                    capacity_score = 100 * (1 - min(1.0, current_debt / max_debt))
                else:
                    capacity_score = 0
                
                # Update weights to include capacity
                weights = {
                    'default': 0.30,
                    'spread': 0.20,
                    'coverage': 0.20,
                    'leverage': 0.15,
                    'capacity': 0.15
                }
                
                risk_score = (
                    default_score * weights['default'] +
                    spread_score * weights['spread'] +
                    coverage_score * weights['coverage'] +
                    leverage_score * weights['leverage'] +
                    capacity_score * weights['capacity']
                )
            else:
                # Original weights without capacity
                weights = {
                    'default': 0.35,
                    'spread': 0.25,
                    'coverage': 0.25,
                    'leverage': 0.15
                }
                
                risk_score = (
                    default_score * weights['default'] +
                    spread_score * weights['spread'] +
                    coverage_score * weights['coverage'] +
                    leverage_score * weights['leverage']
                )
            
            return max(0.0, min(100.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 50  # Default moderate score
    
    def _get_bond_metrics(self) -> Dict[str, Any]:
        """Get bond portfolio metrics"""
        try:
            if not self.bonds:
                return {}
            
            metrics = {
                'total_bonds': len(self.bonds),
                'total_value': sum(bond.current_price for bond in self.bonds),
                'average_rating': self._calculate_average_rating(),
                'yield_curve': self._get_yield_curve(),
                'duration_profile': self._get_duration_profile(),
                'bonds': []
            }
            
            # Add individual bond details
            for bond in self.bonds:
                metrics['bonds'].append({
                    'isin': bond.isin,
                    'maturity': bond.maturity_date.strftime('%Y-%m-%d'),
                    'coupon': bond.coupon_rate,
                    'ytm': bond.ytm,
                    'rating': bond.credit_rating,
                    'duration': bond.duration,
                    'price': bond.current_price
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting bond metrics: {str(e)}")
            return {}
    
    def _calculate_average_rating(self) -> str:
        """Calculate weighted average credit rating"""
        if not self.bonds:
            return 'NR'
        
        # Rating scores (lower is better)
        rating_scores = {
            'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
            'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10,
            'BB+': 11, 'BB': 12, 'BB-': 13,
            'B+': 14, 'B': 15, 'B-': 16,
            'CCC+': 17, 'CCC': 18, 'CCC-': 19,
            'CC': 20, 'C': 21, 'D': 22
        }
        
        # Calculate weighted average
        total_value = sum(bond.current_price for bond in self.bonds)
        weighted_score = sum(
            rating_scores.get(bond.credit_rating, 9) * bond.current_price
            for bond in self.bonds
        ) / total_value
        
        # Convert back to rating
        score_to_rating = {v: k for k, v in rating_scores.items()}
        closest_score = min(rating_scores.values(), key=lambda x: abs(x - weighted_score))
        
        return score_to_rating[closest_score]
    
    def _get_yield_curve(self) -> Dict[str, float]:
        """Get yield curve from bond portfolio"""
        if not self.bonds:
            return {}
        
        curve = {}
        for bond in self.bonds:
            years_to_maturity = (bond.maturity_date - datetime.now()).days / 365.25
            key = f"{int(years_to_maturity)}Y"
            if key not in curve or bond.ytm > curve[key]:
                curve[key] = bond.ytm
        
        return dict(sorted(curve.items()))
    
    def _get_duration_profile(self) -> Dict[str, float]:
        """Get duration profile of bond portfolio"""
        if not self.bonds:
            return {}
        
        total_value = sum(bond.current_price for bond in self.bonds)
        profile = {
            'short_term': 0.0,    # 0-3 years
            'medium_term': 0.0,   # 3-7 years
            'long_term': 0.0      # 7+ years
        }
        
        for bond in self.bonds:
            weight = bond.current_price / total_value
            if bond.duration < 3:
                profile['short_term'] += weight
            elif bond.duration < 7:
                profile['medium_term'] += weight
            else:
                profile['long_term'] += weight
        
        return profile

    def calculate_risk_metrics(self, company: CompanyFinancials, country_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            # Get company metrics
            credit_metrics = company.get_credit_metrics()
            coverage_ratios = company.get_coverage_ratios()
            portfolio_metrics = company.get_bond_portfolio_metrics()
            
            # Get or fetch country data
            if not country_data:
                country_data = self.country_fetcher.get_comprehensive_metrics('USA')  # Default to USA
            
            # Calculate individual risk components
            credit_risk = self._calculate_credit_risk(
                credit_metrics,
                coverage_ratios,
                country_data
            )
            
            duration_risk = self._calculate_duration_risk(
                portfolio_metrics,
                country_data
            )
            
            liquidity_risk = self._calculate_liquidity_risk(
                portfolio_metrics,
                country_data
            )
            
            # Calculate combined risk score
            combined_risk = (
                self.weights['credit_risk'] * credit_risk +
                self.weights['duration_risk'] * duration_risk +
                self.weights['liquidity_risk'] * liquidity_risk
            ) / sum(self.weights.values())
            
            return {
                'credit_risk': credit_risk,
                'duration_risk': duration_risk,
                'liquidity_risk': liquidity_risk,
                'combined_risk': combined_risk,
                'risk_components': {
                    'default_probability': credit_metrics['default_probability'],
                    'recovery_rate': 0.40,  # Industry standard assumption
                    'duration_score': duration_risk,
                    'liquidity_score': liquidity_risk,
                    'country_risk': self._calculate_country_risk(country_data)
                }
            }
            
        except Exception as e:
            print(f"Error calculating risk metrics: {str(e)}")
            return {
                'credit_risk': 0.5,
                'duration_risk': 0.5,
                'liquidity_risk': 0.5,
                'combined_risk': 0.5,
                'risk_components': {
                    'default_probability': 0.01,
                    'recovery_rate': 0.40,
                    'duration_score': 0.5,
                    'liquidity_score': 0.5,
                    'country_risk': 0.5
                }
            }
    
    def _calculate_credit_risk(
        self,
        credit_metrics: Dict[str, Any],
        coverage_ratios: Dict[str, float],
        country_data: Dict[str, Any]
    ) -> float:
        """Calculate credit risk score (0-1)"""
        try:
            # Get metrics
            default_prob = credit_metrics['default_probability']
            leverage = credit_metrics['leverage_ratio']
            debt_to_assets = credit_metrics['debt_to_assets']
            operating_margin = credit_metrics['operating_margin']
            
            # Get coverage ratios
            interest_coverage = coverage_ratios['interest_coverage']
            debt_service = coverage_ratios['debt_service_coverage']
            
            # Get country risk factors
            country_risk = self._calculate_country_risk(country_data)
            
            # Calculate component scores
            default_score = min(1.0, default_prob * 5)  # Scale up default probability
            leverage_score = min(1.0, leverage / 10)  # Assume 10x leverage is maximum risk
            coverage_score = max(0.0, min(1.0, 2.0 / interest_coverage))  # 2x coverage as benchmark
            country_factor = 0.5 + (country_risk * 0.5)  # Country risk as multiplier
            
            # Combined score with weights
            credit_risk = (
                0.4 * default_score +
                0.3 * leverage_score +
                0.3 * coverage_score
            ) * country_factor
            
            return min(1.0, max(0.0, credit_risk))
            
        except Exception as e:
            print(f"Error calculating credit risk: {str(e)}")
            return 0.5
    
    def _calculate_duration_risk(
        self,
        portfolio_metrics: Dict[str, Any],
        country_data: Dict[str, Any]
    ) -> float:
        """Calculate duration risk score (0-1)"""
        try:
            # Get metrics
            duration = portfolio_metrics.get('weighted_duration', 5.0)
            convexity = portfolio_metrics.get('weighted_convexity', 0.5)
            
            # Get market environment
            rates_environment = country_data.get('market', {}).get('interest_rates', {})
            rate_volatility = 0.15  # Assumed annual rate volatility
            
            # Calculate base duration score
            duration_score = min(1.0, duration / 10)  # 10 years as maximum risk
            
            # Adjust for convexity
            convexity_adjustment = max(0.0, min(0.2, convexity * 0.1))
            
            # Adjust for rate environment
            rate_level = float(rates_environment.get('10Y', 0.035))
            rate_risk = min(1.0, rate_level * 10)  # Higher rates = higher risk
            
            # Combined score
            duration_risk = (
                0.6 * duration_score +
                0.2 * rate_risk
            ) - convexity_adjustment
            
            return min(1.0, max(0.0, duration_risk))
            
        except Exception as e:
            print(f"Error calculating duration risk: {str(e)}")
            return 0.5
    
    def _calculate_liquidity_risk(
        self,
        portfolio_metrics: Dict[str, Any],
        country_data: Dict[str, Any]
    ) -> float:
        """Calculate liquidity risk score (0-1)"""
        try:
            # Get metrics
            market_liquidity = country_data.get('market', {}).get('bond_market_liquidity', 0.7)
            
            # Get portfolio characteristics
            num_bonds = portfolio_metrics.get('number_of_bonds', 0)
            total_value = portfolio_metrics.get('total_value', 0)
            
            # Calculate component scores
            size_score = max(0.0, min(1.0, 1.0 - (total_value / 1000000000)))  # $1B as benchmark
            diversity_score = max(0.0, min(1.0, 1.0 - (num_bonds / 10)))  # 10 bonds as benchmark
            market_score = 1.0 - market_liquidity
            
            # Combined score
            liquidity_risk = (
                0.4 * size_score +
                0.3 * diversity_score +
                0.3 * market_score
            )
            
            return min(1.0, max(0.0, liquidity_risk))
            
        except Exception as e:
            print(f"Error calculating liquidity risk: {str(e)}")
            return 0.5
    
    def _calculate_country_risk(self, country_data: Dict[str, Any]) -> float:
        """Calculate country risk score (0-1)"""
        try:
            # Get metrics
            macro = country_data.get('macro', {})
            risk = country_data.get('risk', {})
            
            # Extract key metrics
            gdp_growth = float(macro.get('gdp_growth', 2.0)) / 100
            inflation = float(macro.get('inflation', 2.0)) / 100
            debt_gdp = float(macro.get('debt_to_gdp', 60.0)) / 100
            
            # Calculate component scores
            growth_risk = max(0.0, min(1.0, (2.0 - gdp_growth) / 4.0))
            inflation_risk = max(0.0, min(1.0, abs(inflation - 0.02) / 0.05))
            debt_risk = max(0.0, min(1.0, debt_gdp / 1.2))
            
            # Get pre-calculated risks
            economic_risk = risk.get('economic_risk', 0.5)
            political_risk = risk.get('political_risk', 0.5)
            financial_risk = risk.get('financial_risk', 0.5)
            
            # Combined score
            country_risk = (
                0.3 * economic_risk +
                0.3 * political_risk +
                0.2 * financial_risk +
                0.1 * growth_risk +
                0.1 * debt_risk
            )
            
            return min(1.0, max(0.0, country_risk))
            
        except Exception as e:
            print(f"Error calculating country risk: {str(e)}")
            return 0.5
    
    def calculate_scenario_impact(
        self,
        company: CompanyFinancials,
        scenario: str,
        adjustments: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate impact of scenario on risk metrics"""
        try:
            # Get base metrics
            base_metrics = self.calculate_risk_metrics(company)
            
            # Define scenario adjustments
            scenarios = {
                'stress_test': {
                    'yield_multiplier': 2.0,
                    'spread_multiplier': 2.0,
                    'volatility_multiplier': 1.5,
                    'growth_adjustment': 0.5
                },
                'optimistic': {
                    'yield_multiplier': 0.8,
                    'spread_multiplier': 0.7,
                    'volatility_multiplier': 0.8,
                    'growth_adjustment': 1.2
                }
            }
            
            # Use custom adjustments if provided
            scenario_adjustments = adjustments if adjustments else scenarios.get(scenario, {})
            
            # Apply adjustments
            adjusted_metrics = {
                'credit_risk': base_metrics['credit_risk'] * scenario_adjustments.get('spread_multiplier', 1.0),
                'duration_risk': base_metrics['duration_risk'] * scenario_adjustments.get('yield_multiplier', 1.0),
                'liquidity_risk': base_metrics['liquidity_risk'] * scenario_adjustments.get('volatility_multiplier', 1.0)
            }
            
            # Recalculate combined risk
            adjusted_metrics['combined_risk'] = (
                self.weights['credit_risk'] * adjusted_metrics['credit_risk'] +
                self.weights['duration_risk'] * adjusted_metrics['duration_risk'] +
                self.weights['liquidity_risk'] * adjusted_metrics['liquidity_risk']
            ) / sum(self.weights.values())
            
            return {
                'base_case': base_metrics,
                'scenario': adjusted_metrics,
                'impact': {
                    metric: adjusted_metrics[metric] - base_metrics[metric]
                    for metric in ['credit_risk', 'duration_risk', 'liquidity_risk', 'combined_risk']
                }
            }
            
        except Exception as e:
            print(f"Error calculating scenario impact: {str(e)}")
            return {
                'base_case': {},
                'scenario': {},
                'impact': {}
            }
    
    def get_risk_free_rate(self, maturity: str = '10Y') -> float:
        """Get risk-free rate from Treasury data based on maturity"""
        try:
            metrics = self.country_fetcher.get_comprehensive_metrics('USA')
            return metrics['market']['interest_rates'].get(maturity, 0.0)
        except Exception as e:
            logger.warning(f"Error getting risk-free rate: {str(e)}")
            return 0.0
    
    @staticmethod
    def yield_to_maturity(
        price: float,
        par_value: float,
        coupon_rate: float,
        years_to_maturity: float,
        frequency: int = 2
    ) -> float:
        """Calculate YTM using Newton-Raphson method"""
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [price, par_value, coupon_rate, years_to_maturity]):
                logger.warning("Invalid input types for YTM calculation")
                return 0.05  # Return default YTM if inputs are invalid
            
            if price <= 0 or par_value <= 0 or years_to_maturity <= 0:
                logger.warning("Non-positive inputs for YTM calculation")
                return 0.05  # Return default YTM if inputs are non-positive
            
            periods = frequency * years_to_maturity
            dt = [(i+1)/frequency for i in range(int(periods))]
            coupon = (coupon_rate/frequency) * par_value

            def ytm_func(y):
                try:
                    return sum([coupon / (1 + y/frequency)**(frequency*t)
                            for t in dt]) + \
                        par_value / (1 + y/frequency)**(frequency*years_to_maturity) - price
                except (ZeroDivisionError, OverflowError):
                    return float('inf')

            # Increased maxiter and adjusted tolerance for better convergence
            result = newton(ytm_func, x0=0.05, maxiter=1000, tol=1e-4, rtol=1e-4)
            
            # Bound result between 0% and 100%
            return max(0.0, min(1.0, result))
            
        except Exception as e:
            logger.warning(f"Error calculating YTM: {str(e)}")
            return 0.05  # Return default YTM on error

    @staticmethod
    def modified_duration(cash_flows: List[float],
                        times: List[float],
                        ytm: float) -> float:
        """Calculate modified duration for semi-annual bonds"""
        try:
            if not cash_flows or not times:
                logger.warning("Empty cash flows or times for duration calculation")
                return 0.0
                
            if ytm <= -2:  # Prevent negative rates that would cause invalid calculations
                logger.warning("Invalid YTM for duration calculation")
                return 0.0
                
            semi_rate = ytm / 2
            pv_factors = [(1 + semi_rate)**(-2 * t) if t > 0 else 1 
                         for t in times]
            pv_cashflows = [cf * pv for cf, pv in zip(cash_flows, pv_factors)]
            total_pv = sum(pv_cashflows)
            
            if total_pv == 0:
                logger.warning("Zero present value in duration calculation")
                return 0.0
                
            macaulay_duration = sum(t * pv_cf for t, pv_cf in zip(times, pv_cashflows)) / total_pv
            
            if semi_rate <= -1:
                logger.warning("Invalid semi-annual rate for modified duration")
                return 0.0
                
            return macaulay_duration / (1 + semi_rate)
            
        except Exception as e:
            logger.warning(f"Error calculating modified duration: {str(e)}")
            return 0.0

    def calculate_market_risk(self, market_data: Dict) -> float:
        """Calculate market risk score"""
        volatility = market_data.get('volatility', 0.2)
        beta = market_data.get('beta', 1.0)
        interest_rate_sensitivity = market_data.get('interest_rate_sensitivity', 0.5)
        
        # Combine factors with weights
        market_risk = (
            0.4 * min(1.0, volatility * 5) +  # Scale volatility
            0.3 * min(1.0, beta) +
            0.3 * min(1.0, interest_rate_sensitivity * 2)
        )
        
        return min(1.0, market_risk)
    
    def calculate_liquidity_risk(self, trading_data: Dict) -> float:
        """Calculate liquidity risk score"""
        volume = trading_data.get('volume', 1000000)
        bid_ask_spread = trading_data.get('bid_ask_spread', 0.01)
        days_to_maturity = trading_data.get('days_to_maturity', 365)
        
        # Volume score (lower volume = higher risk)
        volume_score = 1 - min(1.0, volume / 10000000)
        
        # Spread score (higher spread = higher risk)
        spread_score = min(1.0, bid_ask_spread * 100)
        
        # Maturity score (longer maturity = higher risk)
        maturity_score = min(1.0, days_to_maturity / 3650)  # Scale to 10 years
        
        # Combine scores
        liquidity_risk = (
            0.4 * volume_score +
            0.4 * spread_score +
            0.2 * maturity_score
        )
        
        return min(1.0, liquidity_risk)
    
    def calculate_duration_risk(self, duration: float, yield_volatility: float) -> float:
        """Calculate duration risk score"""
        # Normalize duration (assuming 10-year max duration)
        normalized_duration = min(1.0, duration / 10)
        
        # Scale yield volatility (assuming 2% max volatility)
        normalized_volatility = min(1.0, yield_volatility / 0.02)
        
        # Combine for total duration risk
        duration_risk = normalized_duration * (0.7 + 0.3 * normalized_volatility)
        
        return min(1.0, duration_risk)
    
    def calculate_combined_risk_score(self, metrics: Dict) -> float:
        """Calculate combined risk score from individual components"""
        weighted_sum = sum(
            self.weights[risk_type] * score
            for risk_type, score in metrics.items()
            if risk_type in self.weights
        )
        
        return min(1.0, weighted_sum)
    
    def get_risk_breakdown(self, metrics: Dict) -> Dict[str, float]:
        """Get contribution of each risk factor to total risk"""
        total_risk = self.calculate_combined_risk_score(metrics)
        
        if total_risk == 0:
            return {risk_type: 0.0 for risk_type in self.weights}
        
        return {
            risk_type: (self.weights[risk_type] * metrics[risk_type]) / total_risk
            for risk_type in self.weights
            if risk_type in metrics
        }
    
    def get_risk_recommendations(self, metrics: Dict) -> List[str]:
        """Generate risk management recommendations based on metrics"""
        recommendations = []
        breakdown = self.get_risk_breakdown(metrics)
        
        # Add recommendations based on highest risk factors
        for risk_type, contribution in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            if contribution > 0.3:  # Significant contribution
                if risk_type == 'credit_risk':
                    recommendations.append(
                        "Consider credit enhancement or guarantees to mitigate high credit risk."
                    )
                elif risk_type == 'market_risk':
                    recommendations.append(
                        "Implement hedging strategies to reduce market risk exposure."
                    )
                elif risk_type == 'liquidity_risk':
                    recommendations.append(
                        "Consider increasing position in more liquid bonds or maintaining higher cash reserves."
                    )
                elif risk_type == 'duration_risk':
                    recommendations.append(
                        "Review duration strategy and consider shorter-term alternatives."
                    )
        
        return recommendations