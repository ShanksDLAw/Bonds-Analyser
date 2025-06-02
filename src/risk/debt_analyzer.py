"""Debt analysis module for corporate bonds."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from src.models.company import CompanyFinancials
from src.data.country_fetcher import CountryDataFetcher

logger = logging.getLogger(__name__)

class DebtAnalyzer:
    """Analyzes company debt capacity and sustainability"""
    
    def __init__(self, company: CompanyFinancials):
        """Initialize analyzer with company data"""
        self.company = company
        self.country_fetcher = CountryDataFetcher()
        
        # Industry benchmarks
        self.benchmarks = {
            'debt_to_ebitda': {
                'low_risk': 3.0,
                'medium_risk': 4.5,
                'high_risk': 6.0
            },
            'debt_to_assets': {
                'low_risk': 0.3,
                'medium_risk': 0.5,
                'high_risk': 0.7
            },
            'debt_service_coverage': {
                'low_risk': 2.0,
                'medium_risk': 1.5,
                'high_risk': 1.2
            },
            'interest_coverage': {
                'low_risk': 4.0,
                'medium_risk': 2.5,
                'high_risk': 1.5
            }
        }
    
    def analyze_debt_profile(self) -> Dict[str, Any]:
        """Analyze company debt profile with market-based metrics"""
        try:
            # Get real-time market data
            market_data = self.company.get_market_data()
            
            # Calculate market-implied metrics
            implied_metrics = self._calculate_market_implied_metrics(market_data)
            
            # Get macro environment data
            macro_data = self._get_macro_environment()
            
            # Calculate adjusted metrics based on macro environment
            adjusted_metrics = self._adjust_metrics_for_macro(implied_metrics, macro_data)
            
            # Generate comprehensive risk assessment
            risk_assessment = self._assess_comprehensive_risk(adjusted_metrics)
            
            return {
                'market_implied_metrics': implied_metrics,
                'macro_adjusted_metrics': adjusted_metrics,
                'risk_assessment': risk_assessment
            }
        except Exception as e:
            logger.error(f"Error analyzing debt profile: {str(e)}")
            raise
    
    def _calculate_current_metrics(
        self,
        balance_sheet: Dict,
        income_stmt: Dict,
        cash_flow: Dict
    ) -> Dict[str, float]:
        """Calculate current debt metrics"""
        try:
            # Get key financial items with defaults
            total_debt = balance_sheet.get('total_debt', 0)
            short_term_debt = balance_sheet.get('short_term_debt', total_debt * 0.3 if total_debt > 0 else 0)
            long_term_debt = balance_sheet.get('long_term_debt', total_debt * 0.7 if total_debt > 0 else 0)
            total_assets = balance_sheet.get('total_assets', total_debt * 2 if total_debt > 0 else 0)
            ebitda = income_stmt.get('ebitda', total_assets * 0.15 if total_assets > 0 else 0)
            interest_expense = income_stmt.get('interest_expense', total_debt * 0.05 if total_debt > 0 else 0)
            operating_cash_flow = cash_flow.get('operating_cash_flow')
            if operating_cash_flow is None or operating_cash_flow <= 0:
                # Estimate OCF as 80% of EBITDA if not available or negative
                operating_cash_flow = ebitda * 0.8 if ebitda > 0 else total_assets * 0.12
            net_income = income_stmt.get('net_income', ebitda * 0.6 if ebitda > 0 else 0)
            total_equity = balance_sheet.get('total_equity', total_assets * 0.4 if total_assets > 0 else 0)
            current_assets = balance_sheet.get('current_assets', total_assets * 0.3 if total_assets > 0 else 0)
            current_liabilities = balance_sheet.get('current_liabilities', total_debt * 0.3 if total_debt > 0 else 0)
            revenue = income_stmt.get('revenue', ebitda * 5 if ebitda > 0 else 0)
            
            # Calculate core metrics
            metrics = {
                # Leverage metrics
                'total_debt': total_debt,
                'short_term_debt': short_term_debt,
                'long_term_debt': long_term_debt,
                'debt_to_ebitda': total_debt / ebitda if ebitda > 0 else 6.0,  # Default to high risk
                'debt_to_assets': total_debt / total_assets if total_assets > 0 else 0.7,  # Default to high risk
                'debt_to_equity': total_debt / total_equity if total_equity > 0 else 2.0,  # Default to high risk
                'debt_to_revenue': total_debt / revenue if revenue > 0 else 1.0,  # Default to high risk
                'debt_to_fcf': total_debt / operating_cash_flow if operating_cash_flow > 0 else 8.0,  # Default to high risk
                
                # Coverage metrics
                'interest_coverage': ebitda / interest_expense if interest_expense > 0 else 2.0,  # Default to medium risk
                'debt_service_coverage': operating_cash_flow / (interest_expense + short_term_debt) if (interest_expense + short_term_debt) > 0 else 1.5,  # Default to medium risk
                'fixed_charge_coverage': (ebitda + interest_expense) / (interest_expense * 1.2) if interest_expense > 0 else 2.0,  # Default to medium risk
                'cash_coverage_ratio': operating_cash_flow / total_debt if total_debt > 0 else 0.2,  # Default to medium risk
                
                # Structure metrics
                'short_term_ratio': short_term_debt / total_debt if total_debt > 0 else 0.3,  # Default to typical ratio
                'secured_debt_ratio': balance_sheet.get('secured_debt', total_debt * 0.4) / total_debt if total_debt > 0 else 0.4,  # Default to typical ratio
                'variable_rate_debt_ratio': balance_sheet.get('variable_rate_debt', total_debt * 0.3) / total_debt if total_debt > 0 else 0.3,  # Default to typical ratio
                
                # Profitability metrics
                'operating_margin': ebitda / revenue if revenue > 0 else 0.15,  # Default to typical margin
                'net_margin': net_income / revenue if revenue > 0 else 0.08,  # Default to typical margin
                'return_on_assets': net_income / total_assets if total_assets > 0 else 0.06,  # Default to typical ROA
                
                # Liquidity metrics
                'current_ratio': current_assets / current_liabilities if current_liabilities > 0 else 1.5,  # Default to typical ratio
                'quick_ratio': (current_assets - balance_sheet.get('inventory', current_assets * 0.3)) / current_liabilities if current_liabilities > 0 else 1.2,  # Default to typical ratio
                'cash_ratio': balance_sheet.get('cash', current_assets * 0.2) / current_liabilities if current_liabilities > 0 else 0.3  # Default to typical ratio
            }
            
            # Calculate risk scores
            metrics.update({
                'credit_risk': self._calculate_credit_risk(metrics),
                'duration_risk': self._calculate_duration_risk(metrics),
                'refinancing_risk': self._calculate_refinancing_risk(metrics)
            })
            
            # Cap infinite values
            for key, value in metrics.items():
                if isinstance(value, float) and (value == float('inf') or value == float('-inf')):
                    metrics[key] = 10.0  # Cap at 10x for ratios
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating current metrics: {str(e)}")
            return {
                'total_debt': 0,
                'short_term_debt': 0,
                'long_term_debt': 0,
                'debt_to_ebitda': 6.0,
                'debt_to_assets': 0.7,
                'debt_to_equity': 2.0,
                'interest_coverage': 2.0,
                'debt_service_coverage': 1.5,
                'fixed_charge_coverage': 2.0,
                'operating_margin': 0.15,
                'credit_risk': 0.5,
                'duration_risk': 0.5,
                'refinancing_risk': 0.5
            }
    
    def _calculate_credit_risk(self, metrics: Dict[str, float]) -> float:
        """Calculate credit risk score"""
        try:
            # Weight different factors
            weights = {
                'debt_to_ebitda': 0.3,
                'interest_coverage': 0.2,
                'debt_to_assets': 0.2,
                'return_on_assets': 0.15,
                'current_ratio': 0.15
            }
            
            # Calculate component scores (0-1, higher is riskier)
            scores = {
                'debt_to_ebitda': min(1.0, metrics.get('debt_to_ebitda', 0) / 6.0),
                'interest_coverage': min(1.0, 1.0 / metrics.get('interest_coverage', 1.0)),
                'debt_to_assets': min(1.0, metrics.get('debt_to_assets', 0)),
                'return_on_assets': min(1.0, max(0, 0.1 - metrics.get('return_on_assets', 0)) / 0.1),
                'current_ratio': min(1.0, 2.0 / metrics.get('current_ratio', 2.0))
            }
            
            # Calculate weighted average
            risk_score = sum(
                weights[metric] * scores[metric]
                for metric in weights.keys()
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating credit risk: {str(e)}")
            return 0.5
    
    def _calculate_duration_risk(self, metrics: Dict[str, float]) -> float:
        """Calculate duration risk score"""
        try:
            # Weight different factors
            weights = {
                'short_term_ratio': 0.4,
                'debt_service_coverage': 0.3,
                'interest_coverage': 0.3
            }
            
            # Calculate component scores (0-1, higher is riskier)
            scores = {
                'short_term_ratio': min(1.0, metrics.get('short_term_ratio', 0) * 2),
                'debt_service_coverage': min(1.0, 1.0 / metrics.get('debt_service_coverage', 1.0)),
                'interest_coverage': min(1.0, 1.0 / metrics.get('interest_coverage', 1.0))
            }
            
            # Calculate weighted average
            risk_score = sum(
                weights[metric] * scores[metric]
                for metric in weights.keys()
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating duration risk: {str(e)}")
            return 0.3
    
    def _calculate_liquidity_risk(self, metrics: Dict[str, float]) -> float:
        """Calculate liquidity risk score"""
        try:
            # Weight different factors
            weights = {
                'current_ratio': 0.4,
                'quick_ratio': 0.3,
                'cash_ratio': 0.3
            }
            
            # Calculate component scores (0-1, higher is riskier)
            scores = {
                'current_ratio': min(1.0, 2.0 / metrics.get('current_ratio', 2.0)),
                'quick_ratio': min(1.0, 1.0 / metrics.get('quick_ratio', 1.0)),
                'cash_ratio': min(1.0, 0.5 / metrics.get('cash_ratio', 0.5))
            }
            
            # Calculate weighted average
            risk_score = sum(
                weights[metric] * scores[metric]
                for metric in weights.keys()
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {str(e)}")
            return 0.4
    
    def _get_industry_benchmarks(self, industry: str) -> Dict[str, Any]:
        try:
            # Fetch industry benchmarks from market data provider
            industry_data = self.market_data_provider.get_industry_metrics(industry)
            return {
                'debt_to_ebitda': industry_data.get('median_debt_to_ebitda', {}),
                'debt_to_assets': industry_data.get('median_debt_to_assets', {}),
                'interest_coverage': industry_data.get('median_interest_coverage', {})
            }
        except Exception as e:
            logger.error(f"Error fetching industry benchmarks: {str(e)}")
            return {}
    
    def _get_industry_factor(self) -> float:
        """Get industry adjustment factor"""
        try:
            # Higher factor means more debt tolerance
            industry_factors = {
                'Utilities': 1.3,
                'Real Estate': 1.2,
                'Telecommunications': 1.1,
                'Consumer Staples': 1.0,
                'Healthcare': 0.9,
                'Technology': 0.8,
                'Consumer Discretionary': 0.9
            }
            
            return industry_factors.get(self.company.industry, 1.0)
            
        except Exception:
            return 1.0
    
    def _get_country_factor(self, country_data: Dict[str, Any]) -> float:
        """Get country adjustment factor"""
        try:
            # Calculate based on sovereign rating and market development
            rating_factor = {
                'AAA': 1.2,
                'AA': 1.1,
                'A': 1.0,
                'BBB': 0.9,
                'BB': 0.8,
                'B': 0.7
            }.get(country_data.get('rating', 'BBB'), 0.9)
            
            market_factor = country_data.get('market_data', {}).get('bond_market_liquidity', 1.0)
            
            return rating_factor * market_factor
            
        except Exception:
            return 1.0
    
    def _calculate_debt_capacity(
        self,
        current_metrics: Dict[str, float],
        balance_sheet: Dict,
        income_stmt: Dict
    ) -> Dict[str, float]:
        """Calculate debt capacity"""
        try:
            # Get EBITDA and cash flow projections
            ebitda_growth = self._project_ebitda_growth(income_stmt)
            fcf_growth = self._project_fcf_growth()
            
            # Calculate capacity based on different metrics
            ebitda_capacity = current_metrics.get('ebitda', 0) * self.benchmarks['debt_to_ebitda']['medium_risk']
            asset_capacity = current_metrics.get('total_assets', 0) * self.benchmarks['debt_to_assets']['medium_risk']
            
            # Calculate sustainable debt level
            sustainable_debt = min(ebitda_capacity, asset_capacity)
            
            # Calculate additional debt capacity
            additional_capacity = max(0, sustainable_debt - current_metrics.get('total_debt', 0))
            
            # Calculate optimal debt structure
            optimal_structure = self._calculate_optimal_structure(
                sustainable_debt, current_metrics, balance_sheet
            )
            
            return {
                'sustainable_debt_level': sustainable_debt,
                'additional_capacity': additional_capacity,
                'ebitda_based_capacity': ebitda_capacity,
                'asset_based_capacity': asset_capacity,
                'ebitda_growth_projection': ebitda_growth,
                'fcf_growth_projection': fcf_growth,
                'optimal_structure': optimal_structure
            }
            
        except Exception as e:
            logger.error(f"Error calculating debt capacity: {str(e)}")
            return {}
    
    def _calculate_optimal_structure(
        self,
        target_debt: float,
        current_metrics: Dict[str, float],
        balance_sheet: Dict
    ) -> Dict[str, float]:
        """Calculate optimal debt structure"""
        try:
            # Get market conditions
            market_data = self.country_fetcher.get_comprehensive_metrics(
                self.company.country_code
            ).get('market_data', {})
            
            # Calculate optimal term structure with safe default for operating_cash_flow
            operating_cash_flow = current_metrics.get('operating_cash_flow', 0)
            if operating_cash_flow <= 0:
                # Estimate operating cash flow as percentage of EBITDA if not available
                ebitda = current_metrics.get('ebitda', 0)
                operating_cash_flow = ebitda * 0.8  # Assume 80% conversion rate
            
            # Calculate optimal term structure
            short_term_optimal = min(
                0.3 * target_debt,  # Max 30% short-term
                operating_cash_flow * 0.5  # Coverage constraint
            )
            
            long_term_optimal = target_debt - short_term_optimal
            
            # Calculate optimal fixed vs floating mix
            fixed_rate_portion = min(
                0.8,  # Max 80% fixed
                max(0.5, 1 - market_data.get('market_volatility', 0.15))  # Higher volatility = more fixed
            )
            
            return {
                'short_term_debt': short_term_optimal,
                'long_term_debt': long_term_optimal,
                'fixed_rate_portion': fixed_rate_portion,
                'floating_rate_portion': 1 - fixed_rate_portion
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal structure: {str(e)}")
            return {}
    
    def _project_ebitda_growth(self, income_stmt: Dict) -> float:
        """Project EBITDA growth"""
        try:
            # Get historical EBITDA
            historical_ebitda = income_stmt.get('historical_ebitda', [])
            
            if len(historical_ebitda) >= 2:
                # Calculate growth rate
                growth_rates = []
                for i in range(1, len(historical_ebitda)):
                    if historical_ebitda[i-1] > 0:
                        growth = (historical_ebitda[i] - historical_ebitda[i-1]) / historical_ebitda[i-1]
                        growth_rates.append(growth)
                
                if growth_rates:
                    # Use weighted average of growth rates
                    weights = np.linspace(0.5, 1.0, len(growth_rates))
                    return np.average(growth_rates, weights=weights)
            
            # Default growth rate if can't calculate
            return 0.02
            
        except Exception:
            return 0.02
    
    def _project_fcf_growth(self) -> float:
        """Project free cash flow growth"""
        try:
            # Get historical FCF
            financials = self.company.get_credit_metrics()
            historical_fcf = financials.get('historical_fcf', [])
            
            if len(historical_fcf) >= 2:
                # Calculate growth rate
                growth_rates = []
                for i in range(1, len(historical_fcf)):
                    if historical_fcf[i-1] > 0:
                        growth = (historical_fcf[i] - historical_fcf[i-1]) / historical_fcf[i-1]
                        growth_rates.append(growth)
                
                if growth_rates:
                    # Use weighted average of growth rates
                    weights = np.linspace(0.5, 1.0, len(growth_rates))
                    return np.average(growth_rates, weights=weights)
            
            # Default growth rate if can't calculate
            return 0.02
            
        except Exception:
            return 0.02
    
    def _analyze_debt_structure(self, balance_sheet: Dict) -> Dict[str, Any]:
        """Analyze current debt structure"""
        try:
            total_debt = balance_sheet.get('total_debt', 0)
            if total_debt == 0:
                return {}
            
            structure = {
                'short_term_ratio': balance_sheet.get('short_term_debt', 0) / total_debt,
                'long_term_ratio': balance_sheet.get('long_term_debt', 0) / total_debt,
                'secured_ratio': balance_sheet.get('secured_debt', 0) / total_debt,
                'subordinated_ratio': balance_sheet.get('subordinated_debt', 0) / total_debt
            }
            
            # Add risk assessment
            structure['risk_assessment'] = self._assess_structure_risk(structure)
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing debt structure: {str(e)}")
            return {}
    
    def _assess_structure_risk(self, structure: Dict[str, float]) -> Dict[str, str]:
        """Assess risks in debt structure"""
        try:
            risks = {}
            
            # Assess short-term risk
            if structure['short_term_ratio'] > 0.4:
                risks['refinancing_risk'] = 'High'
            elif structure['short_term_ratio'] > 0.25:
                risks['refinancing_risk'] = 'Medium'
            else:
                risks['refinancing_risk'] = 'Low'
            
            # Assess secured debt risk
            if structure['secured_ratio'] > 0.5:
                risks['flexibility_risk'] = 'High'
            elif structure['secured_ratio'] > 0.3:
                risks['flexibility_risk'] = 'Medium'
            else:
                risks['flexibility_risk'] = 'Low'
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing structure risk: {str(e)}")
            return {}
    
    def _get_maturity_profile(self) -> Dict[str, float]:
        """Get or create synthetic debt maturity profile"""
        try:
            # Try to get actual maturity profile from company data
            maturities = self.company.get_debt_maturities()
            if maturities:
                return maturities
            
            # Create synthetic profile based on balance sheet data
            balance_sheet = self.company.get_balance_sheet()
            total_debt = balance_sheet.get('total_debt', 0)
            short_term_debt = balance_sheet.get('short_term_debt', total_debt * 0.3)
            long_term_debt = balance_sheet.get('long_term_debt', total_debt * 0.7)
            
            # Distribute debt across typical maturity buckets
            synthetic_profile = {
                1: short_term_debt,  # Due within 1 year
                3: long_term_debt * 0.4,  # Due in 2-3 years
                5: long_term_debt * 0.3,  # Due in 4-5 years
                7: long_term_debt * 0.2,  # Due in 6-7 years
                10: long_term_debt * 0.1   # Due in 8-10 years
            }
            
            return synthetic_profile
            
        except Exception as e:
            logger.error(f"Error getting maturity profile: {str(e)}")
            return {
                1: 0,
                3: 0,
                5: 0,
                7: 0,
                10: 0
            }
    
    def _get_recommendations(
        self,
        current_metrics: Dict[str, float],
        debt_capacity: Dict[str, float]
    ) -> List[str]:
        """Get debt management recommendations"""
        try:
            # Get current metrics and debt capacity
            balance_sheet = self.company.get_balance_sheet()
            income_stmt = self.company.get_income_statement()
            cash_flow = self.company.get_cash_flow()
            
            current_metrics = self._calculate_current_metrics(
                balance_sheet, income_stmt, cash_flow
            )
            
            debt_capacity = self._calculate_debt_capacity(
                current_metrics, balance_sheet, income_stmt
            )
            
            # Get recommendations based on metrics and capacity
            return self._get_recommendations(current_metrics, debt_capacity)
            
        except Exception as e:
            logger.error(f"Error getting debt recommendations: {str(e)}")
            return ["Unable to generate recommendations due to data limitations"]
    
    def get_peer_comparison(self) -> Dict[str, Any]:
        """Compare debt metrics with peers"""
        try:
            # Get company metrics
            company_metrics = self._calculate_current_metrics(
                self.company.get_balance_sheet(),
                self.company.get_income_statement(),
                self.company.get_cash_flow()
            )
            
            # Get peer metrics (this would typically come from a data provider)
            peer_metrics = self._get_peer_metrics()
            
            # Calculate percentiles
            percentiles = {}
            for metric in ['debt_to_ebitda', 'interest_coverage', 'debt_to_assets']:
                peer_values = [p.get(metric, 0) for p in peer_metrics]
                if peer_values:
                    percentile = np.percentile(peer_values, 50)  # Median
                    percentiles[metric] = {
                        'company': company_metrics.get(metric, 0),
                        'peer_median': percentile,
                        'percentile': np.mean(np.array(peer_values) <= company_metrics.get(metric, 0))
                    }
            
            return {
                'company_metrics': company_metrics,
                'peer_metrics': peer_metrics,
                'percentiles': percentiles
            }
            
        except Exception as e:
            logger.error(f"Error in peer comparison: {str(e)}")
            return {}
    
    def _get_peer_metrics(self) -> List[Dict[str, float]]:
        """Get peer company metrics"""
        try:
            # This would typically come from a data provider
            # For now, using synthetic data
            return [
                {
                    'debt_to_ebitda': 3.5,
                    'interest_coverage': 3.0,
                    'debt_to_assets': 0.4
                },
                {
                    'debt_to_ebitda': 4.0,
                    'interest_coverage': 2.8,
                    'debt_to_assets': 0.45
                },
                {
                    'debt_to_ebitda': 3.8,
                    'interest_coverage': 3.2,
                    'debt_to_assets': 0.42
                }
            ]
            
        except Exception as e:
            logger.error(f"Error getting peer metrics: {str(e)}")
            return []
    
    def _calculate_refinancing_risk(self, metrics: Dict[str, float]) -> float:
        """Calculate refinancing risk score"""
        try:
            # Weight different factors
            weights = {
                'short_term_ratio': 0.3,
                'cash_coverage_ratio': 0.3,
                'debt_service_burden': 0.2,
                'variable_rate_debt_ratio': 0.2
            }
            
            # Calculate component scores (0-1, higher is riskier)
            scores = {
                'short_term_ratio': min(1.0, metrics.get('short_term_ratio', 0)),
                'cash_coverage_ratio': min(1.0, 1.0 / metrics.get('cash_coverage_ratio', 1.0)),
                'debt_service_burden': min(1.0, metrics.get('debt_service_burden', 0)),
                'variable_rate_debt_ratio': min(1.0, metrics.get('variable_rate_debt_ratio', 0))
            }
            
            # Calculate weighted average
            risk_score = sum(
                weights[metric] * scores[metric]
                for metric in weights.keys()
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating refinancing risk: {str(e)}")
            return 0.5
    
    def _calculate_overall_risk(self, metrics: Dict[str, float]) -> float:
        """Calculate overall risk score combining all risk factors"""
        try:
            # Weight different risk components
            weights = {
                'credit_risk': 0.35,
                'refinancing_risk': 0.25,
                'duration_risk': 0.20,
                'liquidity_risk': 0.20
            }
            
            # Get component risks
            risks = {
                'credit_risk': metrics.get('credit_risk', 0.5),
                'refinancing_risk': metrics.get('refinancing_risk', 0.5),
                'duration_risk': metrics.get('duration_risk', 0.5),
                'liquidity_risk': metrics.get('liquidity_risk', 0.5)
            }
            
            # Calculate weighted average
            overall_risk = sum(
                weights[risk] * risks[risk]
                for risk in weights.keys()
            )
            
            return min(1.0, max(0.0, overall_risk))
            
        except Exception as e:
            logger.error(f"Error calculating overall risk: {str(e)}")
            return 0.5
    
    def analyze_debt_capacity(self) -> Dict[str, Any]:
        """Analyze company's debt capacity based on financial metrics"""
        try:
            # Get financial statements
            balance_sheet = self.company.get_balance_sheet()
            income_stmt = self.company.get_income_statement()
            cash_flow = self.company.get_cash_flow()
            
            # Get current metrics
            metrics = self._calculate_current_metrics(balance_sheet, income_stmt, cash_flow)
            
            # Calculate capacity based on EBITDA multiple
            ebitda = income_stmt.get('ebitda', 0)
            max_debt_ebitda = ebitda * self.benchmarks['debt_to_ebitda']['medium_risk']
            
            # Calculate capacity based on asset coverage
            total_assets = balance_sheet.get('total_assets', 0)
            max_debt_assets = total_assets * self.benchmarks['debt_to_assets']['medium_risk']
            
            # Calculate capacity based on cash flow coverage
            operating_cash_flow = cash_flow.get('operating_cash_flow', 0)
            max_debt_cash_flow = operating_cash_flow * 4  # Assume 4x operating cash flow as maximum
            
            return {
                'max_debt_ebitda': max_debt_ebitda,
                'max_debt_assets': max_debt_assets,
                'max_debt_cash_flow': max_debt_cash_flow,
                'recommended_max_debt': min(max_debt_ebitda, max_debt_assets, max_debt_cash_flow)
            }
            
        except Exception as e:
            logger.error(f"Error calculating debt capacity: {str(e)}")
            return {
                'max_debt_ebitda': 0,
                'max_debt_assets': 0,
                'max_debt_cash_flow': 0,
                'recommended_max_debt': 0
            }

    def analyze_investment_opportunity(self) -> Dict[str, Any]:
        """Analyze debt investment opportunity"""
        try:
            # Get bond/debt instrument details
            debt_instruments = self.company.get_debt_instruments()
            
            # Calculate relative value metrics
            relative_value = self._calculate_relative_value(debt_instruments)
            
            # Assess credit risk and recovery potential
            credit_analysis = self._analyze_credit_risk()
            recovery_analysis = self._analyze_recovery_potential()
            
            # Generate investment recommendations
            recommendations = self._generate_investment_recommendations(
                relative_value,
                credit_analysis,
                recovery_analysis
            )
            
            return {
                'relative_value': relative_value,
                'credit_analysis': credit_analysis,
                'recovery_analysis': recovery_analysis,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error analyzing investment opportunity: {str(e)}")
            raise

    def get_debt_recommendations(self) -> Dict[str, Any]:
        """Generate debt management recommendations based on analysis"""
        try:
            # Get current metrics
            balance_sheet = self.company.get_balance_sheet()
            income_stmt = self.company.get_income_statement()
            cash_flow = self.company.get_cash_flow_statement()
            
            metrics = self._calculate_current_metrics(balance_sheet, income_stmt, cash_flow)
            
            # Analyze debt capacity
            capacity = self.analyze_debt_capacity()
            
            recommendations = {
                'structure_recommendations': [],
                'refinancing_recommendations': [],
                'risk_mitigation_recommendations': []
            }
            
            # Structure recommendations
            if metrics['short_term_ratio'] > 0.4:
                recommendations['structure_recommendations'].append(
                    "Consider terming out short-term debt to reduce refinancing risk"
                )
            
            if metrics['variable_rate_debt_ratio'] > 0.5:
                recommendations['structure_recommendations'].append(
                    "Consider hedging interest rate risk through fixed rate conversion or swaps"
                )
            
            # Refinancing recommendations
            if metrics['debt_service_coverage'] < 1.5:
                recommendations['refinancing_recommendations'].append(
                    "Evaluate debt restructuring options to improve debt service coverage"
                )
            
            if metrics['interest_coverage'] < 2.5:
                recommendations['refinancing_recommendations'].append(
                    "Consider refinancing high-cost debt to reduce interest burden"
                )
            
            # Risk mitigation
            if metrics['debt_to_ebitda'] > 4.5:
                recommendations['risk_mitigation_recommendations'].append(
                    "Focus on EBITDA growth and debt reduction to improve leverage metrics"
                )
            
            if metrics['credit_risk'] > 0.6:
                recommendations['risk_mitigation_recommendations'].append(
                    "Develop comprehensive credit risk mitigation strategy"
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating debt recommendations: {str(e)}")
            return {
                'structure_recommendations': [],
                'refinancing_recommendations': [],
                'risk_mitigation_recommendations': []
            }