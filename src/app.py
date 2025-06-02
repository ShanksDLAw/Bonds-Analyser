"""Main application module for Bond Risk Analyzer"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.data.fetcher import BondRiskFetcher
from src.data.country_fetcher import CountryDataFetcher
from src.models.company import CompanyFinancials, Bond
from src.risk.metrics import RiskAnalyzer
from src.risk.simulations import RiskSimulator
from src.config import Config
from src.risk.model_survival import SurvivalAnalyzer
from src.data.bond_fetcher import BondFetcher
from src.risk.debt_analyzer import DebtAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    if value is None:
        return "N/A"
    if value >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif value >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    else:
        return f"${value:.{decimals}f}"

def format_basis_points(value: float) -> str:
    """Format value as basis points"""
    if value is None:
        return "N/A"
    return f"{value:.0f} bps"

def create_risk_gauge(value: float, title: str) -> go.Figure:
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ]
        }
    ))
    fig.update_layout(height=200)
    return fig

def create_coverage_chart(ratios: dict) -> go.Figure:
    """Create a bar chart for coverage ratios"""
    fig = go.Figure()
    
    ratios_data = pd.DataFrame({
        'Ratio': list(ratios.keys()),
        'Value': list(ratios.values()),
        'Benchmark': [4.0, 3.0, 2.5]  # Industry benchmarks
    })
    
    fig.add_trace(go.Bar(
        name='Actual',
        x=ratios_data['Ratio'],
        y=ratios_data['Value'],
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        name='Benchmark',
        x=ratios_data['Ratio'],
        y=ratios_data['Benchmark'],
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        barmode='group',
        title="Coverage Ratios vs Benchmarks",
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_debt_profile_chart(debt_analysis: Dict[str, Any]) -> go.Figure:
    """Create debt profile visualization"""
    metrics = debt_analysis.get('current_metrics', {})
    benchmarks = debt_analysis.get('industry_benchmarks', {})
    
    fig = go.Figure()
    
    # Add actual metrics
    fig.add_trace(go.Scatterpolar(
        r=[
            metrics.get('debt_to_ebitda', 0),
            metrics.get('debt_to_assets', 0),
            metrics.get('debt_service_coverage', 0),
            metrics.get('interest_coverage', 0)
        ],
        theta=['Debt/EBITDA', 'Debt/Assets', 'DSCR', 'ICR'],
        fill='toself',
        name='Company',
        line_color='rgb(55, 83, 109)'
    ))
    
    # Add benchmark metrics
    fig.add_trace(go.Scatterpolar(
        r=[
            benchmarks.get('debt_to_ebitda', {}).get('medium_risk', 0),
            benchmarks.get('debt_to_assets', {}).get('medium_risk', 0),
            benchmarks.get('debt_service_coverage', {}).get('medium_risk', 0),
            benchmarks.get('interest_coverage', {}).get('medium_risk', 0)
        ],
        theta=['Debt/EBITDA', 'Debt/Assets', 'DSCR', 'ICR'],
        fill='toself',
        name='Industry Benchmark',
        line_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(metrics.get('debt_to_ebitda', 0), 8)]
            )
        ),
        showlegend=True,
        title="Debt Profile vs Industry",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_macro_benchmark_chart(company_metrics: Dict[str, Any], country_data: Dict[str, Any]) -> go.Figure:
    """Create macro benchmarking visualization"""
    fig = go.Figure()
    
    # Company metrics with defaults
    company_values = [
        company_metrics.get('debt_to_assets', 0),
        company_metrics.get('interest_coverage', 0),
        company_metrics.get('debt_service_coverage', 0),
        company_metrics.get('operating_margin', 0)
    ]
    
    # Country benchmarks with defaults
    country_values = [
        country_data.get('corporate_metrics', {}).get('median_leverage', 2.5),
        country_data.get('corporate_metrics', {}).get('median_interest_coverage', 3.0),
        country_data.get('corporate_metrics', {}).get('median_debt_service', 1.8),
        country_data.get('corporate_metrics', {}).get('median_operating_margin', 0.15)
    ]
    
    # Add traces
    fig.add_trace(go.Bar(
        name='Company',
        x=['Leverage', 'ICR', 'DSCR', 'Op. Margin'],
        y=company_values,
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        name='Country Median',
        x=['Leverage', 'ICR', 'DSCR', 'Op. Margin'],
        y=country_values,
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        barmode='group',
        title="Company vs Country Benchmarks",
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis_title="Ratio",
        xaxis_title="Metric"
    )
    
    return fig

class BondRiskAnalyzer:
    """Main application class"""
    
    def __init__(self):
        """Initialize application"""
        self.country_fetcher = CountryDataFetcher()
        self.risk_analyzer = None
        self.debt_analyzer = None
        self.survival_analyzer = SurvivalAnalyzer()
        
        # Initialize session state
        if 'bond_data' not in st.session_state:
            st.session_state.bond_data = None
        if 'company_data' not in st.session_state:
            st.session_state.company_data = None
        if 'survival_results' not in st.session_state:
            st.session_state.survival_results = None
        if 'country_data' not in st.session_state:
            st.session_state.country_data = None
    
    def run(self):
        """Run the application"""
        st.set_page_config(page_title="Corporate Bond Risk Analyzer", layout="wide")
        
        # Title and description
        st.title("Corporate Bond Risk Analyzer")
        st.write("This tool analyzes corporate bond risks by considering:")
        st.write("- Sovereign risk factors")
        st.write("- Company-specific metrics")
        st.write("- Market conditions")
        st.write("- Scenario analysis")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content
        if st.session_state.company_data:
            self._render_analysis()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("Risk Analysis Settings")
            
            # Company search
            company_name = st.text_input("Enter Company Name")
            if st.button("Analyze Risk"):
                with st.spinner("Analyzing company data..."):
                    try:
                        company = CompanyFinancials(company_name)
                        st.session_state.company_data = company
                        self.risk_analyzer = RiskAnalyzer(company)
                        self.debt_analyzer = DebtAnalyzer(company)
                    except Exception as e:
                        st.error(f"Error analyzing company: {str(e)}")
            
            # Country selection
            country = st.selectbox(
                "Select Country",
                ["USA", "UK", "Germany", "France", "Japan", "China", "India"]
            )
            
            if country:
                with st.spinner("Loading country data..."):
                    try:
                        country_data = self.country_fetcher.get_comprehensive_metrics(country)
                        if not country_data:
                            st.warning("Using default country metrics")
                            country_data = {
                                'macro': {
                                    'gdp_growth': 0.025,
                                    'inflation': 0.02,
                                    'unemployment': 0.05,
                                    'debt_to_gdp': 0.6,
                                    'current_account': 0.0,
                                    'policy_rate': 0.03,
                                    'fiscal_balance': -0.03
                                },
                                'corporate_metrics': {
                                    'median_leverage': 2.5,
                                    'median_interest_coverage': 3.0,
                                    'median_debt_service': 1.8,
                                    'median_operating_margin': 0.15
                                },
                                'risk_metrics': {
                                    'sovereign_risk': 0.3,
                                    'political_risk': 0.3,
                                    'market_risk': 0.3
                                },
                                'market_data': {
                                    'bond_market_liquidity': 0.7,
                                    'credit_conditions': {'lending_sentiment': 0.6},
                                    'market_volatility': 0.15,
                                    'corporate_spreads': {'BBB': 0.02}
                                }
                            }
                        st.session_state.country_data = country_data
                    except Exception as e:
                        st.error(f"Error loading country data: {str(e)}")
            
            # Analysis type
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Full Analysis", "Debt Only", "Bond Only"]
            )
            
            # Peer comparison
            st.subheader("Peer Comparison")
            peer_tickers = st.text_input("Enter peer tickers (comma-separated)")
            
            if peer_tickers and self.debt_analyzer:
                with st.spinner("Analyzing peers..."):
                    try:
                        peers = [ticker.strip() for ticker in peer_tickers.split(',')]
                        st.session_state.peer_analysis = self.debt_analyzer.benchmark_against_peers(peers)
                    except Exception as e:
                        st.error(f"Error analyzing peers: {str(e)}")
    
    def _render_analysis(self):
        """Render main analysis section"""
        if st.session_state.company_data:
            tabs = st.tabs([
                "Risk Metrics",
                "Debt Analysis",
                "Survival Analysis",
                "Macro Benchmarking",
                "Peer Comparison",
                "Recommendations"
            ])
            
            with tabs[0]:
                self._render_risk_metrics()
            with tabs[1]:
                self._render_debt_analysis()
            with tabs[2]:
                self._render_survival_analysis()
            with tabs[3]:
                self._render_macro_benchmarking()
            with tabs[4]:
                self._render_peer_comparison()
            with tabs[5]:
                self._render_recommendations()
    
    def _render_risk_metrics(self):
        """Render risk metrics section"""
        try:
            if self.risk_analyzer:
                st.subheader("Risk Analysis")
                
                # Get metrics
                metrics = self.risk_analyzer.get_risk_metrics()
                
                # Create metrics display
                cols = st.columns(3)
                
                with cols[0]:
                    st.metric(
                        "Default Probability",
                        format_percentage(metrics.get('default_probability')),
                        "vs Industry"
                    )
                
                with cols[1]:
                    st.metric(
                        "Yield Spread",
                        format_basis_points(metrics.get('yield_spread')),
                        "vs Benchmark"
                    )
                
                with cols[2]:
                    st.metric(
                        "Risk Score",
                        f"{metrics.get('risk_score', 50):.0f}/100",
                        "0-100 scale"
                    )
                
        except Exception as e:
            st.error(f"Error rendering risk metrics: {str(e)}")
    
    def _render_debt_analysis(self):
        """Render debt analysis section"""
        try:
            if not self.debt_analyzer:
                st.warning("No company data available for debt analysis")
                return
                
            debt_analysis = self.debt_analyzer.analyze_debt_profile()
            if not debt_analysis:
                st.warning("Unable to perform debt analysis")
                return
                
            # Core metrics
            st.subheader("Debt Profile Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Credit Risk",
                    format_percentage(debt_analysis['current_metrics'].get('credit_risk', 0)),
                    help="Overall credit risk score based on financial metrics"
                )
            with col2:
                st.metric(
                    "Duration Risk",
                    format_percentage(debt_analysis['current_metrics'].get('duration_risk', 0)),
                    help="Risk from debt maturity profile"
                )
            with col3:
                st.metric(
                    "Refinancing Risk",
                    format_percentage(debt_analysis['current_metrics'].get('refinancing_risk', 0)),
                    help="Risk from potential refinancing needs"
                )
            
            # Debt burden metrics
            st.subheader("Debt Burden Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Debt/EBITDA',
                        'Debt/Assets',
                        'Debt/Revenue',
                        'Debt/FCF'
                    ],
                    'Value': [
                        f"{debt_analysis['current_metrics'].get('debt_to_ebitda', 0):.2f}x",
                        format_percentage(debt_analysis['current_metrics'].get('debt_to_assets', 0)),
                        f"{debt_analysis['current_metrics'].get('debt_to_revenue', 0):.2f}x",
                        f"{debt_analysis['current_metrics'].get('debt_to_fcf', 0):.2f}x"
                    ],
                    'Industry Avg': [
                        f"{debt_analysis['industry_benchmarks'].get('debt_to_ebitda', {}).get('medium_risk', 0):.2f}x",
                        format_percentage(debt_analysis['industry_benchmarks'].get('debt_to_assets', {}).get('medium_risk', 0)),
                        'N/A',
                        'N/A'
                    ]
                })
                st.dataframe(metrics_df, hide_index=True)
            
            with col2:
                st.plotly_chart(create_debt_profile_chart(debt_analysis), use_container_width=True)
            
            # Coverage metrics
            st.subheader("Coverage Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                coverage_df = pd.DataFrame({
                    'Metric': [
                        'Interest Coverage',
                        'Fixed Charge Coverage',
                        'Debt Service Coverage',
                        'Cash Coverage'
                    ],
                    'Value': [
                        f"{debt_analysis['current_metrics'].get('interest_coverage', 0):.2f}x",
                        f"{debt_analysis['current_metrics'].get('fixed_charge_coverage', 0):.2f}x",
                        f"{debt_analysis['current_metrics'].get('debt_service_coverage', 0):.2f}x",
                        f"{debt_analysis['current_metrics'].get('cash_coverage_ratio', 0):.2f}x"
                    ],
                    'Benchmark': [
                        f"{debt_analysis['industry_benchmarks'].get('interest_coverage', {}).get('medium_risk', 0):.2f}x",
                        f"{debt_analysis['industry_benchmarks'].get('fixed_charge_coverage', {}).get('medium_risk', 0):.2f}x",
                        f"{debt_analysis['industry_benchmarks'].get('debt_service_coverage', {}).get('medium_risk', 0):.2f}x",
                        'N/A'
                    ]
                })
                st.dataframe(coverage_df, hide_index=True)
            
            with col2:
                st.plotly_chart(create_coverage_chart({
                    'Interest Coverage': debt_analysis['current_metrics'].get('interest_coverage', 0),
                    'Fixed Charge': debt_analysis['current_metrics'].get('fixed_charge_coverage', 0),
                    'Debt Service': debt_analysis['current_metrics'].get('debt_service_coverage', 0)
                }), use_container_width=True)
            
            # Debt structure
            st.subheader("Debt Structure")
            col1, col2 = st.columns(2)
            
            with col1:
                structure_df = pd.DataFrame({
                    'Component': [
                        'Short-term Debt',
                        'Long-term Debt',
                        'Secured Debt',
                        'Variable Rate'
                    ],
                    'Amount': [
                        format_currency(debt_analysis['current_metrics'].get('short_term_debt', 0)),
                        format_currency(debt_analysis['current_metrics'].get('long_term_debt', 0)),
                        format_currency(debt_analysis['current_metrics'].get('secured_debt_ratio', 0) * debt_analysis['current_metrics'].get('total_debt', 0)),
                        format_currency(debt_analysis['current_metrics'].get('variable_rate_debt_ratio', 0) * debt_analysis['current_metrics'].get('total_debt', 0))
                    ],
                    'Percentage': [
                        format_percentage(debt_analysis['current_metrics'].get('short_term_ratio', 0)),
                        format_percentage(1 - debt_analysis['current_metrics'].get('short_term_ratio', 0)),
                        format_percentage(debt_analysis['current_metrics'].get('secured_debt_ratio', 0)),
                        format_percentage(debt_analysis['current_metrics'].get('variable_rate_debt_ratio', 0))
                    ]
                })
                st.dataframe(structure_df, hide_index=True)
            
            with col2:
                # Debt capacity analysis
                capacity = debt_analysis.get('debt_capacity', {})
                if capacity:
                    capacity_df = pd.DataFrame({
                        'Metric': [
                            'Current Debt',
                            'Sustainable Level',
                            'Additional Capacity',
                            'EBITDA Capacity',
                            'Asset Capacity'
                        ],
                        'Amount': [
                            format_currency(debt_analysis['current_metrics'].get('total_debt', 0)),
                            format_currency(capacity.get('sustainable_debt_level', 0)),
                            format_currency(capacity.get('additional_capacity', 0)),
                            format_currency(capacity.get('ebitda_based_capacity', 0)),
                            format_currency(capacity.get('asset_based_capacity', 0))
                        ]
                    })
                    st.dataframe(capacity_df, hide_index=True)
            
            # Recommendations
            st.subheader("Recommendations")
            for rec in debt_analysis.get('recommendations', []):
                st.info(rec)
                
        except Exception as e:
            logger.error(f"Error rendering debt analysis: {str(e)}")
            st.error("Error displaying debt analysis")
    
    def _render_survival_analysis(self):
        """Render survival analysis section"""
        st.subheader("Survival Analysis")
        
        # Get company data
        company = st.session_state.company_data
        bond = st.session_state.bond_data
        
        # If no bond data is available, display a message and return
        if not bond:
            st.warning("No bond data available for this company. Please try another company or check data sources.")
            return
            
        # Calculate survival metrics
        metrics = self.survival_analyzer.calculate_expected_loss(
            exposure=bond.face_value,
            rating=bond.credit_rating,
            time_horizon=1.0
        )
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Probability of Default",
                format_percentage(metrics.probability_of_default)
            )
            st.metric(
                "Loss Given Default",
                format_percentage(metrics.loss_given_default)
            )
        with col2:
            st.metric(
                "Expected Loss",
                format_currency(metrics.expected_loss)
            )
            st.metric(
                "Time to Default",
                f"{metrics.time_to_default:.1f} years"
            )
        with col3:
            st.metric(
                "Exposure at Default",
                format_currency(metrics.exposure_at_default)
            )
        # Show synthetic bonds if using fallback
        if not st.session_state.bond_data:
            st.subheader("Synthetic Bonds")
            synthetic_bonds = pd.DataFrame([
                {
                    'ID': 'SYNTHETIC3Y',
                    'Maturity': (datetime.now().replace(year=datetime.now().year + 3, month=5, day=30)).strftime('%Y-%m-%d'),
                    'Coupon': '6.0%',
                    'YTM': '7.2%',
                    'Rating': 'BBB',
                    'Duration': '2.80',
                    'Amount': format_currency(bond.face_value/3)
                },
                {
                    'ID': 'SYNTHETIC5Y',
                    'Maturity': (datetime.now().replace(year=datetime.now().year + 5, month=5, day=30)).strftime('%Y-%m-%d'),
                    'Coupon': '6.0%',
                    'YTM': '8.0%',
                    'Rating': 'BBB',
                    'Duration': '4.63',
                    'Amount': format_currency(bond.face_value/3)
                },
                {
                    'ID': 'SYNTHETIC10Y',
                    'Maturity': (datetime.now().replace(year=datetime.now().year + 10, month=5, day=29)).strftime('%Y-%m-%d'),
                    'Coupon': '6.0%',
                    'YTM': '10.0%',
                    'Rating': 'BBB',
                    'Duration': '9.09',
                    'Amount': format_currency(bond.face_value/3)
                }
            ])
            st.table(synthetic_bonds)
    
    def _render_macro_benchmarking(self):
        """Render macro benchmarking section"""
        try:
            if not self.debt_analyzer or 'country_data' not in st.session_state:
                st.warning("No data available for macro benchmarking")
                return
                
            country_data = st.session_state.country_data
            debt_analysis = self.debt_analyzer.analyze_debt_profile()
            
            if not country_data or not debt_analysis:
                st.warning("Unable to perform macro benchmarking")
                return
            
            # Country risk dashboard
            st.subheader("Country Risk Dashboard")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Sovereign Risk",
                    format_percentage(country_data.get('risk_metrics', {}).get('sovereign_risk', 0.3)),
                    help="Overall sovereign risk score"
                )
            with col2:
                st.metric(
                    "Political Risk",
                    format_percentage(country_data.get('risk_metrics', {}).get('political_risk', 0.3)),
                    help="Political and institutional risk"
                )
            with col3:
                st.metric(
                    "Market Risk",
                    format_percentage(country_data.get('risk_metrics', {}).get('market_risk', 0.3)),
                    help="Market and liquidity risk"
                )
            
            # Economic indicators
            st.subheader("Economic Indicators")
            indicators_df = pd.DataFrame({
                'Indicator': [
                    'GDP Growth',
                    'Inflation',
                    'Debt/GDP',
                    'Current Account',
                    'Unemployment',
                    'Policy Rate'
                ],
                'Value': [
                    format_percentage(country_data.get('macro', {}).get('gdp_growth', 0.025)),
                    format_percentage(country_data.get('macro', {}).get('inflation', 0.02)),
                    format_percentage(country_data.get('macro', {}).get('debt_to_gdp', 0.6)),
                    format_percentage(country_data.get('macro', {}).get('current_account', 0)),
                    format_percentage(country_data.get('macro', {}).get('unemployment', 0.05)),
                    format_percentage(country_data.get('macro', {}).get('policy_rate', 0.03))
                ]
            })
            st.dataframe(indicators_df, hide_index=True)
            
            # Corporate sector benchmarking
            st.subheader("Corporate Sector Benchmarking")
            col1, col2 = st.columns(2)
            
            with col1:
                benchmark_df = pd.DataFrame({
                    'Metric': [
                        'Leverage Ratio',
                        'Interest Coverage',
                        'Debt Service',
                        'Operating Margin'
                    ],
                    'Company': [
                        f"{debt_analysis.get('current_metrics', {}).get('debt_to_assets', 0):.2f}x",
                        f"{debt_analysis.get('current_metrics', {}).get('interest_coverage', 0):.2f}x",
                        f"{debt_analysis.get('current_metrics', {}).get('debt_service_coverage', 0):.2f}x",
                        format_percentage(debt_analysis.get('current_metrics', {}).get('operating_margin', 0))
                    ],
                    'Country Median': [
                        f"{country_data.get('corporate_metrics', {}).get('median_leverage', 2.5):.2f}x",
                        f"{country_data.get('corporate_metrics', {}).get('median_interest_coverage', 3.0):.2f}x",
                        f"{country_data.get('corporate_metrics', {}).get('median_debt_service', 1.8):.2f}x",
                        format_percentage(country_data.get('corporate_metrics', {}).get('median_operating_margin', 0.15))
                    ]
                })
                st.dataframe(benchmark_df, hide_index=True)
            
            with col2:
                st.plotly_chart(create_macro_benchmark_chart(
                    debt_analysis.get('current_metrics', {}),
                    country_data
                ), use_container_width=True)
            
            # Market conditions
            st.subheader("Market Conditions")
            col1, col2 = st.columns(2)
            
            with col1:
                market_df = pd.DataFrame({
                    'Metric': [
                        'Bond Market Liquidity',
                        'Credit Conditions',
                        'Market Volatility',
                        'Corporate Spreads (BBB)'
                    ],
                    'Value': [
                        format_percentage(country_data.get('market_data', {}).get('bond_market_liquidity', 0.7)),
                        format_percentage(country_data.get('market_data', {}).get('credit_conditions', {}).get('lending_sentiment', 0.6)),
                        format_percentage(country_data.get('market_data', {}).get('market_volatility', 0.15)),
                        format_basis_points(country_data.get('market_data', {}).get('corporate_spreads', {}).get('BBB', 0.02) * 100)
                    ]
                })
                st.dataframe(market_df, hide_index=True)
            
            with col2:
                # Cross-reference analysis
                cross_ref_df = pd.DataFrame({
                    'Metric': [
                        'Relative Leverage',
                        'Coverage vs Market',
                        'Spread Premium',
                        'Risk Score'
                    ],
                    'Value': [
                        f"{debt_analysis.get('current_metrics', {}).get('debt_to_assets', 0) / country_data.get('corporate_metrics', {}).get('median_leverage', 2.5):.2f}x",
                        f"{debt_analysis.get('current_metrics', {}).get('interest_coverage', 0) / country_data.get('corporate_metrics', {}).get('median_interest_coverage', 3.0):.2f}x",
                        format_basis_points((debt_analysis.get('current_metrics', {}).get('credit_risk', 0) - country_data.get('market_data', {}).get('corporate_spreads', {}).get('BBB', 0.02)) * 100),
                        f"{(debt_analysis.get('current_metrics', {}).get('credit_risk', 0) + country_data.get('risk_metrics', {}).get('sovereign_risk', 0.3)) / 2 * 100:.0f}/100"
                    ]
                })
                st.dataframe(cross_ref_df, hide_index=True)
                
        except Exception as e:
            logger.error(f"Error rendering macro benchmarking: {str(e)}")
            st.error("Error displaying macro benchmarking")
    
    def _render_peer_comparison(self):
        """Render peer comparison section"""
        try:
            if 'peer_analysis' in st.session_state:
                st.subheader("Peer Comparison")
                
                analysis = st.session_state.peer_analysis
                
                # Show relative position
                for metric, percentile in analysis.get('relative_position', {}).items():
                    st.metric(
                        metric,
                        f"{percentile:.0f}th",
                        "Percentile"
                    )
                
                # Show peer averages
                st.write("Peer Averages")
                averages = analysis.get('peer_averages', {})
                for metric, value in averages.items():
                    if 'coverage' in metric.lower():
                        st.write(f"{metric}: {value:.2f}x")
                    else:
                        st.write(f"{metric}: {format_percentage(value)}")
                
        except Exception as e:
            st.error(f"Error rendering peer comparison: {str(e)}")
    
    def _render_recommendations(self):
        """Render recommendations section"""
        try:
            if self.debt_analyzer:
                st.subheader("Recommendations")
                
                recommendations = self.debt_analyzer.get_debt_recommendations()
                
                for rec in recommendations:
                    st.write("• " + rec)
                
        except Exception as e:
            st.error(f"Error rendering recommendations: {str(e)}")

def main():
    """Main entry point"""
    try:
        app = BondRiskAnalyzer()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()