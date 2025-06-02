from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import json
import yfinance as yf
import pandas as pd
from scipy import stats
import requests
import time
import logging
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Config
from src.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

class FinanceAPIError(Exception):
    """Custom exception for API errors"""
    pass

@dataclass
class Bond:
    """Bond instrument data class"""
    isin: str
    cusip: Optional[str]
    issuer: str
    maturity_date: datetime
    coupon_rate: float
    face_value: float
    currency: str
    issue_date: datetime
    credit_rating: str
    bond_type: str  # e.g., 'corporate', 'government'
    seniority: str  # e.g., 'senior', 'subordinated'
    secured: bool
    callable: bool
    current_price: float
    ytm: float
    duration: float
    convexity: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'isin': self.isin,
            'cusip': self.cusip,
            'issuer': self.issuer,
            'maturity_date': self.maturity_date.isoformat(),
            'coupon_rate': self.coupon_rate,
            'face_value': self.face_value,
            'currency': self.currency,
            'issue_date': self.issue_date.isoformat(),
            'credit_rating': self.credit_rating,
            'bond_type': self.bond_type,
            'seniority': self.seniority,
            'secured': self.secured,
            'callable': self.callable,
            'current_price': self.current_price,
            'ytm': self.ytm,
            'duration': self.duration,
            'convexity': self.convexity
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Bond':
        """Create from dictionary"""
        data['maturity_date'] = datetime.fromisoformat(data['maturity_date'])
        data['issue_date'] = datetime.fromisoformat(data['issue_date'])
        return cls(**data)

class CompanyFinancials:
    """Company financial data and analysis"""
    
    def __init__(self, ticker: str):
        self.primary_data_provider = None
        self.secondary_data_provider = None
        """Initialize with company ticker"""
        self.ticker = ticker.upper()
        self.config = Config()
        self._last_api_call = 0
        self._min_call_interval = 0.2  # 200ms between calls
        self._load_data()
        
    def _load_data(self):
        """Load company data with optimized API usage"""
        try:
            # Use Yahoo Finance with enhanced error handling
            self._load_yf_data()
            
            # Merge and validate data
            self._merge_financial_data()
        except Exception as e:
            logger.error(f"Error loading company data: {str(e)}")
            # Fallback to default data if Yahoo Finance fails
            self._init_default_data()
    
    def _load_fmp_data(self):
        """Load data from Financial Modeling Prep API"""
        api_key = self.config.fmp_key()
        base_url = "https://financialmodelingprep.com/api/v3"
        
        # Ensure rate limiting
        self._rate_limit()
        
        # Get company profile
        profile_url = f"{base_url}/profile/{self.ticker}?apikey={api_key}"
        profile_response = requests.get(profile_url)
        if profile_response.status_code != 200:
            raise FinanceAPIError(f"FMP API error: {profile_response.status_code}")
        
        profile_data = profile_response.json()[0]
        
        # Get financial ratios
        ratios_url = f"{base_url}/ratios/{self.ticker}?apikey={api_key}"
        ratios_response = requests.get(ratios_url)
        if ratios_response.status_code != 200:
            raise FinanceAPIError(f"FMP API error: {ratios_response.status_code}")
        
        ratios_data = ratios_response.json()[0]
        
        # Get financial statements
        statements_url = f"{base_url}/financial-statements/{self.ticker}?apikey={api_key}"
        statements_response = requests.get(statements_url)
        if statements_response.status_code != 200:
            raise FinanceAPIError(f"FMP API error: {statements_response.status_code}")
        
        statements_data = statements_response.json()
        
        # Set company attributes
        self.name = profile_data.get('companyName', '')
        self.industry = profile_data.get('industry', '')
        self.sector = profile_data.get('sector', '')
        self.market_cap = profile_data.get('mktCap', 0)
        self.beta = profile_data.get('beta', 1.0)
        
        # Create financial statements DataFrames
        self.balance_sheet = pd.DataFrame(statements_data.get('balance_sheet', []))
        self.income_stmt = pd.DataFrame(statements_data.get('income_statement', []))
        self.cash_flow = pd.DataFrame(statements_data.get('cash_flow_statement', []))
        
        # Load synthetic bonds
        self._load_bonds()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _load_yf_data(self):
        """Load data from Yahoo Finance with retry and rate limiting"""
        self._rate_limit()
        
        try:
            company = yf.Ticker(self.ticker)
            
            # Set timeout for requests
            timeout = 10  # 10 seconds timeout
            
            # Basic info with timeout
            self.info = company.info
            self.name = self.info.get('longName', '')
            self.industry = self.info.get('industry', '')
            self.sector = self.info.get('sector', '')
            
            # Financial statements with retry and timeout
            self.balance_sheet = self._get_yf_statement(company.balance_sheet, timeout)
            self.income_stmt = self._get_yf_statement(company.income_stmt, timeout)
            self.cash_flow = self._get_yf_statement(company.cash_flow, timeout)
            
            # Market data
            self.market_cap = self.info.get('marketCap', 0)
            self.beta = self.info.get('beta', 1.0)
            
            # Load synthetic bonds
            self._load_bonds()
            
        except requests.Timeout:
            logger.error(f"Timeout while fetching data for {self.ticker}")
            raise
        except Exception as e:
            logger.error(f"Error in Yahoo Finance data loading: {str(e)}")
            raise
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        
        if time_since_last_call < self._min_call_interval:
            time.sleep(self._min_call_interval - time_since_last_call)
        
        self._last_api_call = time.time()
    
    @staticmethod
    def _get_yf_statement(statement: pd.DataFrame, timeout: int = 10) -> pd.DataFrame:
        """Safely get financial statement with retry and timeout"""
        try:
            if statement.empty:
                return pd.DataFrame()
            return statement
        except requests.Timeout:
            logger.warning("Timeout while fetching statement data")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error fetching statement data: {str(e)}")
            return pd.DataFrame()
    
    def _init_default_data(self):
        """Initialize with default/synthetic data"""
        # Get market cap from Yahoo Finance as base for synthetic data
        try:
            yf_ticker = yf.Ticker(self.ticker)
            self.market_cap = yf_ticker.info.get('marketCap', 1e9)  # Default to 1B if not found
            self.name = yf_ticker.info.get('longName', self.ticker)
            self.industry = yf_ticker.info.get('industry', 'Unknown')
            self.sector = yf_ticker.info.get('sector', 'Unknown')
            self.beta = yf_ticker.info.get('beta', 1.0)
        except:
            self.market_cap = 1e9  # Default to 1B
            self.name = self.ticker
            self.industry = 'Unknown'
            self.sector = 'Unknown'
            self.beta = 1.0
        
        # Create synthetic balance sheet
        self.balance_sheet = pd.DataFrame([{
            'Total Debt': self.market_cap * 0.3,
            'Short Term Debt': self.market_cap * 0.1,
            'Long Term Debt': self.market_cap * 0.2,
            'Total Assets': self.market_cap * 1.5,
            'Total Equity': self.market_cap * 0.7,
            'Current Assets': self.market_cap * 0.4,
            'Current Liabilities': self.market_cap * 0.3,
            'Cash and Cash Equivalents': self.market_cap * 0.1,
            'Inventory': self.market_cap * 0.1,
            'Secured Debt': self.market_cap * 0.12,  # 40% of total debt
            'Variable Rate Debt': self.market_cap * 0.09  # 30% of total debt
        }])
        
        # Create synthetic income statement
        self.income_stmt = pd.DataFrame([{
            'EBITDA': self.market_cap * 0.15,
            'EBIT': self.market_cap * 0.12,
            'Interest Expense': self.market_cap * 0.02,
            'Net Income': self.market_cap * 0.08,
            'Total Revenue': self.market_cap * 0.5,
            'Operating Income': self.market_cap * 0.13,
            'Gross Profit': self.market_cap * 0.25
        }])
        
        # Create synthetic cash flow
        self.cash_flow = pd.DataFrame([{
            'Operating Cash Flow': self.market_cap * 0.12,
            'Free Cash Flow': self.market_cap * 0.10,
            'Capital Expenditure': -self.market_cap * 0.05,
            'Debt Repayment': -self.market_cap * 0.03
        }])

    def _load_bonds(self):
        """Load company's outstanding bonds"""
        try:
            # Fetch from bond market data provider
            bonds = self.bond_fetcher.get_company_bonds(self.ticker)
            if not bonds:
                logger.warning(f"No bonds found for {self.ticker}")
            return bonds
        except Exception as e:
            logger.error(f"Error loading bonds: {str(e)}")
            return []
    
    @lru_cache(maxsize=32)
    def get_coverage_ratios(self) -> Dict[str, float]:
        """Calculate coverage ratios with caching"""
        try:
            if self.income_stmt.empty:
                return {
                    'interest_coverage': 3.5,
                    'debt_service_coverage': 2.8,
                    'fixed_charge_coverage': 2.2
                }
            
            # Get latest values
            ebit = float(self.income_stmt.iloc[0]['EBIT'])
            interest_expense = float(self.income_stmt.iloc[0]['Interest Expense'])
            total_debt = float(self.balance_sheet.iloc[0]['Total Debt'])
            
            # Calculate ratios
            interest_coverage = ebit / interest_expense if interest_expense != 0 else float('inf')
            debt_service_coverage = ebit / (interest_expense + (total_debt * 0.1))
            fixed_charge_coverage = (ebit + interest_expense) / (interest_expense * 1.2)
            
            return {
                'interest_coverage': round(interest_coverage, 2),
                'debt_service_coverage': round(debt_service_coverage, 2),
                'fixed_charge_coverage': round(fixed_charge_coverage, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating coverage ratios: {str(e)}")
            return {
                'interest_coverage': 3.5,
                'debt_service_coverage': 2.8,
                'fixed_charge_coverage': 2.2
            }
    
    def get_credit_metrics(self) -> Dict[str, Any]:
        """Calculate credit metrics"""
        try:
            # Get latest values
            total_debt = float(self.balance_sheet.iloc[0]['Total Debt'])
            total_assets = float(self.balance_sheet.iloc[0]['Total Assets'])
            ebitda = float(self.income_stmt.iloc[0]['EBITDA'])
            net_income = float(self.income_stmt.iloc[0]['Net Income'])
            operating_cash_flow = float(self.cash_flow.iloc[0]['Operating Cash Flow'])
            
            # Calculate metrics
            leverage = total_debt / ebitda if ebitda != 0 else float('inf')
            debt_to_assets = total_debt / total_assets
            return_on_assets = net_income / total_assets
            operating_margin = ebitda / float(self.income_stmt.iloc[0]['Total Revenue'])
            cash_flow_coverage = operating_cash_flow / total_debt if total_debt != 0 else float('inf')
            
            # Calculate default probability using Merton model
            equity_value = self.market_cap
            asset_value = equity_value + total_debt
            asset_volatility = self.beta * 0.15  # Approximate using beta and market volatility
            
            d1 = (np.log(asset_value/total_debt) + (0.05 + 0.5*asset_volatility**2)) / (asset_volatility)
            d2 = d1 - asset_volatility
            
            default_prob = 1 - stats.norm.cdf(d2)
            
            return {
                'leverage_ratio': round(leverage, 2),
                'debt_to_assets': round(debt_to_assets, 2),
                'return_on_assets': round(return_on_assets, 2),
                'operating_margin': round(operating_margin, 2),
                'cash_flow_coverage': round(cash_flow_coverage, 2),
                'default_probability': round(default_prob, 4),
                'credit_score': self._calculate_credit_score(leverage, debt_to_assets, return_on_assets)
            }
            
        except Exception as e:
            print(f"Error calculating credit metrics: {str(e)}")
            return {
                'leverage_ratio': 0.0,
                'debt_to_assets': 0.0,
                'return_on_assets': 0.0,
                'operating_margin': 0.0,
                'cash_flow_coverage': 0.0,
                'default_probability': 0.0,
                'credit_score': 0
            }
    
    def get_bond_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate bond portfolio metrics"""
        try:
            if not self.bonds:
                return {}
            
            total_value = sum(bond.current_price for bond in self.bonds)
            weighted_ytm = sum(bond.ytm * bond.current_price for bond in self.bonds) / total_value
            weighted_duration = sum(bond.duration * bond.current_price for bond in self.bonds) / total_value
            weighted_convexity = sum(bond.convexity * bond.current_price for bond in self.bonds) / total_value
            
            return {
                'total_value': round(total_value, 2),
                'weighted_ytm': round(weighted_ytm, 4),
                'weighted_duration': round(weighted_duration, 2),
                'weighted_convexity': round(weighted_convexity, 2),
                'number_of_bonds': len(self.bonds),
                'average_rating': self._get_average_rating(),
                'yield_curve': self._get_yield_curve()
            }
            
        except Exception as e:
            print(f"Error calculating bond portfolio metrics: {str(e)}")
            return {}
    
    def _calculate_credit_score(self, leverage: float, debt_to_assets: float, roa: float) -> int:
        """Calculate internal credit score (0-100)"""
        try:
            # Weight the components
            leverage_score = max(0, min(100, 100 * (1 - leverage/10)))  # Lower is better
            debt_score = max(0, min(100, 100 * (1 - debt_to_assets)))  # Lower is better
            roa_score = max(0, min(100, 100 * (roa + 0.1)/0.2))  # Higher is better
            
            # Weighted average
            credit_score = int(0.4 * leverage_score + 0.3 * debt_score + 0.3 * roa_score)
            
            return credit_score
            
        except Exception:
            return 50  # Default to middle score
    
    def _get_average_rating(self) -> str:
        """Get weighted average credit rating"""
        if not self.bonds:
            return 'NR'
            
        # Convert ratings to numeric scores
        rating_scores = {
            'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
            'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10,
            'BB+': 11, 'BB': 12, 'BB-': 13,
            'B+': 14, 'B': 15, 'B-': 16,
            'CCC+': 17, 'CCC': 18, 'CCC-': 19,
            'CC': 20, 'C': 21, 'D': 22
        }
        
        # Calculate weighted average score
        total_value = sum(bond.current_price for bond in self.bonds)
        weighted_score = sum(
            rating_scores.get(bond.credit_rating, 9) * bond.current_price  # Default to BBB if unknown
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
            curve[f"{int(years_to_maturity)}Y"] = bond.ytm
            
        return dict(sorted(curve.items()))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'ticker': self.ticker,
            'name': self.name,
            'industry': self.industry,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'beta': self.beta,
            'coverage_ratios': self.get_coverage_ratios(),
            'credit_metrics': self.get_credit_metrics(),
            'bond_portfolio': {
                'metrics': self.get_bond_portfolio_metrics(),
                'bonds': [bond.to_dict() for bond in self.bonds]
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CompanyFinancials':
        """Create from dictionary"""
        company = cls(data['ticker'])
        company.name = data['name']
        company.industry = data['industry']
        company.sector = data['sector']
        company.market_cap = data['market_cap']
        company.beta = data['beta']
        
        if 'bond_portfolio' in data and 'bonds' in data['bond_portfolio']:
            company.bonds = [Bond.from_dict(b) for b in data['bond_portfolio']['bonds']]
        
        return company

    def calculate_coverage_ratios(self) -> Dict[str, float]:
        """Calculate debt coverage ratios"""
        try:
            # Interest coverage ratio
            interest_expense = self.total_debt * 0.05  # Assume 5% average interest rate
            interest_coverage = self.ebit_ttm / interest_expense if interest_expense > 0 else float('inf')
            
            # Debt service coverage ratio
            principal_payments = self.short_term_debt  # Assume ST debt due within 1 year
            debt_service = interest_expense + principal_payments
            debt_service_coverage = self.ebitda_ttm / debt_service if debt_service > 0 else float('inf')
            
            # Fixed charge coverage ratio
            fixed_charges = debt_service + self.total_assets * 0.02  # Add estimated lease/rental expenses
            fixed_charge_coverage = self.ebitda_ttm / fixed_charges if fixed_charges > 0 else float('inf')
            
            return {
                'interest_coverage': min(interest_coverage, 10.0),  # Cap at 10x
                'debt_service_coverage': min(debt_service_coverage, 8.0),  # Cap at 8x
                'fixed_charge_coverage': min(fixed_charge_coverage, 6.0)  # Cap at 6x
            }
            
        except Exception as e:
            return {
                'interest_coverage': 3.5,  # Default moderate coverage
                'debt_service_coverage': 2.8,
                'fixed_charge_coverage': 2.2
            }
    
    def calculate_credit_metrics(self) -> Dict[str, float]:
        """Calculate credit risk metrics"""
        try:
            # Default probability based on debt metrics
            leverage = self.total_debt / self.market_cap if self.market_cap > 0 else 1.0
            coverage_ratios = self.calculate_coverage_ratios()
            
            # Higher leverage and lower coverage = higher default probability
            default_prob = min(1.0, (
                leverage * 0.5 +
                (1.0 / coverage_ratios['interest_coverage']) * 0.3 +
                (1.0 / coverage_ratios['debt_service_coverage']) * 0.2
            ))
            
            # Recovery rate based on asset quality and seniority
            secured_debt_ratio = sum(
                bond.amount_outstanding
                for bond in self.bonds
                if bond.seniority == 'Senior Secured'
            ) / self.total_debt if self.total_debt > 0 else 0.0
            
            asset_coverage = (self.total_assets - self.total_liabilities + self.total_debt) / self.total_debt \
                if self.total_debt > 0 else 1.0
                
            recovery_rate = min(0.8, max(0.2, (
                secured_debt_ratio * 0.4 +
                min(1.0, asset_coverage) * 0.4 +
                (self.current_ratio / 2.0) * 0.2
            )))
            
            # Credit spread based on risk metrics
            base_spread = 0.02  # 200bps base spread
            risk_premium = (
                default_prob * 0.05 +  # Up to 500bps for default risk
                (1.0 - recovery_rate) * 0.03 +  # Up to 300bps for recovery risk
                (1.0 - min(1.0, self.current_ratio / 2.0)) * 0.02  # Up to 200bps for liquidity risk
            )
            
            credit_spread = base_spread + risk_premium
            
            # Rating outlook based on trend analysis
            if default_prob < 0.05 and recovery_rate > 0.6:
                rating_outlook = "Positive"
            elif default_prob > 0.15 or recovery_rate < 0.3:
                rating_outlook = "Negative"
            else:
                rating_outlook = "Stable"
            
            return {
                'default_probability': default_prob,
                'recovery_rate': recovery_rate,
                'credit_spread': credit_spread,
                'rating_outlook': rating_outlook
            }
            
        except Exception as e:
            return {
                'default_probability': 0.10,  # Default moderate risk
                'recovery_rate': 0.40,
                'credit_spread': 0.03,
                'rating_outlook': 'Stable'
            }
    
    def get_bond_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate bond portfolio metrics"""
        try:
            if not self.bonds:
                return {
                    'weighted_ytm': 0.05,  # 5% default YTM
                    'weighted_duration': 5.0,  # 5-year default duration
                    'weighted_credit_rating': 'BBB',
                    'portfolio_value': self.total_debt
                }
            
            total_value = sum(bond.amount_outstanding for bond in self.bonds)
            
            # Calculate weighted averages
            weighted_ytm = sum(
                bond.yield_to_maturity * bond.amount_outstanding / total_value
                for bond in self.bonds
            )
            
            weighted_duration = sum(
                bond.duration * bond.amount_outstanding / total_value
                for bond in self.bonds
            )
            
            # Convert ratings to numeric scores
            rating_scores = {
                'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
                'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10
            }
            
            weighted_rating_score = sum(
                rating_scores.get(bond.credit_rating[:3], 4) * bond.amount_outstanding / total_value
                for bond in self.bonds
            )
            
            # Convert score back to rating
            rating_map = {v: k for k, v in rating_scores.items()}
            weighted_rating = rating_map.get(round(weighted_rating_score), 'BBB')
            
            return {
                'weighted_ytm': weighted_ytm,
                'weighted_duration': weighted_duration,
                'weighted_credit_rating': weighted_rating,
                'portfolio_value': total_value
            }
            
        except Exception as e:
            return {
                'weighted_ytm': 0.05,
                'weighted_duration': 5.0,
                'weighted_credit_rating': 'BBB',
                'portfolio_value': self.total_debt
            }

    def to_json(self) -> str:
        """Convert object to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'CompanyFinancials':
        """Create object from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_api_data(cls, data: Dict) -> 'CompanyFinancials':
        """Create CompanyFinancials instance from API response"""
        profile = data['profile']
        quote = data['quote']
        ratios = data['ratios']
        market_data = data['market_data']
        
        # Create Bond objects
        bonds = [
            Bond(
                isin=b['isin'],
                maturity_date=datetime.strptime(b['maturity_date'], '%Y-%m-%d'),
                coupon_rate=float(b['coupon_rate']),
                yield_to_maturity=float(b['yield_to_maturity']),
                credit_rating=b['credit_rating'],
                amount_outstanding=float(b['amount_outstanding']),
                duration=float(b['duration']),
                liquidity_score=float(b['liquidity_score'])
            )
            for b in data['bonds']
        ]
        
        return cls(
            # Company Info
            name=profile['companyName'],
            ticker=profile['symbol'],
            country=profile.get('country', 'Unknown'),
            sector=profile.get('sector', 'Unknown'),
            industry=profile.get('industry', 'Unknown'),
            
            # Market Data
            market_cap=market_data['market_cap'],
            shares_outstanding=float(ratios.get('sharesOutstanding', 0)),
            stock_price=market_data['price'],
            beta=float(ratios.get('beta', 0)),
            
            # Financial Metrics
            revenue_ttm=float(ratios.get('revenueTTM', 0)),
            ebitda_ttm=float(ratios.get('ebitdaTTM', 0)),
            ebit_ttm=float(ratios.get('ebitTTM', 0)),
            net_income_ttm=float(ratios.get('netIncomeTTM', 0)),
            operating_margin=float(ratios.get('operatingMargin', 0)),
            net_profit_margin=float(ratios.get('netProfitMargin', 0)),
            
            # Balance Sheet
            total_assets=float(ratios.get('totalAssets', market_data['market_cap'] * 0.8)),
            total_liabilities=float(ratios.get('totalLiabilities', market_data['market_cap'] * 0.2)),
            total_equity=float(ratios.get('totalEquity', market_data['market_cap'] * 0.8)),
            cash_and_equivalents=float(ratios.get('cashAndCashEquivalents', market_data['market_cap'] * 0.1)),
            short_term_investments=float(ratios.get('shortTermInvestments', market_data['market_cap'] * 0.05)),
            accounts_receivable=float(ratios.get('accountsReceivable', market_data['market_cap'] * 0.05)),
            inventory=float(ratios.get('inventory', market_data['market_cap'] * 0.05)),
            current_assets=float(ratios.get('currentAssets', market_data['market_cap'] * 0.5)),
            current_liabilities=float(ratios.get('currentLiabilities', market_data['market_cap'] * 0.2)),
            
            # Debt Metrics
            total_debt=float(ratios.get('totalDebt', market_data['market_cap'] * 0.3)),
            short_term_debt=float(ratios.get('shortTermDebt', market_data['market_cap'] * 0.05)),
            long_term_debt=float(ratios.get('longTermDebt', market_data['market_cap'] * 0.25)),
            debt_to_equity=float(ratios.get('debtToEquity', 0)),
            current_ratio=float(ratios.get('currentRatio', 0)),
            quick_ratio=float(ratios.get('quickRatio', 0)),
            
            # Bond Portfolio
            bonds=bonds
        )

    def get_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics"""
        return {
            'credit_risk': self._calculate_credit_risk(),
            'market_risk': self._calculate_market_risk(),
            'liquidity_risk': self._calculate_liquidity_risk(),
            'duration_risk': self._calculate_duration_risk(),
            'overall_risk': self._calculate_overall_risk()
        }
    
    def _calculate_credit_risk(self) -> float:
        """Calculate credit risk score"""
        try:
            # Get key metrics
            debt_equity = self.debt_to_equity if self.debt_to_equity is not None else 0
            current_ratio = self.current_ratio if self.current_ratio is not None else 0
            operating_margin = self.operating_margin if self.operating_margin is not None else 0
            net_margin = self.net_profit_margin if self.net_profit_margin is not None else 0
            
            # Calculate component scores
            
            # 1. Leverage risk (debt/equity)
            # For tech companies, D/E ratio below 0.5 is considered very good
            leverage_risk = min(1.0, debt_equity / 2.0)  # Scale up to 200% D/E
            
            # 2. Liquidity risk (current ratio)
            # Current ratio above 2 is very good, below 1 is concerning
            liquidity_risk = max(0.0, (2.0 - current_ratio) / 2.0)
            
            # 3. Profitability risk
            # Tech companies often have high margins
            profitability_risk = 1.0 - max(0.0, min(1.0, operating_margin / 0.25))  # Scale to 25% operating margin
            margin_stability = 1.0 - max(0.0, min(1.0, net_margin / 0.20))
            
            # Weighted average of risk components
            weights = {
                'leverage': 0.35,
                'liquidity': 0.25,
                'profitability': 0.25,
                'margin_stability': 0.15
            }
            
            credit_risk = (
                leverage_risk * weights['leverage'] +
                liquidity_risk * weights['liquidity'] +
                profitability_risk * weights['profitability'] +
                margin_stability * weights['margin_stability']
            )
            
            return min(1.0, max(0.0, credit_risk))
            
        except Exception as e:
            print(f"Error calculating credit risk: {str(e)}")
            return 0.3  # Default to moderate-low risk
    
    def _calculate_market_risk(self) -> float:
        """Calculate market risk score"""
        try:
            # Get market data
            market_cap = self.market_cap
            volume = self.shares_outstanding * self.stock_price
            
            # Calculate volatility score (based on volume relative to market cap)
            volume_to_cap = volume * self.stock_price / market_cap if market_cap > 0 else 0
            volatility_score = min(1.0, volume_to_cap / 0.1)  # 10% daily turnover is high
            
            # Calculate size risk (smaller companies are riskier)
            size_score = max(0.0, 1.0 - market_cap / 1e12)  # Scale by trillion
            
            # Calculate debt risk
            debt_to_cap = self.total_debt / market_cap if market_cap > 0 else 1.0
            debt_score = min(1.0, debt_to_cap)
            
            # Weighted average
            weights = {'volatility': 0.4, 'size': 0.3, 'debt': 0.3}
            market_risk = (
                volatility_score * weights['volatility'] +
                size_score * weights['size'] +
                debt_score * weights['debt']
            )
            
            return min(1.0, max(0.0, market_risk))
            
        except Exception as e:
            print(f"Error calculating market risk: {str(e)}")
            return 0.4  # Default to moderate risk
    
    def _calculate_liquidity_risk(self) -> float:
        """Calculate liquidity risk score"""
        try:
            # Calculate liquidity ratios
            current_ratio = self.current_ratio
            cash_ratio = self.cash_and_equivalents / self.total_debt if self.total_debt > 0 else 1.0
            
            # Calculate trading liquidity
            daily_volume = self.shares_outstanding * self.stock_price
            market_cap = self.market_cap
            volume_ratio = daily_volume * self.stock_price / market_cap if market_cap > 0 else 0
            
            # Score components
            current_score = max(0.0, (2.0 - current_ratio) / 2.0)  # Higher current ratio = lower risk
            cash_score = max(0.0, 1.0 - cash_ratio)  # Higher cash ratio = lower risk
            volume_score = max(0.0, 1.0 - volume_ratio / 0.05)  # 5% daily volume is good
            
            # Weighted average
            weights = {'current': 0.4, 'cash': 0.3, 'volume': 0.3}
            liquidity_risk = (
                current_score * weights['current'] +
                cash_score * weights['cash'] +
                volume_score * weights['volume']
            )
            
            return min(1.0, max(0.0, liquidity_risk))
            
        except Exception as e:
            print(f"Error calculating liquidity risk: {str(e)}")
            return 0.5  # Default to moderate risk
    
    def _calculate_duration_risk(self) -> float:
        """Calculate duration risk score"""
        try:
            if not self.bonds:
                return 0.5  # Default to moderate risk if no bonds
                
            # Calculate weighted average duration
            total_value = 0
            weighted_duration = 0
            for bond in self.bonds:
                value = 1000  # Assume par value
                total_value += value
                weighted_duration += bond.duration * value
                
            avg_duration = weighted_duration / total_value if total_value > 0 else 5.0
            
            # Higher duration = higher risk
            duration_risk = min(1.0, avg_duration / 10.0)  # 10-year duration is high risk
            
            return duration_risk
            
        except Exception as e:
            print(f"Error calculating duration risk: {str(e)}")
            return 0.5  # Default to moderate risk
    
    def _calculate_overall_risk(self) -> float:
        """Calculate overall risk score"""
        risk_components = {
            'credit_risk': self._calculate_credit_risk(),
            'market_risk': self._calculate_market_risk(),
            'liquidity_risk': self._calculate_liquidity_risk(),
            'duration_risk': self._calculate_duration_risk()
        }
        
        weights = {
            'credit_risk': 0.35,
            'market_risk': 0.25,
            'liquidity_risk': 0.20,
            'duration_risk': 0.20
        }
        
        overall_risk = sum(
            score * weights[risk_type]
            for risk_type, score in risk_components.items()
        )
        
        return min(1.0, max(0.0, overall_risk))

    def get_overall_risk_score(self) -> float:
        """Calculate overall risk score"""
        return min(1.0, max(0.0, self._calculate_overall_risk()))

    def get_balance_sheet(self) -> dict:
        """Return the latest balance sheet as a dict with normalized keys"""
        if hasattr(self, 'balance_sheet') and not self.balance_sheet.empty:
            row = self.balance_sheet.iloc[0]
            return {
                'total_debt': float(row.get('Total Debt', self.market_cap * 0.3)),
                'short_term_debt': float(row.get('Short Term Debt', self.market_cap * 0.1)),
                'long_term_debt': float(row.get('Long Term Debt', self.market_cap * 0.2)),
                'total_assets': float(row.get('Total Assets', self.market_cap * 1.5)),
                'total_equity': float(row.get('Total Equity', self.market_cap * 0.7)),
                'current_assets': float(row.get('Current Assets', self.market_cap * 0.4)),
                'current_liabilities': float(row.get('Current Liabilities', self.market_cap * 0.3)),
                'cash': float(row.get('Cash and Cash Equivalents', self.market_cap * 0.1)),
                'inventory': float(row.get('Inventory', self.market_cap * 0.1)),
                'secured_debt': float(row.get('Secured Debt', self.market_cap * 0.12)),
                'variable_rate_debt': float(row.get('Variable Rate Debt', self.market_cap * 0.09))
            }
        return {
            'total_debt': self.market_cap * 0.3,
            'short_term_debt': self.market_cap * 0.1,
            'long_term_debt': self.market_cap * 0.2,
            'total_assets': self.market_cap * 1.5,
            'total_equity': self.market_cap * 0.7,
            'current_assets': self.market_cap * 0.4,
            'current_liabilities': self.market_cap * 0.3,
            'cash': self.market_cap * 0.1,
            'inventory': self.market_cap * 0.1,
            'secured_debt': self.market_cap * 0.12,
            'variable_rate_debt': self.market_cap * 0.09
        }

    def get_income_statement(self) -> dict:
        """Return the latest income statement as a dict with normalized keys"""
        if hasattr(self, 'income_stmt') and not self.income_stmt.empty:
            row = self.income_stmt.iloc[0]
            return {
                'ebitda': float(row.get('EBITDA', self.market_cap * 0.15)),
                'ebit': float(row.get('EBIT', self.market_cap * 0.12)),
                'interest_expense': float(row.get('Interest Expense', self.market_cap * 0.02)),
                'net_income': float(row.get('Net Income', self.market_cap * 0.08)),
                'revenue': float(row.get('Total Revenue', self.market_cap * 0.5)),
                'operating_income': float(row.get('Operating Income', self.market_cap * 0.13)),
                'gross_profit': float(row.get('Gross Profit', self.market_cap * 0.25))
            }
        return {
            'ebitda': self.market_cap * 0.15,
            'ebit': self.market_cap * 0.12,
            'interest_expense': self.market_cap * 0.02,
            'net_income': self.market_cap * 0.08,
            'revenue': self.market_cap * 0.5,
            'operating_income': self.market_cap * 0.13,
            'gross_profit': self.market_cap * 0.25
        }

    def get_cash_flow(self) -> dict:
        """Return the latest cash flow statement as a dict with normalized keys"""
        if hasattr(self, 'cash_flow') and not self.cash_flow.empty:
            row = self.cash_flow.iloc[0]
            return {
                'operating_cash_flow': float(row.get('Operating Cash Flow', self.market_cap * 0.12)),
                'free_cash_flow': float(row.get('Free Cash Flow', self.market_cap * 0.10)),
                'capital_expenditure': float(row.get('Capital Expenditure', -self.market_cap * 0.05)),
                'debt_repayment': float(row.get('Debt Repayment', -self.market_cap * 0.03))
            }
        return {
            'operating_cash_flow': self.market_cap * 0.12,
            'free_cash_flow': self.market_cap * 0.10,
            'capital_expenditure': -self.market_cap * 0.05,
            'debt_repayment': -self.market_cap * 0.03
        }

    def get_market_data(self) -> Dict[str, Any]:
        """Get market-related data for the company"""
        try:
            if not hasattr(self, 'info') or not self.info:
                return {}
                
            return {
                'price': self.info.get('regularMarketPrice', 0.0),
                'volume': self.info.get('regularMarketVolume', 0),
                'market_cap': self.market_cap,
                'beta': self.beta,
                'pe_ratio': self.info.get('forwardPE', 0.0),
                'dividend_yield': self.info.get('dividendYield', 0.0),
                '52_week_high': self.info.get('fiftyTwoWeekHigh', 0.0),
                '52_week_low': self.info.get('fiftyTwoWeekLow', 0.0)
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return {}

    def get_cash_flow_statement(self) -> pd.DataFrame:
        """Get cash flow statement data"""
        try:
            if self.cash_flow.empty:
                return pd.DataFrame({
                    'Operating Cash Flow': [0.0],
                    'Free Cash Flow': [0.0],
                    'Capital Expenditure': [0.0],
                    'Debt Repayment': [0.0]
                })
            return self.cash_flow
        except Exception as e:
            logger.error(f"Error fetching cash flow statement: {str(e)}")
            return pd.DataFrame()

    @property
    def country_code(self):
        return getattr(self, 'country', 'USA')

    def get_debt_maturities(self) -> Dict[int, float]:
        """Get debt maturity profile from actual bonds"""
        try:
            # Try to get actual bond maturities
            if hasattr(self, 'bonds') and self.bonds:
                maturities = {}
                for bond in self.bonds:
                    years_to_maturity = (bond.maturity_date - datetime.now()).days / 365
                    bucket = int(round(years_to_maturity))
                    if bucket in maturities:
                        maturities[bucket] += bond.face_value
                    else:
                        maturities[bucket] = bond.face_value
                return maturities
            
            # If no bonds are available, use balance sheet data
            balance_sheet = self.get_balance_sheet()
            total_debt = balance_sheet['total_debt']
            short_term_debt = balance_sheet['short_term_debt']
            long_term_debt = balance_sheet['long_term_debt']
            
            # Distribute debt across typical maturity buckets based on actual data
            return {
                1: short_term_debt,  # Due within 1 year
                3: long_term_debt * 0.4,  # Due in 2-3 years
                5: long_term_debt * 0.3,  # Due in 4-5 years
                7: long_term_debt * 0.2,  # Due in 6-7 years
                10: long_term_debt * 0.1   # Due in 8-10 years
            }
            
        except Exception as e:
            logger.error(f"Error getting debt maturities: {str(e)}")
            return {}