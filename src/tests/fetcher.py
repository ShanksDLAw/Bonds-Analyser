import requests
import os
import sys
from dataclasses import dataclass
from datetime import datetime
import datetime as dt
from typing import List, Optional, Dict, Tuple
from tenacity import retry, wait_exponential, stop_after_attempt
import logging
import pandas as pd
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)

from src.config import Config
from src.data.cache import CacheManager
from src.data.country_fetcher import TreasuryClient
from src.models.company import CompanyFinancials, Bond
from src.macro.macro_router import MacroRouter

@dataclass
class BondData:
    isin: str
    maturity_date: str
    coupon_rate: float
    yield_to_maturity: float
    duration: float
    credit_rating: str
    amount_outstanding: float
    liquidity_score: float

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

class BondRiskFetcher:
    """Quant-grade data fetcher with integrated FMP API"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.treasury = TreasuryClient()
        self.macro_router = MacroRouter()
        self.config = Config()
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.api_key = self.config.fmp_key()
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found in configuration")

    def _make_api_request(self, endpoint: str) -> Dict:
        """Make a rate-limited API request"""
        if not self.cache.check_rate_limit():
            raise APIError("Daily API call limit (250) exceeded. Please try again tomorrow.")
            
        url = f"{self.base_url}/{endpoint}?apikey={self.api_key}"
        response = requests.get(url)
        
        if response.ok:
            self.cache.increment_rate_limit()
            return response.json()
        else:
            logger.error(f"API request failed: {response.status_code}")
            raise APIError(f"API request failed: {response.status_code}")

    def _search_company_symbol(self, name: str) -> Optional[str]:
        """Search for company symbol using FMP API"""
        cache_key = f"search_{name.lower()}"
        cached = self.cache.get(cache_key)
        if cached and isinstance(cached, str):
            return cached
            
        try:
            data = self._make_api_request(f"search?query={name}&limit=1")
            if not data:
                return None
                
            symbol = data[0]['symbol']
            self.cache.set(cache_key, symbol)
            return symbol
            
        except Exception as e:
            logger.error(f"Error searching for symbol: {str(e)}")
            return None
            
    def _get_company_profile(self, symbol: str) -> Dict:
        """Get company profile from FMP API"""
        try:
            data = self._make_api_request(f"profile/{symbol}")
            if not data:
                raise APIError(f"No profile data found for {symbol}")
            return data[0]
            
        except Exception as e:
            logger.error(f"Error fetching company profile: {str(e)}")
            raise APIError(f"Failed to fetch company profile: {str(e)}")

    def _get_company_metrics(self, symbol: str) -> Dict:
        """Get company financial metrics from FMP API"""
        try:
            data = self._make_api_request(f"key-metrics-ttm/{symbol}")
            if not data:
                raise APIError(f"No metrics found for {symbol}")
            return data[0]
            
        except Exception as e:
            logger.error(f"Error fetching company metrics: {str(e)}")
            raise APIError(f"Failed to fetch company metrics: {str(e)}")

    def _get_company_bonds(self, symbol: str) -> List[BondData]:
        """Get company bonds from FMP API"""
        try:
            data = self._make_api_request(f"bond/{symbol}")
            if not data:
                logger.warning(f"No bond data found for {symbol}")
                return []
                
            bonds = []
            for bond in data:
                try:
                    maturity_date = bond.get('maturityDate', '')
                    if not maturity_date:
                        continue
                        
                    bonds.append(BondData(
                        isin=bond.get('isin', ''),
                        maturity_date=maturity_date,
                        coupon_rate=float(bond.get('couponRate', 0)),
                        yield_to_maturity=float(bond.get('yield', 0)),
                        duration=float(bond.get('duration', 0)),
                        credit_rating=bond.get('rating', 'NR'),
                        amount_outstanding=float(bond.get('outstandingAmount', 0)),
                        liquidity_score=float(bond.get('liquidityScore', 0.5))
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error parsing bond data: {str(e)}")
                    continue
                    
            return bonds
            
        except Exception as e:
            logger.error(f"Error fetching bonds: {str(e)}")
            return []

    def _get_risk_free_rate(self) -> float:
        """Get current risk-free rate"""
        try:
            data = self._make_api_request("treasury")
            if not data:
                return self.macro_router.get_risk_free_rate('10Y')
                
            # Get 10-year Treasury rate
            for rate in data:
                if rate.get('maturity') == '10 years':
                    return float(rate.get('rate', 3.5)) / 100
                    
            return self.macro_router.get_risk_free_rate('10Y')
            
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {str(e)}")
            return self.macro_router.get_risk_free_rate('10Y')

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), 
           stop=stop_after_attempt(3))
    def get_company_data(self, company_name: str) -> CompanyFinancials:
        """Get comprehensive company data including bonds"""
        try:
            # Step 1: Search for company
            symbol = self._search_company_symbol(company_name)
            if not symbol:
                raise APIError(f"No symbol found for {company_name}")
            
            # Step 2: Get company profile and metrics
            profile = self._get_company_profile(symbol)
            metrics = self._get_company_metrics(symbol)
            
            # Step 3: Get bond data
            bonds = self._get_company_bonds(symbol)
            
            # Convert BondData to Bond objects
            bond_objects = []
            for bond in bonds:
                try:
                    bond_objects.append(Bond(
                        isin=bond.isin,
                        maturity_date=datetime.strptime(bond.maturity_date, '%Y-%m-%d'),
                        coupon_rate=bond.coupon_rate,
                        yield_to_maturity=bond.yield_to_maturity,
                        credit_rating=bond.credit_rating,
                        amount_outstanding=bond.amount_outstanding,
                        duration=bond.duration,
                        liquidity_score=bond.liquidity_score
                    ))
                except ValueError as e:
                    logger.warning(f"Error converting bond data: {str(e)}")
                    continue
            
            # Create CompanyFinancials object
            company = CompanyFinancials(
                symbol=symbol,
                name=profile.get('companyName', company_name),
                sector=profile.get('sector', 'Unknown'),
                industry=profile.get('industry', 'Unknown'),
                country=profile.get('country', 'USA'),
                market_cap=float(profile.get('mktCap', 0)),
                price=float(profile.get('price', 0)),
                volume=float(profile.get('volAvg', 0)),
                pe_ratio=float(metrics.get('peRatioTTM', 0)),
                debt_to_equity=float(metrics.get('debtToEquityTTM', 0)),
                current_ratio=float(metrics.get('currentRatioTTM', 0)),
                operating_margin=float(metrics.get('operatingMarginTTM', 0)),
                net_profit_margin=float(metrics.get('netProfitMarginTTM', 0)),
                total_debt=float(metrics.get('totalDebtTTM', 0)),
                long_term_debt=float(metrics.get('longTermDebtTTM', 0)),
                short_term_debt=float(metrics.get('shortTermDebtTTM', 0)),
                cash_and_equivalents=float(metrics.get('cashAndCashEquivalentsTTM', 0)),
                bonds=bond_objects,
                risk_free_rate=self._get_risk_free_rate()
            )
            
            return company

        except APIError as e:
            logger.error(f"API error for {company_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {company_name}: {str(e)}")
            raise APIError(f"Failed to fetch company data: {str(e)}")

# Test function
if __name__ == "__main__":
    print("BondRiskFetcher Test")
    print("-" * 50)
    
    # Create fetcher instance
    fetcher = BondRiskFetcher()
    
    # Test company search
    test_companies = ["AAPL", "MSFT", "GOOGL"]
    
    for company in test_companies:
        print(f"\nTesting with {company}...")
        try:
            data = fetcher.get_company_data(company)
            print(f"Successfully fetched data for {data.name}")
            print(f"Market Cap: ${data.market_cap:,.2f}")
            print(f"P/E Ratio: {data.pe_ratio:.2f}")
            print(f"Debt/Equity: {data.debt_to_equity:.2f}")
            if data.bonds:
                print(f"\nFound {len(data.bonds)} bonds")
                for bond in data.bonds:
                    print(f"Bond: {bond.isin}, YTM: {bond.yield_to_maturity*100:.2f}%")
            else:
                print("No bonds found")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error testing {company}: {str(e)}")
            continue