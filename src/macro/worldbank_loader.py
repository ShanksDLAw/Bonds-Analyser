"""World Bank data loader module."""

import wbgapi
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import functools
import time
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CountryMetadata:
    """Country metadata"""
    region: str = "Unknown"
    income_group: str = "Unknown"
    lending_type: str = "Unknown"

class WorldBankClient:
    """Client for fetching World Bank data with caching and fallbacks"""
    
    def __init__(self):
        """Initialize client"""
        self.wb = wbgapi
        self.cache_duration = 3600  # 1 hour cache
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 500ms between requests
        
        # Key economic indicators with fallback values and validation ranges
        # Remove hardcoded fallbacks
        self.indicators = {
            'GDP_GROWTH': {
                'code': 'NY.GDP.MKTP.KD.ZG',
                'min_value': -15.0,
                'max_value': 15.0
            },
            'INFLATION': {
                'code': 'FP.CPI.TOTL.ZG',
                'fallback': 2.5,
                'min_value': -2.0,
                'max_value': 20.0
            },
            'UNEMPLOYMENT': {
                'code': 'SL.UEM.TOTL.ZS',
                'fallback': 5.0,
                'min_value': 0.0,
                'max_value': 30.0
            },
            'DEBT_TO_GDP': {
                'code': 'GC.DOD.TOTL.GD.ZS',
                'fallback': 60.0,
                'min_value': 0.0,
                'max_value': 200.0
            },
            'CURRENT_ACCOUNT': {
                'code': 'BN.CAB.XOKA.GD.ZS',
                'fallback': 0.0,
                'min_value': -20.0,
                'max_value': 20.0
            },
            'RESERVES': {
                'code': 'FI.RES.TOTL.CD',
                'fallback': 100e9,
                'min_value': 0.0,
                'max_value': 5e12
            },
            'POLICY_RATE': {
                'code': 'FR.INR.RINR',
                'fallback': 3.0,
                'min_value': -2.0,
                'max_value': 15.0
            },
            'BOND_YIELD': {
                'code': 'FR.INR.BOND',
                'fallback': 4.0,
                'min_value': -1.0,
                'max_value': 15.0
            }
        }
        
        # Initialize cache
        self._cache = {}
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last_request)
        self._last_request_time = time.time()
    
    def _get_cached_value(self, cache_key: str) -> Optional[Any]:
        """Get value from cache if still valid"""
        if cache_key in self._cache:
            timestamp, value = self._cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return value
            else:
                del self._cache[cache_key]
        return None
    
    def _set_cached_value(self, cache_key: str, value: Any):
        """Set value in cache"""
        self._cache[cache_key] = (time.time(), value)
    
    def _validate_value(
        self,
        indicator: str,
        value: Optional[float]
    ) -> float:
        """Validate and clean indicator value"""
        if value is None:
            return self.indicators[indicator]['fallback']
            
        try:
            value = float(value)
            min_val = self.indicators[indicator]['min_value']
            max_val = self.indicators[indicator]['max_value']
            
            if np.isnan(value) or value < min_val or value > max_val:
                return self.indicators[indicator]['fallback']
                
            return value
            
        except (ValueError, TypeError):
            return self.indicators[indicator]['fallback']
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_indicator(self, indicator_code: str, country_code: str, year: int) -> Optional[float]:
        """Fetch single indicator with improved error handling"""
        self._rate_limit()
        
        try:
            result = self.wb.data.get(
                indicator_code,
                economy=country_code,
                time=year,
                skipBlanks=True
            )
            
            if not result:
                logger.warning(f"No data found for {indicator_code} in {country_code} for {year}")
                return None
                
            # Get the most recent non-null value
            values = [v for v in result.values() if v[0] is not None]
            if not values:
                return None
                
            try:
                return float(values[0][0])
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error converting value for {indicator_code}: {str(e)}")
                return None
                
        except Exception as e:
            logger.warning(f"Error fetching indicator {indicator_code}: {str(e)}")
            return None
    
    def get_country_data(self, country_code: str, year: Optional[int] = None) -> Dict[str, Union[float, str]]:
        """Get comprehensive country data with improved fallback handling"""
        try:
            if year is None:
                year = datetime.now().year - 1
            
            cache_key = f"{country_code}_{year}"
            cached_data = self._get_cached_value(cache_key)
            if cached_data is not None:
                return cached_data
            
            data = {}
            
            # Fetch all indicators with individual error handling
            for indicator_name, indicator_info in self.indicators.items():
                try:
                    value = self._fetch_indicator(indicator_info['code'], country_code, year)
                    data[indicator_name] = self._validate_value(indicator_name, value)
                except Exception as e:
                    logger.warning(f"Error fetching {indicator_name}: {str(e)}")
                    data[indicator_name] = indicator_info.get('fallback', 0.0)
            
            # Get country metadata with separate error handling
            try:
                country_info = self.wb.economy.get(country_code)
                metadata = CountryMetadata(
                    region=country_info.get('region', {}).get('value', 'Unknown'),
                    income_group=country_info.get('incomeLevel', {}).get('value', 'Unknown'),
                    lending_type=country_info.get('lendingType', {}).get('value', 'Unknown')
                )
                data.update({
                    'REGION': metadata.region,
                    'INCOME_GROUP': metadata.income_group,
                    'LENDING_TYPE': metadata.lending_type
                })
            except Exception as e:
                logger.warning(f"Error fetching country metadata: {str(e)}")
                data.update({
                    'REGION': 'Unknown',
                    'INCOME_GROUP': 'Unknown',
                    'LENDING_TYPE': 'Unknown'
                })
            
            self._set_cached_value(cache_key, data)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching World Bank data: {str(e)}")
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> Dict[str, Union[float, str]]:
        """Get fallback data when API fails"""
        fallback_data = {}
        
        # Add fallback values for all indicators
        for name, info in self.indicators.items():
            if 'fallback' in info:
                fallback_data[name] = info['fallback']
            else:
                # Default fallbacks based on indicator type
                if 'GROWTH' in name:
                    fallback_data[name] = 2.5
                elif 'INFLATION' in name:
                    fallback_data[name] = 2.0
                elif 'DEBT' in name:
                    fallback_data[name] = 60.0
                elif 'RATE' in name:
                    fallback_data[name] = 3.0
                else:
                    fallback_data[name] = 0.0
        
        # Add metadata fallbacks
        fallback_data.update({
            'REGION': 'Unknown',
            'INCOME_GROUP': 'Unknown',
            'LENDING_TYPE': 'Unknown'
        })
        
        return fallback_data
    
    def get_historical_data(
        self,
        country_code: str,
        indicator: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """Get historical data for an indicator with validation"""
        try:
            cache_key = f"{country_code}_{indicator}_{start_year}_{end_year}"
            cached_data = self._get_cached_value(cache_key)
            if cached_data is not None:
                return cached_data
            
            if indicator not in self.indicators:
                return pd.DataFrame()
            
            self._rate_limit()
            
            data = self.wb.data.get(
                self.indicators[indicator]['code'],
                economy=country_code,
                time=range(start_year, end_year + 1),
                skipBlanks=True
            )
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame and validate
            df = pd.DataFrame(data).T
            df.columns = ['value']
            df.index = pd.to_datetime(df.index + '-12-31')
            
            # Clean values
            df['value'] = df['value'].apply(
                lambda x: self._validate_value(indicator, x)
            )
            
            # Cache the results
            self._set_cached_value(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_countries(
        self,
        country_codes: List[str],
        indicators: Optional[List[str]] = None,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """Get data for multiple countries with parallel processing"""
        try:
            if year is None:
                year = datetime.now().year - 1
            
            if indicators is None:
                indicators = list(self.indicators.keys())
            
            data = []
            for country in country_codes:
                country_data = self.get_country_data(country, year)
                country_data['COUNTRY'] = country
                data.append(country_data)
            
            df = pd.DataFrame(data)
            
            # Ensure all indicators are present with fallback values
            for indicator in indicators:
                if indicator not in df.columns and indicator in self.indicators:
                    df[indicator] = self.indicators[indicator]['fallback']
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching multiple countries: {str(e)}")
            return pd.DataFrame()