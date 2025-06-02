"""IMF data loader module for Bond Risk Analyzer"""

import requests
import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.config import Config, ConfigError

# Configure logging
logger = logging.getLogger(__name__)

class IMFError(Exception):
    """Custom exception for IMF API errors"""
    pass

class IMFClient:
    """Client for fetching data from IMF's Financial Soundness Indicators (FSI) API"""
    
    def __init__(self):
        """Initialize IMF client with configuration"""
        try:
            self.config = Config()
            self.base_url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
            self.session = requests.Session()
            
            # Default parameters
            self.timeout = 30  # seconds
            self.max_retries = 3
            self.retry_delay = 1  # seconds
            
        except ConfigError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise IMFError(f"Failed to initialize IMF client: {str(e)}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def _fetch_series(
        self,
        country_code: str,
        series_id: str,
        start_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """Fetch time series data from IMF FSI API
        
        Args:
            country_code: ISO country code
            series_id: IMF series identifier
            start_date: Optional start date in YYYY-MM-DD format
            
        Returns:
            Time series data or None if not available
            
        Raises:
            IMFError: If there is an error fetching the data
        """
        try:
            # Build URL
            url = f'{self.base_url}CompactData/FSI/Q.{country_code}.{series_id}'
            params = {}
            if start_date:
                params['startPeriod'] = start_date
            
            # Make request
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            # Check response
            if response.status_code == 404:
                logger.warning(f"No data found for {series_id} in {country_code}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            if not self._validate_response(data):
                logger.warning(
                    f"Invalid response structure for {series_id} in {country_code}"
                )
                return None
            
            # Extract observations
            obs = data['CompactData']['DataSet']['Series']['Obs']
            if not obs:
                logger.warning(f"No observations for {series_id} in {country_code}")
                return None
            
            # Parse observations
            series_data = {}
            for ob in obs:
                try:
                    date = pd.to_datetime(ob['@TIME_PERIOD'])
                    value = float(ob['@OBS_VALUE'])
                    series_data[date] = value
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"Error parsing observation for {series_id}: {str(e)}"
                    )
                    continue
            
            if not series_data:
                logger.warning(f"No valid observations for {series_id}")
                return None
            
            return pd.Series(series_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {series_id}: {str(e)}")
            raise IMFError(f"Failed to fetch {series_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error for {series_id}: {str(e)}")
            raise IMFError(f"Error processing {series_id}: {str(e)}")
    
    def _validate_response(self, data: Dict[str, Any]) -> bool:
        """Validate IMF API response structure
        
        Args:
            data: Response JSON data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            return (
                isinstance(data, dict) and
                'CompactData' in data and
                'DataSet' in data['CompactData'] and
                'Series' in data['CompactData']['DataSet'] and
                'Obs' in data['CompactData']['DataSet']['Series']
            )
        except Exception as e:
            logger.warning(f"Error validating response: {str(e)}")
            return False
    
    def get_financial_soundness(self, country_code: str) -> Dict[str, float]:
        """Get latest financial soundness indicators for a country
        
        Args:
            country_code: ISO country code
            
        Returns:
            Dictionary of financial soundness indicators
        """
        try:
            # Define indicators to fetch
            indicators = {
                'FSANL_PT': 'non_performing_loans',
                'FSCAR_PT': 'capital_adequacy_ratio',
                'FSRL_PT': 'return_on_loans',
                'FSKA_PT': 'capital_to_assets',
                'FSDEP_PT': 'deposits_to_assets'
            }
            
            results = {}
            for series_id, name in indicators.items():
                series = self._fetch_series(country_code, series_id)
                if series is not None and not series.empty:
                    # Get most recent value
                    results[name] = float(series.iloc[-1])
                else:
                    results[name] = None
            
            return results
            
        except IMFError:
            raise  # Re-raise IMF specific errors
        except Exception as e:
            logger.error(f"Error getting financial soundness: {str(e)}")
            raise IMFError(f"Failed to get financial soundness: {str(e)}")
    
    def get_macro_stability(self, country_code: str) -> Dict[str, float]:
        """Get macroeconomic stability indicators for a country
        
        Args:
            country_code: ISO country code
            
        Returns:
            Dictionary of macro stability indicators
        """
        try:
            # Define indicators to fetch
            indicators = {
                'FSDR_PT': 'debt_to_reserves',
                'FSFX_PT': 'foreign_assets_to_capital',
                'FSGR_PT': 'geographic_risk_ratio',
                'FSLR_PT': 'liquidity_ratio',
                'FSTD_PT': 'trading_income_ratio'
            }
            
            results = {}
            for series_id, name in indicators.items():
                series = self._fetch_series(country_code, series_id)
                if series is not None and not series.empty:
                    # Get most recent value
                    results[name] = float(series.iloc[-1])
                else:
                    results[name] = None
            
            return results
            
        except IMFError:
            raise  # Re-raise IMF specific errors
        except Exception as e:
            logger.error(f"Error getting macro stability: {str(e)}")
            raise IMFError(f"Failed to get macro stability: {str(e)}")
    
    def get_market_risk(self, country_code: str) -> Dict[str, float]:
        """Get market risk indicators for a country
        
        Args:
            country_code: ISO country code
            
        Returns:
            Dictionary of market risk indicators
        """
        try:
            # Define indicators to fetch
            indicators = {
                'FSSI_PT': 'stock_market_volatility',
                'FSER_PT': 'exchange_rate_volatility',
                'FSIR_PT': 'interest_rate_volatility',
                'FSMR_PT': 'market_liquidity_ratio',
                'FSTR_PT': 'trading_volume_ratio'
            }
            
            results = {}
            for series_id, name in indicators.items():
                series = self._fetch_series(country_code, series_id)
                if series is not None and not series.empty:
                    # Get most recent value
                    results[name] = float(series.iloc[-1])
                else:
                    results[name] = None
            
            return results
            
        except IMFError:
            raise  # Re-raise IMF specific errors
        except Exception as e:
            logger.error(f"Error getting market risk: {str(e)}")
            raise IMFError(f"Failed to get market risk: {str(e)}")
    
    def get_country_metrics(self, country_code: str) -> Dict[str, float]:
        """Get comprehensive country metrics from IMF FSI"""
        metrics = {}
        series_ids = {
            'gdp_growth': 'NGDP_RPCH',
            'inflation': 'PCPIPCH',
            'debt_to_gdp': 'GGXWDG_NGDP',
            'reserves': 'RAXG_USD',
            'financial_soundness': 'FSI'
        }
        
        for metric, series_id in series_ids.items():
            metrics[metric] = self._fetch_series(country_code, series_id)
        
        return metrics