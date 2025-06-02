"""Configuration management for Bond Risk Analyzer.

This module handles loading environment variables, managing paths,
and providing access to configuration settings throughout the application.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any, Union, Dict
from dotenv import load_dotenv
import logging
from dataclasses import dataclass
import json

# Configure logging
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Configuration related errors"""
    pass

class Config:
    """Configuration management"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
            
        # Load environment variables
        self._load_env()
        
        # Initialize configuration
        self._initialize_config()
        
        self._initialized = True
        self._validate_config()
        self._setup_directories()
    
    def _initialize_config(self) -> None:
        """Initialize configuration settings"""
        # API keys
        class Config:
            def __init__(self):
                # API configuration
                self.api_keys = {
                    'yahoo_finance': os.getenv('YAHOO_FINANCE_API_KEY', ''),
                }
                
                # Request settings
                self.request_timeout = 30  # Default timeout in seconds
                self.long_request_timeout = 60  # Timeout for long-running requests
                self.max_retries = 3  # Maximum number of retry attempts
                self.retry_delay = 1  # Initial delay between retries in seconds
                self.max_retry_delay = 10  # Maximum delay between retries in seconds
                
                self.timeouts = {
                    'default': 10,  # Default timeout in seconds
                    'long_request': 30  # Timeout for longer operations
                }
                self.retry_settings = {
                    'max_attempts': 3,
                    'wait_time': 2,  # Base wait time between retries in seconds
                    'max_wait_time': 10  # Maximum wait time between retries
                }
                
                self.rate_limits = {
                    # Remove FMP rate limit
                }
        
        # Remove fmp_key method since it's no longer needed
    
        # Risk weights
        self.risk_weights = {
            'credit_risk': 0.50,  # 50% weight
            'duration_risk': 0.30,  # 30% weight
            'liquidity_risk': 0.20  # 20% weight
        }
        
        # API rate limits (requests per second)
        self.rate_limits = {
            'fmp': 5,
            'yahoo_finance': 2,
            'world_bank': 10
        }
        
        # Market data settings
        self.market_data = {
            'default_country': 'USA',
            'benchmark_rate': '10Y',
            'volatility_window': 252  # trading days
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'low_risk': 0.33,
            'medium_risk': 0.66,
            'high_risk': 1.00
        }
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour default
    
    @staticmethod
    def _load_env() -> None:
        """Load environment variables from .env file"""
        try:
            # Try to find .env file in current directory and parent directories
            current_dir = Path.cwd()
            env_file = None
            
            while current_dir.parent != current_dir:
                test_path = current_dir / '.env'
                if test_path.exists():
                    env_file = test_path
                    break
                current_dir = current_dir.parent
            
            if env_file:
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {env_file}")
            else:
                logger.warning("No .env file found, using system environment variables")
                
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")
            raise ConfigError(f"Failed to load environment variables: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate required configuration variables"""
        try:
            required_vars = [
                'FMP_API_KEY'  # WORLDBANK_API_KEY is optional
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
                logger.error(error_msg)
                raise ConfigError(error_msg)
                
        except Exception as e:
            if not isinstance(e, ConfigError):
                logger.error(f"Error validating configuration: {str(e)}")
                raise ConfigError(f"Configuration validation failed: {str(e)}")
            raise
    
    def _setup_directories(self) -> None:
        """Set up required directories"""
        try:
            # Get root directory (where .env file is located or current directory)
            root_dir = Path.cwd()
            env_file = root_dir / '.env'
            while not env_file.exists() and root_dir.parent != root_dir:
                root_dir = root_dir.parent
                env_file = root_dir / '.env'
            
            # Set up directory structure
            self.root_dir = root_dir
            self.cache_dir = root_dir / 'cache'
            self.data_dir = root_dir / 'data'
            self.log_dir = root_dir / 'logs'
            
            # Create directories if they don't exist
            for directory in [self.cache_dir, self.data_dir, self.log_dir]:
                directory.mkdir(exist_ok=True, parents=True)
                
        except Exception as e:
            error_msg = f"Failed to setup directories: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    @classmethod
    def load_env(cls) -> None:
        """Load environment variables (can be called explicitly)"""
        cls._load_env()
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with optional default"""
        try:
            return os.getenv(key, default)
        except Exception as e:
            logger.error(f"Error getting environment variable {key}: {str(e)}")
            return default
    
    def fmp_key(self) -> str:
        """Get FMP API key"""
        key = self.api_keys.get('fmp')
        if not key:
            raise ConfigError("FMP API key not found. Set FMP_API_KEY environment variable.")
        return key
    
    def world_bank_key(self) -> Optional[str]:
        """Get World Bank API key"""
        return self.api_keys.get('world_bank')
    
    def yf_key(self) -> Optional[str]:
        """Get Yahoo Finance API key"""
        return self.api_keys.get('yahoo_finance')
    
    def get_risk_weight(self, risk_type: str) -> float:
        """Get risk weight"""
        return self.risk_weights.get(risk_type, 0.0)
    
    def get_rate_limit(self, api: str) -> float:
        """Get API rate limit"""
        return 1.0 / self.rate_limits.get(api, 1)  # Convert to seconds between requests
    
    def get_cache_path(self, name: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{name}.json"
    
    def get_risk_level(self, score: float) -> str:
        """Get risk level based on score"""
        if score <= self.risk_thresholds['low_risk']:
            return 'Low'
        elif score <= self.risk_thresholds['medium_risk']:
            return 'Medium'
        else:
            return 'High'
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            return getattr(self, key, default)
        except Exception as e:
            logger.error(f"Error getting configuration value {key}: {str(e)}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        try:
            setattr(self, key, value)
        except Exception as e:
            logger.error(f"Error setting configuration value {key}: {str(e)}")
            raise ConfigError(f"Failed to set configuration value: {str(e)}")
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        try:
            value = self.get(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        except Exception:
            return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        try:
            value = self.get(key, default)
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value"""
        try:
            value = self.get(key, default)
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """Get list configuration value"""
        try:
            value = self.get(key, default or [])
            if isinstance(value, str):
                return json.loads(value)
            return list(value)
        except Exception:
            return default or []
    
    def get_dict(self, key: str, default: Optional[dict] = None) -> Dict:
        """Get dictionary configuration value"""
        try:
            value = self.get(key, default or {})
            if isinstance(value, str):
                return json.loads(value)
            return dict(value)
        except Exception:
            return default or {}
    
    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds"""
        return self.cache_ttl