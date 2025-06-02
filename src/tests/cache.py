"""Cache management module for Bond Risk Analyzer"""

import gzip
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union

from src.config import Config, ConfigError

# Configure logging
logger = logging.getLogger(__name__)

class CacheError(Exception):
    """Custom exception for cache operations"""
    pass

class RateLimitExceededError(Exception):
    """Custom exception for API rate limits"""
    pass

class CacheManager:
    """Cache manager for API responses and computed data"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache manager with configuration
        
        Args:
            cache_dir: Optional cache directory path
            
        Raises:
            CacheError: If initialization fails
        """
        try:
            self.config = Config()
            self.cache_dir = Path(cache_dir) if cache_dir else self.config.cache_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Default cache settings
            self.max_age = timedelta(days=1)  # Default cache expiry
            self.compression = True  # Use gzip compression by default
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {str(e)}")
            raise CacheError(f"Initialization error: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
            
        Raises:
            CacheError: If retrieval fails
        """
        try:
            cache_file = self.cache_dir / f"{key}.json.gz"
            
            if not cache_file.exists():
                return None
            
            # Check if cache is expired
            if self._is_expired(cache_file):
                logger.debug(f"Cache expired for {key}")
                cache_file.unlink()
                return None
            
            # Read and decompress cache
            with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get('value')
            
        except Exception as e:
            logger.warning(f"Error reading cache for {key}: {str(e)}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ) -> None:
        """Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live
            
        Raises:
            CacheError: If setting fails
        """
        try:
            cache_file = self.cache_dir / f"{key}.json.gz"
            
            # Prepare cache data
            cache_data = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'ttl': ttl.total_seconds() if ttl else self.max_age.total_seconds()
            }
            
            # Write and compress cache
            with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {str(e)}")
            raise CacheError(f"Failed to set cache: {str(e)}")
    
    def delete(self, key: str) -> None:
        """Delete value from cache
        
        Args:
            key: Cache key
            
        Raises:
            CacheError: If deletion fails
        """
        try:
            cache_file = self.cache_dir / f"{key}.json.gz"
            if cache_file.exists():
                cache_file.unlink()
                
        except Exception as e:
            logger.error(f"Error deleting cache for {key}: {str(e)}")
            raise CacheError(f"Failed to delete cache: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cached data
        
        Raises:
            CacheError: If clearing fails
        """
        try:
            for cache_file in self.cache_dir.glob("*.json.gz"):
                cache_file.unlink()
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise CacheError(f"Failed to clear cache: {str(e)}")
    
    def _is_expired(self, cache_file: Path) -> bool:
        """Check if cache file is expired
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            True if expired, False otherwise
        """
        try:
            # Read cache metadata
            with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get cache timestamp and TTL
            timestamp = datetime.fromisoformat(data['timestamp'])
            ttl = timedelta(seconds=data['ttl'])
            
            # Check if expired
            return datetime.now() - timestamp > ttl
            
        except Exception as e:
            logger.warning(f"Error checking cache expiry: {str(e)}")
            return True  # Treat as expired on error

# Add the project root to the Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)

# Test function to demonstrate the cache is working
if __name__ == "__main__":
    print("CacheManager Test")
    print("-" * 50)
    
    # Create cache manager
    cache = CacheManager()
    
    # Test cache directory
    print(f"Cache directory: {cache.cache_dir}")
    print(f"Cache directory exists: {cache.cache_dir.exists()}")
    
    # Test caching
    test_key = "test_data_123"
    test_data = {"timestamp": datetime.utcnow().isoformat(), "value": 42}
    
    print(f"\nSetting cache for key: {test_key}")
    cache.set(test_key, test_data)
    
    print(f"Cache file path: {cache.cache_dir / f'{test_key}.json.gz'}")
    print(f"Cache file exists: {(cache.cache_dir / f'{test_key}.json.gz').exists()}")
    
    print(f"\nRetrieving from cache for key: {test_key}")
    retrieved_data = cache.get(test_key)
    print(f"Retrieved data: {retrieved_data}")
    
    print("\nCacheManager test completed successfully!")