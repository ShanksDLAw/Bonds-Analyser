# src/tests/test_cache.py
import pytest
from datetime import datetime
import datetime as dt
from src.data.cache import CacheManager, RateLimitExceededError
from src.data.country_fetcher import CountryDataManager, IMFClient, WorldBankClient, TreasuryClient

@pytest.fixture
def cache_manager():
    """Create a CacheManager instance for testing"""
    return CacheManager()

def test_cache_initialization(cache_manager):
    """Test that cache is initialized correctly"""
    assert cache_manager.cache_dir.exists()
    assert cache_manager.cache_dir.is_dir()

def test_cache_operations(cache_manager):
    """Test basic cache operations"""
    key = "test_key"
    data = {"test": "data"}
    
    # Test setting cache
    cache_manager.set(key, data)
    
    # Test getting cache
    cached_data = cache_manager.get(key)
    assert cached_data == data
    
    # Test cache expiration
    cache_manager.ttl = 0  # Set TTL to 0 to force expiration
    expired_data = cache_manager.get(key)
    assert expired_data is None

def test_cache_invalidation(cache_manager):
    """Test cache invalidation"""
    key = "test_key"
    data = {"test": "data"}
    
    # Set cache
    cache_manager.set(key, data)
    
    # Invalidate cache
    cache_manager.invalidate(key)
    
    # Verify cache is invalidated
    assert cache_manager.get(key) is None

def test_cache_cleanup(cache_manager):
    """Test cache cleanup"""
    # Add some test data
    for i in range(5):
        key = f"test_key_{i}"
        data = {"test": f"data_{i}"}
        cache_manager.set(key, data)
    
    # Run cleanup
    cache_manager.cleanup()
    
    # Verify old cache entries are removed
    for i in range(5):
        key = f"test_key_{i}"
        assert cache_manager.get(key) is None