# src/tests/test_config.py
from src.config import Config  # Changed to absolute import

def test_cache_dir_creation():
    cache_path = Config.cache_dir()
    assert cache_path.exists()
    assert cache_path.is_dir()
    
def test_api_keys_present():
    assert Config.fmp_key() is not None, "FMP_API_KEY missing in .env"
    assert Config.fred_key() is not None, "FRED_API_KEY missing in .env"