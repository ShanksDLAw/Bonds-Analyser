"""Custom exceptions for Bond Risk Analyzer"""

class DataNotFoundError(Exception):
    """Raised when requested data is not found"""
    pass

class RateLimitExceededError(Exception):
    """Raised when API rate limit is exceeded"""
    pass

class TimeoutError(Exception):
    """Raised when a request times out"""
    pass

class ValidationError(Exception):
    """Raised when data validation fails"""
    pass