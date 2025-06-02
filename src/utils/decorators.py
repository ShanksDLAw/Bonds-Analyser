import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (RequestException, Exception)
) -> Callable:
    """
    Retry decorator with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        return None
                    
                    logger.warning(
                        f"Error on attempt {retries}/{max_retries}: {str(e)}. "
                        f"Retrying in {current_delay:.1f} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator 