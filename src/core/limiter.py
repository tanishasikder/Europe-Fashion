from fastapi import FastAPI, status 
from slowapi import Limiter, rate_limit_exceeded_handler
from slowapi.util import get_remote_address # Returns ip for current address
from slowapi.errors import RateLimitExceeded

def get_limit():
    limiter = Limiter(key_func=get_remote_address)
    return limiter
