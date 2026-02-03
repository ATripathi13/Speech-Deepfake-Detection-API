"""
Authentication Middleware for Speech Deepfake Detection API

Implements API key validation for secure access control.
"""

import os
from typing import Optional, List
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# API key header scheme
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def get_valid_api_keys() -> List[str]:
    """
    Get list of valid API keys from environment variable.
    
    Returns:
        List of valid API keys
    """
    # Try to get from environment variable (comma-separated)
    env_keys = os.getenv("VALID_API_KEYS", "")
    
    if env_keys:
        keys = [key.strip() for key in env_keys.split(",") if key.strip()]
        return keys
    
    # Default development key if no environment variable set
    return ["demo-key-12345"]


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify the API key from request header.
    
    Args:
        api_key: API key from x-api-key header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    valid_keys = get_valid_api_keys()
    
    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please provide 'x-api-key' in request headers."
        )
    
    # Validate API key
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Access denied."
        )
    
    return api_key


if __name__ == "__main__":
    # Test authentication
    print("Testing authentication module...")
    
    # Set test environment
    os.environ["VALID_API_KEYS"] = "test-key-1,test-key-2,test-key-3"
    
    valid_keys = get_valid_api_keys()
    print(f"✓ Valid API keys loaded: {len(valid_keys)} keys")
    print(f"  Keys: {', '.join(valid_keys)}")
    
    # Test validation logic
    test_valid = "test-key-1"
    test_invalid = "invalid-key"
    
    print(f"\n  Testing valid key '{test_valid}': {'✓ PASS' if test_valid in valid_keys else '✗ FAIL'}")
    print(f"  Testing invalid key '{test_invalid}': {'✓ PASS' if test_invalid not in valid_keys else '✗ FAIL'}")
