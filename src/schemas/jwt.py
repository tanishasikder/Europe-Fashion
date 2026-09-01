import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError
from dotenv import load_dotenv
import os

load_dotenv()

jwt_url = os.environ.get('JWT_URL')

JWKS_URL = jwt_url
EXPECTED_AUDIENCE = "authenticated"
EXPECTED_ISSUER = "https://wytzqccveecanenvedvb.supabase.co/auth/v1"

jwks_client = PyJWKClient(JWKS_URL)

def verify_jwt(token: str):
    try:
        key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            key.key,
            algorithms=['RS256'],
            audience=EXPECTED_AUDIENCE,
            issuer=EXPECTED_ISSUER
        )

        return payload

    except InvalidTokenError as e:
        print(f"Token validation failed: {e}")
        return None

