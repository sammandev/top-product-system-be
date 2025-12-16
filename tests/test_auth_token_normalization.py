from app.utils import auth as auth_utils


def test_token_normalization_roundtrip():
    # Create a dummy user-like object with the attributes used by the token helpers
    class DummyUser:
        def __init__(self):
            self.username = "norm_user"
            self.is_admin = False
            self.token_version = 1

    user = DummyUser()

    token = auth_utils.create_access_token(user)
    assert token

    # Raw token should decode
    payload = auth_utils.decode_jwt(token)
    assert payload is not None
    assert payload.get("sub") == "norm_user"

    # Token with Bearer prefix should NOT decode now (we enforce strict tokens)
    bearer = f"Bearer {token}"
    payload2 = auth_utils.decode_jwt(bearer)
    assert payload2 is None

    # Token wrapped with quotes should NOT decode now
    quoted = f'"{token}"'
    payload3 = auth_utils.decode_jwt(quoted)
    assert payload3 is None
