from app.utils import auth


def test_decode_jwt_token_rejects_non_string_subject(monkeypatch):
    monkeypatch.setattr(auth, "decode_jwt", lambda _token: {"type": "access", "sub": 42})

    assert auth.decode_jwt_token("token") == (None, False)
