"""
Encryption utilities for sensitive configuration values.

Uses AES-256-GCM authenticated encryption via pycryptodomex.
The encryption key is derived from the ENCRYPTION_KEY environment variable.
"""

import base64
import hashlib
import logging
import os

from Cryptodome.Cipher import AES

logger = logging.getLogger(__name__)

# Derive a 32-byte key from the env var using SHA-256
_RAW_KEY = os.environ.get("ENCRYPTION_KEY", "ast-tools-default-encryption-key-change-me")
_KEY = hashlib.sha256(_RAW_KEY.encode()).digest()


def encrypt_value(plaintext: str) -> str:
    """
    Encrypt a plaintext string using AES-256-GCM.

    Returns:
        Base64-encoded string of nonce + ciphertext + tag
    """
    if not plaintext:
        return ""
    cipher = AES.new(_KEY, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
    # Pack nonce (16) + ciphertext + tag (16)
    packed = cipher.nonce + ciphertext + tag
    return base64.urlsafe_b64encode(packed).decode("ascii")


def decrypt_value(encoded: str) -> str:
    """
    Decrypt an AES-256-GCM encrypted value.

    Args:
        encoded: Base64-encoded string produced by encrypt_value

    Returns:
        Decrypted plaintext string
    """
    if not encoded:
        return ""
    try:
        raw = base64.urlsafe_b64decode(encoded)
        nonce = raw[:16]
        tag = raw[-16:]
        ciphertext = raw[16:-16]
        cipher = AES.new(_KEY, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext.decode("utf-8")
    except Exception:
        logger.warning("Failed to decrypt value — returning empty string")
        return ""


def mask_value(value: str, visible_chars: int = 3, max_dots: int = 3) -> str:
    """
    Mask a sensitive string, showing only the last few characters.

    Args:
        value: The string to mask
        visible_chars: Number of trailing characters to keep visible
        max_dots: Maximum number of masking dots to display (keeps output compact)

    Returns:
        Masked string like '•••abc'
    """
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "•" * len(value)
    return "•" * max_dots + value[-visible_chars:]
