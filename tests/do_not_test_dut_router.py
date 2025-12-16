"""
Tests for DUT router endpoints. These require access to the intranet DUT API or
an authenticated session and are therefore skipped by default.
"""

import os

import pytest

RUN_DUT_TESTS = os.getenv("RUN_DUT_API_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_DUT_TESTS, reason="Requires intranet DUT API connectivity")
