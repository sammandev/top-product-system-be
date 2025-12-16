"""
Tests for external DUT API router endpoints.
"""

import os
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.external_services.dut_api_client import DUTAPIClient
from app.main import app

client = TestClient(app)

RUN_DUT_TESTS = os.getenv("RUN_DUT_API_TESTS", "0") == "1"
pytestmark = pytest.mark.skipif(not RUN_DUT_TESTS, reason="Requires intranet DUT API connectivity")


def test_get_sites():
    """Test getting all sites."""
    with patch("app.dependencies.get_dut_client") as mock_get_client:
        mock_client = Mock(spec=DUTAPIClient)
        mock_client.get_sites.return_value = [{"id": 1, "name": "ABS", "iplas_url": None}]
        mock_get_client.return_value = mock_client
        response = client.get("/api/dut/sites")
        assert response.status_code == 200
        assert len(response.json()) > 0


def test_get_dut_records():
    """Test getting DUT records."""
    dut_id = "DM2520270073965"
    with patch("app.dependencies.get_dut_client") as mock_get_client:
        mock_client = Mock(spec=DUTAPIClient)
        mock_client.get_dut_records.return_value = {
            "site_name": "ABS",
            "model_name": "ABST",
            "record_data": [],
        }
        mock_get_client.return_value = mock_client
        response = client.get(f"/api/dut/records/{dut_id}")
        assert response.status_code == 200
        assert response.json()["site_name"] == "ABS"
