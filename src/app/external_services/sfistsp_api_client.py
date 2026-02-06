"""
SFISTSP (SOAP/HTTP) API Client for ISN Reference Lookup.

This module provides a client for interacting with the SFISTSP Web Service
to look up ISN references (SSN, MAC address, etc.) using the WTSP_GETVERSION endpoint.

The SFISTSP API can be accessed via:
- SOAP 1.2 (XML envelope)
- HTTP GET (query parameters)
- HTTP POST (form-urlencoded)

This implementation uses HTTP GET for simplicity.

Reference: external-api/sfistsp-http-soap.md
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration from Environment
# ============================================================================

# Default SFISTSP server configuration (loaded from environment)
SFISTSP_DEFAULT_BASE_URL = os.environ.get("SFISTSP_API_BASE_URL", "http://10.176.33.13")
SFISTSP_DEFAULT_SERVER_ADDRESS = os.environ.get(
    "SFISTSP_API_SERVER_ADDRESS",
    "http://10.176.33.13/SFISWebService/SFISTSPWebService.asmx",
)
SFISTSP_DEFAULT_PROGRAM_ID = os.environ.get("SFIS_DEFAULT_PROGRAM_ID", "TSP_TDSTB")
SFISTSP_DEFAULT_PROGRAM_PASSWORD = os.environ.get("SFIS_DEFAULT_PROGRAM_PASSWORD", "ap_tbsus")

# Endpoint path for WTSP_GETVERSION
SFISTSP_ENDPOINT_PATH = "/SFISWebService/SFISTSPWebService.asmx/WTSP_GETVERSION"

# Request timeout in seconds (from environment or default 120s)
SFISTSP_REQUEST_TIMEOUT = float(os.environ.get("SFISTSP_API_TIMEOUT", "120"))


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SfistspIsnReference:
    """Parsed ISN reference from SFISTSP API response."""

    isn: str
    ssn: str | None = None
    mac: str | None = None
    raw_response: str = ""
    success: bool = True
    error_message: str | None = None
    # Additional parsed fields
    isn_references: list[str] = field(default_factory=list)


@dataclass
class SfistspConfig:
    """Configuration for SFISTSP API client."""

    base_url: str = SFISTSP_DEFAULT_BASE_URL
    program_id: str = SFISTSP_DEFAULT_PROGRAM_ID
    program_password: str = SFISTSP_DEFAULT_PROGRAM_PASSWORD
    timeout: float = SFISTSP_REQUEST_TIMEOUT


# ============================================================================
# Response Parser
# ============================================================================


def _clean_reference(ref: str) -> str:
    """
    Clean a reference string by removing whitespace, MAC: prefixes, and brackets.

    Args:
        ref: Raw reference string

    Returns:
        Cleaned reference string
    """
    if not ref:
        return ""

    # Remove all whitespace (including internal spaces, newlines, tabs)
    # Also remove any non-printable characters
    cleaned = "".join(c for c in ref if c.isprintable() and not c.isspace())

    # Remove MAC:XXXX patterns (MAC followed by hex chars and optional brackets)
    # Pattern: MAC:12hexchars[...] or MAC:12hexchars
    cleaned = re.sub(r"MAC:[A-Fa-f0-9]*(?:\[[^\]]*\])?", "", cleaned)

    # Remove trailing "MAC:" if present (malformed data without hex)
    while cleaned.endswith("MAC:"):
        cleaned = cleaned[:-4]

    # Remove leading "MAC:" if present
    while cleaned.startswith("MAC:"):
        cleaned = cleaned[4:]

    # Remove any trailing brackets and their content
    bracket_idx = cleaned.find("[")
    if bracket_idx > 0:
        cleaned = cleaned[:bracket_idx]

    # Remove any trailing semicolons
    cleaned = cleaned.rstrip(";")

    return cleaned.strip()


def _deduplicate_string(s: str) -> str:
    """
    Detect and remove duplicated substrings in a string.

    For example: "7UG162WC004707UG162WC00470" -> "7UG162WC00470"

    This handles cases where the SSN is repeated twice in the response.

    Args:
        s: Input string that may contain duplicates

    Returns:
        Deduplicated string
    """
    if not s or len(s) < 2:
        return s

    # Clean the string first - remove any hidden characters
    s = "".join(c for c in s if c.isprintable() and not c.isspace())

    length = len(s)
    if length < 2:
        return s

    # Check if the string is exactly doubled (most common case)
    # This is the primary case we're trying to handle
    if length % 2 == 0:
        half = length // 2
        first_half = s[:half]
        second_half = s[half:]
        if first_half == second_half:
            logger.debug(f"Deduplicated doubled string: '{s}' -> '{first_half}'")
            return first_half

    # Try to find if the string is composed of a repeated pattern
    for pattern_len in range(1, length // 2 + 1):
        if length % pattern_len == 0:
            pattern = s[:pattern_len]
            repetitions = length // pattern_len
            if pattern * repetitions == s:
                logger.debug(f"Deduplicated pattern string: '{s}' -> '{pattern}'")
                return pattern

    return s


def _extract_mac_addresses(data: str) -> list[str]:
    """
    Extract all MAC addresses from the data string.

    MAC addresses are 12 hex characters following "MAC:" prefix.

    Args:
        data: Raw data string

    Returns:
        List of unique MAC addresses (uppercase, 12 hex chars)
    """
    mac_addresses: list[str] = []

    # Find all MAC: patterns followed by hex characters
    # Pattern: MAC:XXXXXXXXXXXX where X is hex (exactly 12 chars)
    pattern = re.compile(r"MAC:([A-Fa-f0-9]{12})")

    for match in pattern.finditer(data):
        mac = match.group(1).upper().strip()
        if mac and len(mac) == 12 and mac not in mac_addresses:
            mac_addresses.append(mac)

    return mac_addresses


def _remove_mac_patterns(data: str) -> str:
    """
    Remove all MAC-related patterns from the data string.

    This handles:
    - MAC:XXXXXXXXXXXX[...] - full MAC with bracket content
    - MAC:XXXXXXXXXXXX - full MAC without brackets
    - MAC: - malformed MAC prefix at end

    Args:
        data: Raw data string

    Returns:
        Data with MAC patterns removed
    """
    # Remove MAC:12chars[...] patterns
    cleaned = re.sub(r"MAC:[A-Fa-f0-9]{12}(?:\[[^\]]*\])?", "", data)

    # Remove MAC: followed by any hex chars and optional brackets
    cleaned = re.sub(r"MAC:[A-Fa-f0-9]*(?:\[[^\]]*\])?", "", cleaned)

    # Remove standalone MAC: at end
    cleaned = re.sub(r"MAC:\s*$", "", cleaned)
    cleaned = re.sub(r"MAC:(?=[;,\s]|$)", "", cleaned)

    return cleaned


def _process_reference(ref: str) -> str | None:
    """
    Process a single reference: clean it, deduplicate it, and validate it.

    Args:
        ref: Raw reference string

    Returns:
        Processed reference or None if invalid/empty
    """
    # Clean the reference
    cleaned = _clean_reference(ref)
    if not cleaned:
        return None

    # Deduplicate
    deduped = _deduplicate_string(cleaned)
    if not deduped:
        return None

    # Validate - should only contain alphanumeric characters
    # and should be at least 2 characters long
    if len(deduped) < 2:
        return None

    return deduped


def parse_sfistsp_response(response_text: str, isn: str) -> SfistspIsnReference:
    """
    Parse SFISTSP WTSP_GETVERSION response to extract ISN references.

    The response format is an XML string element containing a semi-structured text.
    Example response:
    <?xml version="1.0" encoding="utf-8"?>
    <string xmlns="http://www.pegatroncorp.com/SFISWebService/">
    1TYPE:[ISN_BASEINFO] 1:[SELECT SSN BY ISN:264118730000123 DATA GOT!!]
    2:[] 3:[CHK ISN_SN_MAC [ISN||';'||SSN] DATA GOT!!](TSP_GV_ISN_BASEINFO)(TSP_GETVERSION)
    264118730000123;7UG162WC00470MAC:E0C2504EF9C3[NETGM7L]
    </string>

    Parsing rules:
    - The main data follows the closing parenthesis of (TSP_GETVERSION)
    - ISN references are separated by semicolons
    - MAC address follows "MAC:" prefix and ends at "[" or ";" or end of string
    - First reference before ";" is usually the ISN itself
    - Second reference (after ";") is usually SSN
    - All references should be cleaned of whitespace and deduplicated

    Args:
        response_text: Raw XML response from SFISTSP API
        isn: The ISN that was searched

    Returns:
        SfistspIsnReference with parsed data
    """
    result = SfistspIsnReference(isn=isn, raw_response=response_text)

    try:
        # Extract content from XML <string> element
        # Pattern: <string xmlns="...">CONTENT</string>
        string_match = re.search(
            r"<string[^>]*>(.+?)</string>",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )

        if not string_match:
            # Maybe it's just the content without XML wrapper
            content = response_text.strip()
        else:
            content = string_match.group(1).strip()

        # Check for error indicators
        if "ERROR" in content.upper() or "FAIL" in content.upper():
            error_match = re.search(r"(ERROR[^\]]*|FAIL[^\]]*)", content, re.IGNORECASE)
            if error_match:
                result.success = False
                result.error_message = error_match.group(1).strip()
                return result

        # Check for "NO DATA" indicator
        if "NO DATA" in content.upper() or "NOT FOUND" in content.upper():
            result.success = False
            result.error_message = "ISN not found in SFISTSP"
            return result

        # Extract the main data portion after (TSP_GETVERSION)
        # This contains the actual ISN references
        data_match = re.search(r"\(TSP_GETVERSION\)(.+?)(?:<|$)", content, re.DOTALL)
        if data_match:
            data_part = data_match.group(1).strip()
        else:
            # Try alternative: look for ISN followed by semicolon pattern
            data_part = content

        # First, extract all MAC addresses from the data using helper function
        mac_addresses = _extract_mac_addresses(data_part)

        # Set the first MAC address as the primary MAC
        if mac_addresses:
            result.mac = mac_addresses[0]

        # Remove all MAC patterns from data to get clean references
        clean_data = _remove_mac_patterns(data_part)

        # Parse ISN references (separated by semicolons)
        references: list[str] = []

        if ";" in clean_data:
            parts = clean_data.split(";")
            for part in parts:
                # Use _process_reference to clean, deduplicate, and validate
                processed = _process_reference(part)
                if processed and processed not in references:
                    references.append(processed)
        else:
            # No semicolons, try to extract the single reference
            processed = _process_reference(clean_data)
            if processed:
                references.append(processed)

        # Add MAC addresses to references if not already present
        for mac in mac_addresses:
            if mac not in references:
                references.append(mac)

        # Final cleanup: remove any empty strings that might have slipped through
        references = [r for r in references if r and r.strip()]
        result.isn_references = references

        # Determine SSN: usually the second reference (after ISN)
        # or if only one non-ISN reference exists, use that
        ssn_candidate = None
        if len(references) >= 2:
            # Second reference is usually SSN
            ssn_candidate = references[1]
            # If second reference is a MAC address, look for another
            if ssn_candidate in mac_addresses and len(references) > 2:
                for ref in references[2:]:
                    if ref not in mac_addresses:
                        ssn_candidate = ref
                        break
        elif len(references) == 1 and references[0] != isn:
            ssn_candidate = references[0]

        # Apply deduplication and cleaning to SSN
        if ssn_candidate:
            ssn_candidate = _process_reference(ssn_candidate)
        result.ssn = ssn_candidate or ""

        result.success = True

        logger.debug(f"Parsed SFISTSP response for ISN {isn}: ssn={result.ssn!r}, mac={result.mac!r}, references={result.isn_references!r}")

    except Exception as e:
        logger.error(f"Error parsing SFISTSP response for ISN {isn}: {e}")
        result.success = False
        result.error_message = str(e)

    return result


# ============================================================================
# API Client
# ============================================================================


class SfistspApiClient:
    """
    Client for SFISTSP Web Service API.

    Uses HTTP GET requests to the WTSP_GETVERSION endpoint
    to look up ISN references.
    """

    def __init__(self, config: SfistspConfig | None = None):
        """
        Initialize SFISTSP API client.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or SfistspConfig()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SfistspApiClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_url(self, isn: str) -> str:
        """
        Build the WTSP_GETVERSION URL with query parameters.

        Based on sfistsp-http-soap.md documentation:
        - programId: Required - authentication program ID
        - programPassword: Required - authentication password
        - ISN: Required - the ISN to look up
        - device: Required - can be random 4 digits (e.g., "1234")
        - type: "ISN_BASEINFO" - to get ISN information
        - ChkData: "ISN,SSN" - to get ISN and SSN references
        - ChkData2: "SNMAC" - to get MAC address reference

        Args:
            isn: The ISN to look up

        Returns:
            Full URL with query parameters
        """
        params = {
            "programId": self.config.program_id,
            "programPassword": self.config.program_password,
            "ISN": isn,
            "device": "1234",  # Required but can be random 4 digits
            "type": "ISN_BASEINFO",  # Get ISN base info
            "ChkData": "ISN,SSN",  # Get ISN and SSN references
            "ChkData2": "SNMAC",  # Get MAC address
        }
        base_url = self.config.base_url.rstrip("/")
        query_string = urlencode(params)
        return f"{base_url}{SFISTSP_ENDPOINT_PATH}?{query_string}"

    async def lookup_isn(self, isn: str) -> SfistspIsnReference:
        """
        Look up ISN references from SFISTSP.

        Args:
            isn: The ISN to look up

        Returns:
            SfistspIsnReference with parsed data
        """
        url = self._build_url(isn)
        logger.info(f"SFISTSP API: Looking up ISN {isn}")

        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=self.config.timeout)

            response = await self._client.get(url)
            response.raise_for_status()

            result = parse_sfistsp_response(response.text, isn)
            logger.info(f"SFISTSP API: ISN {isn} lookup {'successful' if result.success else 'failed'} - SSN: {result.ssn}, MAC: {result.mac}")
            return result

        except httpx.TimeoutException:
            logger.error(f"SFISTSP API: Timeout looking up ISN {isn}")
            return SfistspIsnReference(
                isn=isn,
                success=False,
                error_message=f"Request timeout after {self.config.timeout}s",
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"SFISTSP API: HTTP error {e.response.status_code} for ISN {isn}")
            return SfistspIsnReference(
                isn=isn,
                success=False,
                error_message=f"HTTP error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"SFISTSP API: Error looking up ISN {isn}: {e}")
            return SfistspIsnReference(
                isn=isn,
                success=False,
                error_message=str(e),
            )

    async def lookup_isns_batch(
        self,
        isns: list[str],
        max_concurrent: int = 10,
    ) -> list[SfistspIsnReference]:
        """
        Look up multiple ISNs in batch using parallel requests.

        Uses asyncio.gather with a semaphore to limit concurrent connections.
        This significantly improves performance for bulk ISN lookups.

        Args:
            isns: List of ISNs to look up
            max_concurrent: Maximum number of concurrent requests (default: 10)

        Returns:
            List of SfistspIsnReference results in the same order as input
        """
        if not isns:
            return []

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def lookup_with_semaphore(isn: str) -> SfistspIsnReference:
            async with semaphore:
                return await self.lookup_isn(isn)

        # Execute all lookups in parallel
        tasks = [lookup_with_semaphore(isn) for isn in isns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        final_results: list[SfistspIsnReference] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(SfistspIsnReference(
                    isn=isns[i],
                    success=False,
                    error_message=str(result),
                ))
            else:
                final_results.append(result)

        logger.info(f"SFISTSP batch lookup: {len(isns)} ISNs, {sum(1 for r in final_results if r.success)} successful")
        return final_results


# ============================================================================
# Factory Function
# ============================================================================


def create_sfistsp_client(
    base_url: str | None = None,
    program_id: str | None = None,
    program_password: str | None = None,
    timeout: float | None = None,
) -> SfistspApiClient:
    """
    Create an SFISTSP API client with optional configuration overrides.

    Args:
        base_url: Optional SFISTSP server base URL
        program_id: Optional program ID for authentication
        program_password: Optional program password
        timeout: Optional request timeout in seconds

    Returns:
        Configured SfistspApiClient instance
    """
    config = SfistspConfig(
        base_url=base_url or SFISTSP_DEFAULT_BASE_URL,
        program_id=program_id or SFISTSP_DEFAULT_PROGRAM_ID,
        program_password=program_password or SFISTSP_DEFAULT_PROGRAM_PASSWORD,
        timeout=timeout or SFISTSP_REQUEST_TIMEOUT,
    )
    return SfistspApiClient(config)
