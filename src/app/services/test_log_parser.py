"""
Service for parsing test log files (.txt) with SFIS Test Result format.

Supports parsing files with the format:
=========[Start SFIS Test Result]======
"TEST_ITEM" <"USL","LSL">  ===> "VALUE"
...
=========[End SFIS Test Result]======

Rules:
- Only parse data between Start and End markers (flexible '=' count)
- Skip test items with values: "PASS", "VALUE"
- Include FAIL items to track failed tests
- Extract test_item, USL (Upper Spec Limit), LSL (Lower Spec Limit), and actual value
- Supports archive files: .zip, .rar, .7z

Enhanced features for BY UPLOAD LOG:
- Metadata extraction from log headers
- Custom criteria file parsing
- Value/non-value classification
- Scoring with LaTeX formulas
"""

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median


class TestLogParser:
    """Parser for SFIS test log files."""

    # Marker patterns - flexible to match any number of '=' characters
    # Only requires the text: [Start SFIS Test Result] and [End SFIS Test Result]
    START_MARKER_PATTERN = re.compile(r"=+\[Start SFIS Test Result\]=+")
    END_MARKER_PATTERN = re.compile(r"=+\[End SFIS Test Result\]=+")

    # Filename patterns - multiple formats supported:
    # Pattern 1: [TestStation]_[ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
    #   Example: Wireless_Test_6G_DM2524470075517_2025_11_20_105056100
    # Pattern 2: [ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
    #   Example: DM2527470012971_2025_11_20_172659650
    # Pattern 3: [ISN]_[TestStation]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
    #   Example: DM2524470075517_Wireless_Test_6G_2025_11_20_105056100
    # Note: ISN never contains underscore character

    # Pattern for ISN followed by date (no station or station after ISN)
    FILENAME_PATTERN_ISN_FIRST = re.compile(r"^(?P<isn>[A-Z0-9]+)_(?:(?P<station>[^_]+(?:_[^_]+)*)_)?(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_(?P<time>\d+)")

    # Pattern for station followed by ISN (original format)
    FILENAME_PATTERN_STATION_FIRST = re.compile(r"^(?P<station>[^_]+(?:_[^_]+)*)_(?P<isn>[A-Z0-9]+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_(?P<time>\d+)")

    # Values to exclude from parsing (non-numeric status values)
    # Note: FAIL is now included to track failed test items
    EXCLUDED_VALUES = {"PASS", "VALUE"}

    # Regex pattern for parsing test item lines
    # Pattern: "TEST_ITEM" <"USL","LSL">  ===> "VALUE"
    # Also handles: "TEST_ITEM" <USL,LSL>  ===> "VALUE" (without quotes around limits)
    # And: "TEST_ITEM" <,>  ===> "VALUE" (empty limits)
    LINE_PATTERN = re.compile(r'"([^"]+)"\s*<\s*"?([^",>]*)"?\s*,\s*"?([^",>]*)"?\s*>\s*===>\s*"([^"]+)"')

    @staticmethod
    def extract_isn_from_filename(filename: str) -> str | None:
        """
        Extract ISN (Independent Serial Number) from filename.

        Supported patterns:
        1. [TestStation]_[ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
           Example: Wireless_Test_6G_DM2524470075517_2025_11_20_105056100
        2. [ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
           Example: DM2527470012971_2025_11_20_172659650
        3. [ISN]_[TestStation]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
           Example: DM2524470075517_Wireless_Test_6G_2025_11_20_105056100

        Note: ISN never contains underscore character

        Args:
            filename: Name of the file (with or without extension)

        Returns:
            ISN string if found, None otherwise
        """
        # Remove extension if present
        name_without_ext = Path(filename).stem

        # Split by underscore
        parts = name_without_ext.split("_")

        if len(parts) < 5:
            return None  # Not enough parts for any valid pattern

        # Look for the date pattern: YYYY_MM_DD_HHmmss (last 4 parts)
        # Find the index where the date starts (should be a 4-digit year)
        date_idx = None
        for i in range(len(parts) - 3):
            if len(parts[i]) == 4 and parts[i].isdigit() and len(parts[i + 1]) == 2 and parts[i + 1].isdigit() and len(parts[i + 2]) == 2 and parts[i + 2].isdigit():
                date_idx = i
                break

        if date_idx is None:
            return None  # No valid date pattern found

        # ISN is the part immediately before the date (or before the station if pattern 3)
        # Parts before date_idx could be: [ISN] or [Station, ISN] or [ISN, Station]

        if date_idx == 1:
            # Pattern 2: [ISN]_[YYYY]_[MM]_[DD]_[HHmmss]
            # ISN is the first part
            return parts[0]

        elif date_idx == 2:
            # Could be Pattern 1 or Pattern 3
            # Pattern 1: [Station]_[ISN]_[Date]
            # Pattern 3: [ISN]_[Station]_[Date]
            # The ISN is the one without underscores that's alphanumeric
            # Check the first part - if it looks like an ISN (no spaces, alphanumeric), try it
            # ISNs are typically all caps/numbers without spaces
            candidate1 = parts[0]
            candidate2 = parts[1]

            # ISN characteristics: no underscore (already split), typically longer alphanumeric
            # Try to detect which is ISN vs station name
            # Heuristic: ISN is usually longer (10+ chars) or starts with specific prefixes

            # Common ISN prefixes
            isn_prefixes = ("ISN", "SN", "DM", "DN", "SER", "SERIAL")

            # Check if either starts with known ISN prefix
            c1_has_prefix = any(candidate1.upper().startswith(prefix) for prefix in isn_prefixes)
            c2_has_prefix = any(candidate2.upper().startswith(prefix) for prefix in isn_prefixes)

            if c1_has_prefix and not c2_has_prefix:
                return candidate1  # Pattern 3: ISN first
            elif c2_has_prefix and not c1_has_prefix:
                return candidate2  # Pattern 1: ISN second
            elif len(candidate1) >= 10 and candidate1.isalnum():
                # Likely Pattern 3: ISN first
                return candidate1
            elif len(candidate2) >= 10 and candidate2.isalnum():
                # Likely Pattern 1: Station first, ISN second
                return candidate2
            else:
                # Fall back: prefer the longer alphanumeric one
                if candidate2.isalnum():
                    return candidate2
                elif candidate1.isalnum():
                    return candidate1

        else:
            # Multiple parts before date (e.g., Wireless_Test_6G_ISN_Date or ISN_Wireless_Test_6G_Date)
            # Could be Pattern 1: [Station_Parts]_[ISN]_[Date] → ISN is last part before date
            # or Pattern 3: [ISN]_[Station_Parts]_[Date] → ISN is first part

            # Get candidates
            first_part = parts[0]
            last_before_date = parts[date_idx - 1]

            # Check characteristics
            first_len = len(first_part)
            last_len = len(last_before_date)
            first_alphanum = first_part.isalnum()
            last_alphanum = last_before_date.isalnum()

            # Strong preference for parts >= 10 characters that are alphanumeric
            if first_alphanum and first_len >= 10:
                return first_part  # Pattern 3: ISN first
            elif last_alphanum and last_len >= 10:
                return last_before_date  # Pattern 1: ISN last before date
            # If neither is >= 10 chars, prefer the longer alphanumeric one
            elif first_alphanum and last_alphanum:
                return first_part if first_len >= last_len else last_before_date
            elif first_alphanum:
                return first_part
            elif last_alphanum:
                return last_before_date

        return None

    @classmethod
    def parse_file(cls, file_path: str) -> dict[str, any]:
        """
        Parse a test log file and extract test items.

        Args:
            file_path: Path to the .txt test log file

        Returns:
            Dictionary containing:
                - filename: Name of the file
                - total_lines: Total lines in the file
                - parsed_items: List of parsed test items
                - skipped_items: Count of skipped items
                - errors: List of parsing errors
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".txt":
            raise ValueError(f"Invalid file type. Expected .txt, got {path.suffix}")

        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return cls.parse_content(content, path.name)

    @classmethod
    def parse_file_enhanced(
        cls,
        file_path: str,
        criteria_rules: dict[str, "TestLogCriteriaRule"] | None = None,
        show_only_criteria: bool = False,
    ) -> dict[str, any]:
        """
        Parse test log file with enhanced metadata, classification, and scoring.

        Args:
            file_path: Path to .txt test log file
            criteria_rules: Optional dict of criteria rules from .ini file
            show_only_criteria: If True, only return items matching criteria

        Returns:
            Dictionary matching TestLogParseResponseEnhanced schema
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".txt":
            raise ValueError(f"Invalid file type. Expected .txt, got {path.suffix}")

        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return cls.parse_content_enhanced(content, path.name, criteria_rules, show_only_criteria)

    @classmethod
    def is_archive(cls, file_path: str) -> bool:
        """
        Check if file is a supported archive format.

        Args:
            file_path: Path to the file

        Returns:
            True if file is .zip, .rar, or .7z
        """
        path = Path(file_path)
        return path.suffix.lower() in {".zip", ".rar", ".7z"}

    @classmethod
    def extract_archive(cls, archive_path: str, extract_to: str | None = None) -> list[str]:
        """
        Extract archive file and return paths to all .txt files inside.

        Args:
            archive_path: Path to the archive file
            extract_to: Directory to extract to (creates temp dir if None)

        Returns:
            List of paths to extracted .txt files

        Raises:
            ValueError: If archive format is not supported or extraction fails
        """
        path = Path(archive_path)
        suffix = path.suffix.lower()

        if extract_to is None:
            extract_to = tempfile.mkdtemp(prefix="test_log_extract_")

        extract_path = Path(extract_to)
        extract_path.mkdir(parents=True, exist_ok=True)

        txt_files = []

        try:
            if suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)
            elif suffix == ".rar":
                try:
                    import rarfile

                    with rarfile.RarFile(archive_path, "r") as rar_ref:
                        rar_ref.extractall(extract_path)
                except ImportError:
                    raise ValueError("rarfile package not installed. Install with: pip install rarfile")
            elif suffix == ".7z":
                try:
                    import py7zr

                    with py7zr.SevenZipFile(archive_path, mode="r") as z:
                        z.extractall(path=extract_path)
                except ImportError:
                    raise ValueError("py7zr package not installed. Install with: pip install py7zr")
            else:
                raise ValueError(f"Unsupported archive format: {suffix}")

            # Find all .txt files in extracted directory
            for txt_file in extract_path.rglob("*.txt"):
                txt_files.append(str(txt_file))

            return txt_files

        except Exception as e:
            # Clean up on error
            if extract_to is None and extract_path.exists():
                shutil.rmtree(extract_path, ignore_errors=True)
            raise ValueError(f"Failed to extract archive: {str(e)}")

    @classmethod
    def parse_archive(cls, archive_path: str) -> dict[str, any]:
        """
        Extract and parse all .txt files from an archive.

        Args:
            archive_path: Path to the archive file (.zip, .rar, .7z)

        Returns:
            Dictionary containing:
                - archive_name: Name of the archive
                - extracted_files: List of extracted .txt filenames
                - results: List of parse results for each file
                - total_files: Number of .txt files found
                - errors: List of extraction/parsing errors
        """
        archive_name = Path(archive_path).name
        temp_dir = None
        errors = []

        try:
            # Extract archive
            temp_dir = tempfile.mkdtemp(prefix="test_log_extract_")
            txt_files = cls.extract_archive(archive_path, temp_dir)

            if not txt_files:
                return {"archive_name": archive_name, "extracted_files": [], "results": [], "total_files": 0, "errors": ["No .txt files found in archive"]}

            # Parse each .txt file
            results = []
            for txt_file in txt_files:
                try:
                    result = cls.parse_file(txt_file)
                    results.append(result)
                except Exception as e:
                    errors.append(f"Failed to parse {Path(txt_file).name}: {str(e)}")

            return {"archive_name": archive_name, "extracted_files": [Path(f).name for f in txt_files], "results": results, "total_files": len(txt_files), "errors": errors}

        except Exception as e:
            return {"archive_name": archive_name, "extracted_files": [], "results": [], "total_files": 0, "errors": [f"Archive extraction failed: {str(e)}"]}

        finally:
            # Clean up temp directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @classmethod
    def parse_content(cls, content: str, filename: str = "uploaded_file.txt") -> dict[str, any]:
        """
        Parse test log content and extract test items.

        Args:
            content: String content of the test log file
            filename: Original filename for reference

        Returns:
            Dictionary with parsing results including ISN
        """
        lines = content.splitlines()

        # Extract ISN from filename
        isn = cls.extract_isn_from_filename(filename)

        # Find the test result section
        test_section = cls._extract_test_section(lines)

        if not test_section:
            return {"filename": filename, "isn": isn, "parsed_count": 0, "parsed_items": [], "errors": ["No test result section found"]}

        # Parse test items from the section (optimized for performance)
        parsed_items = []
        skipped_items = 0

        for line in test_section:
            result = cls._parse_line(line)

            if result is None:
                continue

            test_item, usl, lsl, value = result

            # Skip excluded values
            if value in cls.EXCLUDED_VALUES:
                skipped_items += 1
                continue

            parsed_items.append({"test_item": test_item, "usl": usl if usl else None, "lsl": lsl if lsl else None, "value": value})

        return {"filename": filename, "isn": isn, "parsed_count": len(parsed_items), "parsed_items": parsed_items}

    @classmethod
    def parse_content_enhanced(
        cls,
        content: str,
        filename: str = "uploaded_file.txt",
        criteria_rules: dict | None = None,
        show_only_criteria: bool = False,
    ) -> dict[str, any]:
        """
        Parse test log content with enhanced metadata, classification, and scoring.

        Args:
            content: String content of test log file
            filename: Original filename
            criteria_rules: Dict of criteria rules {test_name: TestLogCriteriaRule}
            show_only_criteria: If True, filter to only criteria-matched items

        Returns:
            Dictionary matching TestLogParseResponseEnhanced schema with:
            - metadata (8 fields from header)
            - station (from filename pattern)
            - parsed_items_enhanced (list of enriched items)
            - value_type_count, non_value_type_count, hex_value_count
            - avg_score, median_score (for value items)
        """
        lines = content.splitlines()

        # Extract ISN and station from filename
        isn = cls.extract_isn_from_filename(filename)
        station_from_filename = _extract_station_from_filename(filename)

        # Extract metadata from header
        metadata_dict = _extract_log_metadata(content)

        # Use station from metadata if available, otherwise use filename
        station = metadata_dict.get("station") or station_from_filename

        # Find test result section
        test_section = cls._extract_test_section(lines)

        if not test_section:
            return {
                "filename": filename,
                "isn": isn,
                "station": station,
                "metadata": metadata_dict,
                "parsed_count": 0,
                "parsed_items_enhanced": [],
                "value_type_count": 0,
                "non_value_type_count": 0,
                "hex_value_count": 0,
                "avg_score": None,
                "median_score": None,
                "errors": ["No test result section found"],
            }

        # Parse items with classification and scoring
        parsed_items_enhanced = []
        value_type_count = 0
        non_value_type_count = 0
        hex_value_count = 0
        scores = []

        for line in test_section:
            result = cls._parse_line(line)
            if result is None:
                continue

            test_item, usl_str, lsl_str, value_str = result

            # Skip excluded values
            if value_str in cls.EXCLUDED_VALUES:
                continue

            # Convert USL/LSL to float
            usl = _to_float(usl_str) if usl_str else None
            lsl = _to_float(lsl_str) if lsl_str else None

            # Classify value
            is_value_type, numeric_value, is_hex, hex_decimal = _classify_test_item_value(value_str)

            # Update counts
            if is_value_type:
                value_type_count += 1
            else:
                non_value_type_count += 1
            if is_hex:
                hex_value_count += 1

            # Check if item matches criteria
            criteria_rule = None
            matches_criteria = False
            if criteria_rules:
                # Try to match against any criteria pattern
                for _pattern_key, rule in criteria_rules.items():
                    if rule.pattern.search(test_item):
                        criteria_rule = rule
                        matches_criteria = True
                        break

            # Apply criteria filter if requested
            if show_only_criteria and not matches_criteria:
                continue

            # Calculate score for value-type items
            target_used = None
            score = None
            score_breakdown = None

            if is_value_type and numeric_value is not None:
                target_used, score, score_breakdown = _calculate_test_log_item_score(
                    test_item=test_item,
                    usl=usl,
                    lsl=lsl,
                    target=None,
                    actual=numeric_value,
                    criteria_rule=criteria_rule,
                )
                # Round score to 2 decimal places
                if score is not None:
                    score = round(score, 2)
                # Round breakdown values to 2 decimal places
                if score_breakdown:
                    for key in ["deviation", "raw_score", "final_score", "actual", "target_used"]:
                        if key in score_breakdown and score_breakdown[key] is not None:
                            score_breakdown[key] = round(score_breakdown[key], 2)
                scores.append(score)

            # Build enhanced item
            item_enhanced = {
                "test_item": test_item,
                "usl": usl,
                "lsl": lsl,
                "value": value_str,
                "is_value_type": is_value_type,
                "numeric_value": numeric_value,
                "is_hex": is_hex,
                "hex_decimal": hex_decimal,
                "matched_criteria": matches_criteria,
                "target": target_used,
                "score": score,
                "score_breakdown": score_breakdown,
            }

            parsed_items_enhanced.append(item_enhanced)

        # Calculate aggregate scores (rounded to 2 decimal places)
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        median_score = round(float(median(scores)), 2) if scores else None

        # Calculate PA ADJUSTED_POW items from SROM_OLD and SROM_NEW pairs
        pa_srom_pairs = cls._pair_pa_srom_items(parsed_items_enhanced)
        for base_key, (old_item, new_item) in pa_srom_pairs.items():
            adjusted_item = cls._create_adjusted_pow_item(base_key, old_item, new_item)
            if adjusted_item:
                parsed_items_enhanced.append(adjusted_item)
                non_value_type_count += 1  # Treated as non-value for display purposes

        return {
            "filename": filename,
            "isn": isn,
            "station": station,
            "metadata": metadata_dict,
            "parsed_count": len(parsed_items_enhanced),
            "parsed_items_enhanced": parsed_items_enhanced,
            "value_type_count": value_type_count,
            "non_value_type_count": non_value_type_count,
            "hex_value_count": hex_value_count,
            "avg_score": avg_score,
            "median_score": median_score,
        }

    @staticmethod
    def _is_pa_srom_test_item(test_item_name: str, pattern_type: str = "all") -> bool:
        """
        Check if a test item name matches PA SROM patterns.

        Args:
            test_item_name: Test item name to check
            pattern_type: "old", "new", or "all" (default)

        Returns:
            True if test item matches the specified pattern type
        """
        if not test_item_name:
            return False

        name_upper = test_item_name.upper()

        # Check for PA{1-4}_SROM_OLD or PA{1-4}_SROM_NEW patterns
        has_old = bool(re.search(r"PA[1-4]_SROM_OLD", name_upper))
        has_new = bool(re.search(r"PA[1-4]_SROM_NEW", name_upper))

        if pattern_type == "old":
            return has_old
        elif pattern_type == "new":
            return has_new
        else:  # "all"
            return has_old or has_new

    @staticmethod
    def _convert_hex_to_decimal(hex_value: str) -> int | None:
        """
        Convert hexadecimal string to decimal integer.

        Args:
            hex_value: Hexadecimal string (e.g., "0x235e" or "235e")

        Returns:
            Decimal integer or None if conversion fails
        """
        if not hex_value:
            return None

        try:
            hex_str = str(hex_value).strip()
            if not hex_str:
                return None

            # Convert hex to decimal (handles both "0x235e" and "235e" formats)
            if hex_str.lower().startswith("0x"):
                return int(hex_str, 16)
            else:
                return int(hex_str, 16)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _pair_pa_srom_items(cls, parsed_items: list[dict]) -> dict[str, tuple[dict, dict]]:
        """
        Pair PA SROM_OLD and SROM_NEW items for adjusted power calculation.

        Args:
            parsed_items: List of parsed test item dictionaries

        Returns:
            Dict mapping base_key to (old_item, new_item) tuples
            Example: "WiFi_PA1_SROM_5985_11AX_MCS9_B80" -> (old_item, new_item)
        """
        old_items = {}  # {base_key: item_dict}
        new_items = {}  # {base_key: item_dict}

        for item in parsed_items:
            test_item = item["test_item"]

            # Only process PA SROM items with hex values
            if not item.get("is_hex"):
                continue

            if cls._is_pa_srom_test_item(test_item, "old"):
                # Extract base key: WiFi_PA1_SROM_OLD_5985... -> WiFi_PA1_SROM_5985...
                base_key = re.sub(r"_SROM_OLD_", "_SROM_", test_item, flags=re.IGNORECASE)
                old_items[base_key] = item
            elif cls._is_pa_srom_test_item(test_item, "new"):
                base_key = re.sub(r"_SROM_NEW_", "_SROM_", test_item, flags=re.IGNORECASE)
                new_items[base_key] = item

        # Create pairs for matching items
        pairs = {}
        for base_key in set(old_items.keys()) & set(new_items.keys()):
            pairs[base_key] = (old_items[base_key], new_items[base_key])

        return pairs

    @classmethod
    def _create_adjusted_pow_item(cls, base_key: str, old_item: dict, new_item: dict) -> dict | None:
        """
        Create PA ADJUSTED_POW item from paired SROM_OLD and SROM_NEW.

        Formula: (SROM_NEW decimal - SROM_OLD decimal) / 256

        Args:
            base_key: Base key for the item pair
            old_item: SROM_OLD item dictionary
            new_item: SROM_NEW item dictionary

        Returns:
            Item dictionary matching ParsedTestItemEnhanced structure or None if calculation fails
        """
        old_decimal = old_item.get("hex_decimal")
        new_decimal = new_item.get("hex_decimal")

        if old_decimal is None or new_decimal is None:
            return None

        # Calculate adjusted power
        adjusted_value = (new_decimal - old_decimal) / 256
        adjusted_value_rounded = round(adjusted_value, 2)

        # Create item name: WiFi_PA1_ADJUSTED_POW_5985_11AX_MCS9_B80
        adjusted_item_name = re.sub(r"_SROM_", "_ADJUSTED_POW_", base_key, flags=re.IGNORECASE)

        # Build item dict matching ParsedTestItemEnhanced structure
        return {
            "test_item": adjusted_item_name,
            "usl": None,
            "lsl": None,
            "value": str(adjusted_value_rounded),
            "is_value_type": True,
            "numeric_value": adjusted_value_rounded,
            "is_hex": False,
            "hex_decimal": None,
            "matched_criteria": False,
            "target": None,
            "score": None,
            "score_breakdown": None,
            "is_calculated": True,
        }

    @classmethod
    def _extract_test_section(cls, lines: list[str]) -> list[str] | None:
        """
        Extract lines between Start and End markers (flexible '=' count).

        Args:
            lines: List of all lines in the file

        Returns:
            List of lines in the test section, or None if markers not found
        """
        start_idx = None
        end_idx = None

        for idx, line in enumerate(lines):
            if cls.START_MARKER_PATTERN.search(line):
                start_idx = idx + 1  # Start from next line after marker
            elif cls.END_MARKER_PATTERN.search(line):
                end_idx = idx
                break

        if start_idx is None or end_idx is None:
            return None

        return lines[start_idx:end_idx]

    @classmethod
    def _parse_line(cls, line: str) -> tuple[str, str, str, str] | None:
        """
        Parse a single test item line.

        Args:
            line: Line to parse

        Returns:
            Tuple of (test_item, usl, lsl, value) or None if line doesn't match pattern
        """
        match = cls.LINE_PATTERN.match(line.strip())

        if not match:
            return None

        test_item, usl, lsl, value = match.groups()

        return test_item, usl, lsl, value

    @classmethod
    def compare_files(cls, file_paths: list[str]) -> dict[str, any]:
        """
        Compare test items across multiple files (optimized for batch processing).

        Args:
            file_paths: List of paths to test log files

        Returns:
            Dictionary containing comparison results with ISNs
        """
        if len(file_paths) < 2:
            raise ValueError("At least 2 files required for comparison")

        # Parse all files (optimized)
        parsed_files = [cls.parse_file(path) for path in file_paths]

        # Build test item index for each file
        file_items = [{item["test_item"]: item for item in parsed["parsed_items"]} for parsed in parsed_files]

        # Find all test items and common items
        all_test_items = set()
        for items_dict in file_items:
            all_test_items.update(items_dict.keys())

        common_items = set.intersection(*[set(items.keys()) for items in file_items])

        # Build comparison data (optimized)
        comparison_results = []

        for test_item in sorted(all_test_items):
            # Collect values from files that have this item
            values = []
            usl = None
            lsl = None

            for idx, items_dict in enumerate(file_items):
                if test_item in items_dict:
                    item = items_dict[test_item]
                    values.append({"isn": parsed_files[idx]["isn"], "value": item["value"]})

                    # Use USL/LSL from first occurrence
                    if usl is None:
                        usl = item["usl"]
                    if lsl is None:
                        lsl = item["lsl"]

            item_data = {"test_item": test_item, "usl": usl, "lsl": lsl, "is_common": test_item in common_items, "values": values}

            # Add numeric analysis only if multiple values and all numeric
            if len(values) > 1:
                try:
                    numeric_values = [float(v["value"]) for v in values]
                    item_data["min"] = min(numeric_values)
                    item_data["max"] = max(numeric_values)
                    item_data["range"] = item_data["max"] - item_data["min"]
                    item_data["avg"] = sum(numeric_values) / len(numeric_values)
                except (ValueError, TypeError):
                    # Values are not numeric, skip analysis
                    pass

            comparison_results.append(item_data)

        return {
            "total_files": len(file_paths),
            "total_items": len(all_test_items),
            "common_items": len(common_items),
            "file_summary": [{"filename": p["filename"], "isn": p["isn"], "parsed_count": p["parsed_count"]} for p in parsed_files],
            "comparison": comparison_results,
        }

    @classmethod
    def compare_files_enhanced(
        cls,
        file_paths: list[str],
        criteria_rules: dict | None = None,
        show_only_criteria: bool = False,
    ) -> dict[str, any]:
        """
        Compare test items across multiple files with enhanced per-ISN deviations and scoring.

        Calculates:
        - Per-ISN values, deviation from median, and individual scores
        - Median value as baseline (or criteria target if available)
        - Separates value-type items (numeric) from non-value items
        - Aggregate statistics (avg deviation, avg score, median score)

        Args:
            file_paths: List of paths to test log files
            criteria_rules: Optional criteria rules dict
            show_only_criteria: If True, only show items matching criteria

        Returns:
            Dictionary matching CompareResponseEnhanced schema with:
            - comparison_value_items: List of numeric items with per-ISN data
            - comparison_non_value_items: List of non-numeric items
            - total_files, total_value_items, total_non_value_items
        """
        if len(file_paths) < 2:
            raise ValueError("At least 2 files required for comparison")

        # Parse all files using enhanced parser
        parsed_files = [
            cls.parse_content_enhanced(
                content=open(path, encoding="utf-8", errors="ignore").read(),
                filename=Path(path).name,
                criteria_rules=criteria_rules,
                show_only_criteria=False,  # Don't filter during parsing
            )
            for path in file_paths
        ]

        # Build test item index for each file
        file_items = [{item["test_item"]: item for item in parsed["parsed_items_enhanced"]} for parsed in parsed_files]

        # Find all test items
        all_test_items = set()
        for items_dict in file_items:
            all_test_items.update(items_dict.keys())

        # Separate value and non-value items
        comparison_value_items = []
        comparison_non_value_items = []

        for test_item in sorted(all_test_items):
            # Collect data from files that have this item
            per_isn_data = []
            usl = None
            lsl = None
            is_value_type = None
            matches_criteria = False
            criteria_rule = None

            # Check criteria match
            if criteria_rules:
                # Try to match against any criteria pattern
                for _pattern_key, rule in criteria_rules.items():
                    if rule.pattern.search(test_item):
                        criteria_rule = rule
                        matches_criteria = True
                        break

            # Apply criteria filter if requested
            if show_only_criteria and not matches_criteria:
                continue

            # Collect per-ISN values
            for idx, items_dict in enumerate(file_items):
                if test_item in items_dict:
                    item = items_dict[test_item]
                    per_isn_data.append(
                        {
                            "isn": parsed_files[idx]["isn"],
                            "value": item["value"],
                            "is_value_type": item["is_value_type"],
                            "numeric_value": item["numeric_value"],
                            "is_hex": item["is_hex"],
                            "hex_decimal": item["hex_decimal"],
                        }
                    )

                    # Use USL/LSL from first occurrence
                    if usl is None:
                        usl = item["usl"]
                    if lsl is None:
                        lsl = item["lsl"]
                    if is_value_type is None:
                        is_value_type = item["is_value_type"]

            # Skip if no data
            if not per_isn_data:
                continue

            # Separate value vs non-value items
            if is_value_type:
                # Extract numeric values for deviation calculation
                numeric_values = [d["numeric_value"] for d in per_isn_data if d["numeric_value"] is not None]

                if not numeric_values:
                    continue

                # Determine baseline (median or criteria target)
                if criteria_rule and criteria_rule.target is not None:
                    baseline = criteria_rule.target
                else:
                    baseline = float(median(numeric_values))

                # Calculate per-ISN deviations and scores
                deviations = []
                scores = []

                for isn_data in per_isn_data:
                    if isn_data["numeric_value"] is not None:
                        deviation = round(isn_data["numeric_value"] - baseline, 2)
                        deviations.append(deviation)

                        # Calculate score
                        _, score, score_breakdown = _calculate_test_log_item_score(
                            test_item=test_item,
                            usl=usl,
                            lsl=lsl,
                            target=baseline,
                            actual=isn_data["numeric_value"],
                            criteria_rule=criteria_rule,
                        )

                        # Round score to 2 decimal places
                        if score is not None:
                            score = round(score, 2)
                        # Round breakdown values to 2 decimal places
                        if score_breakdown:
                            for key in ["deviation", "raw_score", "final_score", "actual", "target_used", "baseline"]:
                                if key in score_breakdown and score_breakdown[key] is not None:
                                    score_breakdown[key] = round(score_breakdown[key], 2)

                        scores.append(score)

                        # Attach to isn_data
                        isn_data["deviation"] = deviation
                        isn_data["score"] = score
                        isn_data["score_breakdown"] = score_breakdown
                    else:
                        isn_data["deviation"] = None
                        isn_data["score"] = None
                        isn_data["score_breakdown"] = None

                # Calculate aggregates (rounded to 2 decimal places)
                avg_deviation = round(sum(abs(d) for d in deviations) / len(deviations), 2) if deviations else 0.0
                avg_score = round(sum(scores) / len(scores), 2) if scores else None
                median_score = round(float(median(scores)), 2) if scores else None

                comparison_value_items.append(
                    {
                        "test_item": test_item,
                        "usl": usl,
                        "lsl": lsl,
                        "baseline": baseline,
                        "per_isn_data": per_isn_data,
                        "avg_deviation": avg_deviation,
                        "avg_score": avg_score,
                        "median_score": median_score,
                        "matched_criteria": matches_criteria,
                    }
                )

            else:
                # Non-value item (hex, status, text)
                comparison_non_value_items.append(
                    {
                        "test_item": test_item,
                        "usl": usl,
                        "lsl": lsl,
                        "per_isn_data": per_isn_data,
                        "matched_criteria": matches_criteria,
                    }
                )

        # Calculate PA ADJUSTED_POW for comparison mode
        # Process each file to create adjusted items
        adjusted_items_by_file = []  # List of {file_idx: int, adjusted_items: []}

        for file_idx, parsed in enumerate(parsed_files):
            pa_srom_pairs = cls._pair_pa_srom_items(parsed["parsed_items_enhanced"])
            file_adjusted = []

            for base_key, (old_item, new_item) in pa_srom_pairs.items():
                adjusted_item = cls._create_adjusted_pow_item(base_key, old_item, new_item)
                if adjusted_item:
                    # Add ISN context
                    adjusted_item["isn"] = parsed["isn"]
                    file_adjusted.append(adjusted_item)

            if file_adjusted:
                adjusted_items_by_file.append({"file_idx": file_idx, "items": file_adjusted})

        # Group adjusted items by test_item across files
        adjusted_items_grouped = {}
        for file_data in adjusted_items_by_file:
            for item in file_data["items"]:
                test_item = item["test_item"]
                if test_item not in adjusted_items_grouped:
                    adjusted_items_grouped[test_item] = []
                adjusted_items_grouped[test_item].append(
                    {"file_idx": file_data["file_idx"], "isn": item["isn"], "value": item["value"], "numeric_value": item["numeric_value"], "is_calculated": True}
                )

        # Create comparison items for adjusted power
        for test_item, per_file_data in adjusted_items_grouped.items():
            per_isn_data = []

            for file_idx in range(len(parsed_files)):
                # Find data for this file
                file_data = next((d for d in per_file_data if d["file_idx"] == file_idx), None)
                if file_data:
                    per_isn_data.append(
                        {
                            "isn": file_data["isn"],
                            "value": file_data["value"],
                            "is_value_type": True,
                            "numeric_value": file_data["numeric_value"],
                            "is_hex": False,
                            "hex_decimal": None,
                            "deviation": None,
                            "score": None,
                            "score_breakdown": None,
                            "is_calculated": True,
                        }
                    )
                else:
                    # No adjusted power for this ISN (missing SROM pair)
                    per_isn_data.append(
                        {
                            "isn": parsed_files[file_idx]["isn"],
                            "value": "N/A",
                            "is_value_type": False,
                            "numeric_value": None,
                            "is_hex": False,
                            "hex_decimal": None,
                            "deviation": None,
                            "score": None,
                            "score_breakdown": None,
                            "is_calculated": True,
                        }
                    )

            comparison_non_value_items.append({"test_item": test_item, "usl": None, "lsl": None, "per_isn_data": per_isn_data, "matched_criteria": False})

        # Sort items using custom test item sorting logic
        comparison_value_items.sort(key=cls._test_item_sort_key)
        comparison_non_value_items.sort(key=cls._test_item_sort_key)

        # Build file summaries with metadata
        file_summaries = [
            {
                "filename": parsed["filename"],
                "isn": parsed["isn"],
                "metadata": parsed["metadata"],
                "parsed_count": parsed["parsed_count"],
                "avg_score": parsed["avg_score"],
            }
            for parsed in parsed_files
        ]

        return {
            "total_files": len(file_paths),
            "total_value_items": len(comparison_value_items),
            "total_non_value_items": len(comparison_non_value_items),
            "file_summaries": file_summaries,
            "comparison_value_items": comparison_value_items,
            "comparison_non_value_items": comparison_non_value_items,
        }

    @staticmethod
    def _test_item_sort_key(item: dict) -> tuple:
        """
        Generate sort key for test items with custom ordering rules.

        Sorting order based on original file order:
        - BT tests: Sort by criteria first (POW→FREQ_KHZ→PER), then antenna
        - WiFi tests: Sort by frequency→antenna→criteria

        Returns:
            Tuple for sorting: varies based on test type
        """
        test_item = item["test_item"]

        # Detect if this is a BT test (no WiFi_ prefix, contains _BT)
        is_bt_test = not test_item.startswith("WiFi_") and "_BT" in test_item

        # Extract frequency + standard + modulation + bandwidth pattern
        # Patterns to match:
        # - NNNN_XX (e.g., 2404_BT, 2480_BT)
        # - NNNN_STANDARD_MODULATION_BNN (e.g., 6175_11AX_MCS9_B20, 6115_11AG_OFDM6_B20)
        # - NNNN_STANDARD_MODULATION_BNNN-N (e.g., 6265_11BE_MCS9_B320-2)
        freq_pattern = re.search(r"(\d{4}_(?:\w+(?:_\w+)*_B\d+(?:-\d+)?|\w+))", test_item)
        freq_std_mod_bw = freq_pattern.group(1) if freq_pattern else ""

        # Extract antenna number (TX1, TX2, RX1, PA1, etc.)
        antenna_match = re.search(r"(TX|RX|PA|ANT)(\d+)", test_item)
        antenna_type = antenna_match.group(1) if antenna_match else ""
        antenna_num = int(antenna_match.group(2)) if antenna_match else 0

        # Define criteria family ordering
        # Order matters - check more specific patterns first to avoid false matches
        criteria_families = {
            # BT tests (special ordering for BT)
            "POW": 5 if is_bt_test else 30,  # BT: POW comes first; WiFi: POW after PA tests
            "FREQ_KHZ": 6,  # BT only, check before FREQ
            # Fixture/problem detection
            "FIXTURE_OR_DUT_PROBLEM_POW": 10,
            "FIXTURE_OR_DUT_PROBLEM_FREQ": 11,
            "FIXTURE_OR_DUT_PROBLEM_EVM": 12,
            # PA family (check specific patterns first)
            "POW_OLD": 20,
            "POW_DIF_ABS": 21,
            "POW_DIF": 21,  # Alias for POW_DIF_ABS
            "SROM_OLD": 22,
            "SROM_NEW": 23,
            # TX/RX family (WiFi tests)
            "EVM": 31,
            "FREQ": 32,
            "MASK": 33,
            "LO_LEAKAGE": 34,
            "PER": 40,
            "RSSI": 41,
        }

        # Find which criteria family this item belongs to
        criteria_order = 100  # Default for unknown criteria
        for criteria_name, order in criteria_families.items():
            if criteria_name in test_item:
                criteria_order = order
                break

        # BT tests: Sort by frequency, then criteria, then antenna
        # WiFi tests: Sort by frequency, then antenna, then criteria
        if is_bt_test:
            return (freq_std_mod_bw, criteria_order, antenna_type, antenna_num, test_item)
        else:
            return (freq_std_mod_bw, antenna_type, antenna_num, criteria_order, test_item)


# ============================================================================
# Enhanced functions for BY UPLOAD LOG feature
# ============================================================================


@dataclass(slots=True)
class TestLogCriteriaRule:
    """Criteria rule for test log parsing."""

    pattern: re.Pattern[str]
    usl: float | None
    lsl: float | None
    target: float | None


# Criteria file parsing pattern (reused from external_api_client.py)
_CRITERIA_LINE_PATTERN = re.compile(r'^\s*"(?P<test>.+?)"\s*<(?P<usl>[^,]*),(?P<lsl>[^>]*)>\s*===\>\s*"(?P<target>.*)"\s*$')


def _to_float(value: str | None) -> float | None:
    """Convert string to float, return None if conversion fails."""
    if not value or not value.strip():
        return None
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return None


def _extract_log_metadata(content: str) -> dict:
    """
    Extract metadata from test log header.

    Searches for header section between << START TESTING >> markers
    and extracts:
    - Test date/time
    - Device ID
    - Station name
    - Script version
    - Duration
    - SFIS status
    - Result
    - Counter

    Args:
        content: Full log file content

    Returns:
        Dictionary with metadata fields (None for missing fields)
    """
    metadata = {
        "test_date": None,
        "device": None,
        "station": None,
        "script_version": None,
        "duration_seconds": None,
        "sfis_status": None,
        "result": None,
        "counter": None,
    }

    # Find header section - look for the equals line before [Time]
    # The header is typically at the beginning of the file, within first 3000 chars
    header_section = content[:3000]

    # Extract fields using regex
    patterns = {
        "test_date": r"\[Time\]\s*=\s*(.+)",
        "device": r"\[Device\]\s*=\s*(.+)",
        "station": r"\[Station\]\s*=\s*(.+)",
        "script_version": r"\[Script_Ver\]\s*=\s*(.+)",
        "duration_seconds": r"\[Duration\]\s*=\s*(\d+)\s*seconds",
        "sfis_status": r"\[SFIS Status\]\s*=\s*(.+)",
        "result": r"\[Result\]\s*=\s*(.+)",
        "counter": r"\[Counter\]\s*=\s*(\d+)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, header_section)
        if match:
            value = match.group(1).strip()

            if field == "test_date":
                # Parse datetime: MM-DD-YYYY HH:MM:SS
                try:
                    metadata[field] = datetime.strptime(value, "%m-%d-%Y %H:%M:%S")
                except ValueError:
                    metadata[field] = None
            elif field in ("duration_seconds", "counter"):
                try:
                    metadata[field] = int(value)
                except ValueError:
                    metadata[field] = None
            else:
                metadata[field] = value

    return metadata


def _extract_station_from_filename(filename: str) -> str:
    """
    Extract station name from filename pattern.

    Pattern: [Station]_[ISN]_[YYYY]_[MM]_[DD]_[HHmmss]
    Returns first part before underscore as station name.

    Args:
        filename: Filename (with or without extension)

    Returns:
        Station name or "Unknown_Station" if not found
    """
    name = Path(filename).stem
    match = re.match(r"^([^_]+)_", name)
    if match:
        return match.group(1)
    return "Unknown_Station"


def _parse_test_log_criteria_file(content: bytes) -> dict[str, TestLogCriteriaRule]:
    """
    Parse criteria .ini file for test log filtering and scoring.

    Format:
        [Test_Items]
        "TEST_ITEM" <USL,LSL>  ===> "TargetValue"

    Args:
        content: Bytes content of .ini file

    Returns:
        Dict mapping lowercase test_item names to TestLogCriteriaRule objects
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("utf-8", errors="ignore")

    rules: dict[str, TestLogCriteriaRule] = {}
    in_test_items_section = False
    section_pattern = re.compile(r"^\s*\[Test_Items\]\s*$", re.IGNORECASE)

    for line in text.splitlines():
        # Remove inline comments (anything after ; or #)
        if ";" in line:
            line = line.split(";", 1)[0]
        if "#" in line:
            line = line.split("#", 1)[0]

        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Check for [Test_Items] section header
        if section_pattern.match(stripped):
            in_test_items_section = True
            continue

        # Check for other section headers (stops parsing)
        if stripped.startswith("["):
            in_test_items_section = False
            continue

        # Parse criteria lines only in Test_Items section
        if not in_test_items_section:
            continue

        # Parse line using criteria pattern
        match = _CRITERIA_LINE_PATTERN.match(stripped)
        if not match:
            continue

        test_pattern = match.group("test")
        usl = _to_float(match.group("usl"))
        lsl = _to_float(match.group("lsl"))
        target = _to_float(match.group("target"))

        # Smart pattern expansion: if pattern contains common prefixes without numbers,
        # automatically make them match numbered variants (TX1-4, PA1-4, RX1-4, etc.)
        # Examples:
        #   "WiFi_TX_FIXTURE" -> "WiFi_TX\d?_FIXTURE" (matches TX_, TX1_, TX2_, TX3_, TX4_, etc.)
        #   "WiFi_PA_POW" -> "WiFi_PA\d?_POW" (matches PA_, PA1_, PA2_, PA3_, PA4_, etc.)
        regex_pattern = test_pattern

        # Common patterns to auto-expand (add number placeholders)
        # Pattern: look for PREFIX followed by underscore or end of word
        # Replacement: PREFIX + optional digit(s) + underscore
        expansions = [
            (r"_TX_", r"_TX\\d?_"),  # _TX_ -> _TX\d?_ (matches TX_, TX1_, TX2_, etc.)
            (r"_RX_", r"_RX\\d?_"),  # _RX_ -> _RX\d?_
            (r"_PA_", r"_PA\\d?_"),  # _PA_ -> _PA\d?_
            (r"_ANT_", r"_ANT\\d?_"),  # _ANT_ -> _ANT\d?_
            (r"_RSSI_", r"_RSSI\\d?_"),  # _RSSI_ -> _RSSI\d?_
            (r"_CHAIN_", r"_CHAIN\\d?_"),  # _CHAIN_ -> _CHAIN\d?_
        ]

        # Apply expansions only if the pattern doesn't already contain regex special chars
        # (to avoid breaking user's intentional regex patterns)
        regex_special_chars = r".*+?[]{}()^$|\\"
        has_regex_chars = any(char in test_pattern for char in regex_special_chars)

        if not has_regex_chars:
            # Auto-expand common numbered patterns
            for pattern, replacement in expansions:
                regex_pattern = re.sub(pattern, replacement, regex_pattern)  # Compile pattern for regex matching
        try:
            compiled = re.compile(regex_pattern, re.IGNORECASE)
        except re.error:
            continue  # Skip invalid patterns

        # Store with lowercase key for matching
        key = test_pattern.lower()
        rules[key] = TestLogCriteriaRule(
            pattern=compiled,
            usl=usl,
            lsl=lsl,
            target=target,
        )

    return rules


def _parse_test_log_criteria_json(content: bytes) -> dict[str, TestLogCriteriaRule]:
    """
    Parse criteria JSON file for test log filtering and scoring.

    JSON Format:
        {
            "criteria": [
                {
                    "test_item": "TEST_ITEM_PATTERN",
                    "ucl": 20.0,
                    "lcl": 10.0,
                    "target": 15.0
                },
                ...
            ]
        }

    Args:
        content: Bytes content of .json file

    Returns:
        Dict mapping lowercase test_item names to TestLogCriteriaRule objects
    """
    import json

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("utf-8", errors="ignore")

    rules: dict[str, TestLogCriteriaRule] = {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e

    # Support both "criteria" array and direct array format
    criteria_list = data.get("criteria", data) if isinstance(data, dict) else data

    if not isinstance(criteria_list, list):
        raise ValueError("JSON criteria must be an array or an object with 'criteria' array")

    for item in criteria_list:
        if not isinstance(item, dict):
            continue

        test_pattern = item.get("test_item", "")
        if not test_pattern:
            continue

        usl = _to_float(str(item.get("ucl", ""))) if item.get("ucl") is not None else None
        lsl = _to_float(str(item.get("lcl", ""))) if item.get("lcl") is not None else None
        target = _to_float(str(item.get("target", ""))) if item.get("target") is not None else None

        # Auto-expand patterns (same logic as INI parser)
        regex_pattern = test_pattern
        regex_special_chars = r".*+?[]{}()^$|\\"
        has_regex_chars = any(char in test_pattern for char in regex_special_chars)

        if not has_regex_chars:
            expansions = [
                (r"_TX_", r"_TX\\d?_"),
                (r"_RX_", r"_RX\\d?_"),
                (r"_PA_", r"_PA\\d?_"),
                (r"_ANT_", r"_ANT\\d?_"),
                (r"_RSSI_", r"_RSSI\\d?_"),
                (r"_CHAIN_", r"_CHAIN\\d?_"),
            ]
            for pattern, replacement in expansions:
                regex_pattern = re.sub(pattern, replacement, regex_pattern)

        try:
            compiled = re.compile(regex_pattern, re.IGNORECASE)
        except re.error:
            continue

        key = test_pattern.lower()
        rules[key] = TestLogCriteriaRule(
            pattern=compiled,
            usl=usl,
            lsl=lsl,
            target=target,
        )

    return rules


def parse_test_log_criteria_file(content: bytes, filename: str) -> dict[str, TestLogCriteriaRule]:
    """
    Parse JSON criteria file.

    Args:
        content: Bytes content of criteria file
        filename: Original filename (must be .json)

    Returns:
        Dict mapping lowercase test_item names to TestLogCriteriaRule objects

    Raises:
        ValueError: If file format is not .json or parsing fails
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".json"):
        return _parse_test_log_criteria_json(content)
    else:
        raise ValueError(f"Unsupported criteria file format: {filename}. Only .json is supported.")


def _classify_test_item_value(value: str) -> tuple[bool, float | None, bool, int | None]:
    """
    Classify test item value as numeric (value) or non-numeric (non-value).

    Checks for:
    - Hexadecimal values (0x...)
    - Numeric float/int values
    - Status strings (PASS/FAIL/VALUE)

    Args:
        value: Raw value string

    Returns:
        Tuple of (is_value_type, numeric_value, is_hex, hex_decimal)
    """
    value_stripped = value.strip()

    # Check for hexadecimal
    hex_match = re.match(r"^0x([0-9a-fA-F]+)$", value_stripped, re.IGNORECASE)
    if hex_match:
        try:
            hex_decimal = int(hex_match.group(1), 16)
            return (False, None, True, hex_decimal)
        except ValueError:
            return (False, None, True, None)

    # Try float conversion
    try:
        numeric_value = float(value_stripped)
        # Check for inf/nan
        if not (numeric_value != numeric_value or numeric_value == float("inf") or numeric_value == float("-inf")):  # noqa: PLR0124
            return (True, numeric_value, False, None)
    except (ValueError, TypeError):
        pass

    # Non-numeric (status string, text, etc.)
    return (False, None, False, None)


def _calculate_test_log_item_score(
    test_item: str,
    usl: float | None,
    lsl: float | None,
    target: float | None,
    actual: float,
    criteria_rule: TestLogCriteriaRule | None,
) -> tuple[float | None, float, dict]:
    """
    Calculate score for a test item with detailed breakdown and LaTeX formula.

    Score calculation logic:
    1. Determine target: criteria target > (USL+LSL)/2 > USL > LSL > actual
    2. Detect measurement category (EVM, Frequency, PER, Power, etc.)
    3. Apply category-specific scoring formula
    4. Return 0-10 score with breakdown

    Args:
        test_item: Test item name
        usl: Upper Spec Limit
        lsl: Lower Spec Limit
        target: Target from criteria (if available)
        actual: Actual measured value
        criteria_rule: Matched criteria rule (if any)

    Returns:
        Tuple of (target_used, score, breakdown_dict)
    """
    # Determine target value
    if criteria_rule and criteria_rule.target is not None:
        target_used = criteria_rule.target
    elif usl is not None and lsl is not None:
        target_used = (usl + lsl) / 2.0
    elif usl is not None:
        target_used = usl
    elif lsl is not None:
        target_used = lsl
    else:
        target_used = actual

    # Use criteria USL/LSL if provided
    if criteria_rule:
        if criteria_rule.usl is not None:
            usl = criteria_rule.usl
        if criteria_rule.lsl is not None:
            lsl = criteria_rule.lsl

    # Detect category
    category = _detect_measurement_category(test_item)

    # Calculate score based on category
    if category == "EVM":
        score, formula = _calculate_evm_score(usl, actual)
        method = "EVM Score"
    elif category == "Frequency":
        score, formula = _calculate_freq_score(usl, lsl, target_used, actual)
        method = "Frequency Score"
    elif category == "PER":
        score, formula = _calculate_per_score(usl, actual)
        method = "PER Score"
    elif category == "PA_ADJUSTED_POWER":
        score, formula = _calculate_pa_adjusted_power_score(actual)
        method = "PA Adjusted Power Score"
    elif category == "POW_DIF_ABS":
        score, formula = _calculate_pa_pow_dif_abs_score(actual, usl)
        method = "Power Difference Score"
    elif usl is not None and lsl is not None:
        score, formula = _calculate_bounded_measurement_score(usl, lsl, target_used, actual)
        method = "Bounded Measurement"
    else:
        # Fallback: simple deviation-based score
        if target_used != 0:
            deviation = abs(target_used - actual)
            score = max(0.0, 10.0 - (deviation / abs(target_used)) * 10.0)
        else:
            score = 10.0 if actual == 0 else 0.0
        formula = r"$score = \max(0, 10 - \frac{|target - actual|}{|target|} \times 10)$"
        method = "Simple Deviation"

    # Build breakdown
    breakdown = {
        "category": category or "General",
        "method": method,
        "usl": usl,
        "lsl": lsl,
        "target_used": target_used,
        "actual": actual,
        "deviation": abs(target_used - actual) if target_used is not None else 0.0,
        "raw_score": score,
        "final_score": score,
        "formula_latex": formula,
    }

    return (target_used, score, breakdown)


def _detect_measurement_category(test_item: str) -> str | None:
    """Detect measurement category from test item name (simplified version)."""
    name_upper = test_item.upper()

    if "EVM" in name_upper:
        return "EVM"
    elif "FREQ" in name_upper or "FREQUENCY" in name_upper:
        return "Frequency"
    elif "PER" in name_upper and ("_PER_" in name_upper or name_upper.endswith("PER")):
        return "PER"
    elif "PA_ADJUSTED_POWER" in name_upper:
        return "PA_ADJUSTED_POWER"
    elif "POW_DIF_ABS" in name_upper:
        return "POW_DIF_ABS"
    return None


def _calculate_evm_score(usl: float | None, actual: float) -> tuple[float, str]:
    """Calculate EVM score (lower is better)."""
    if usl is None or usl == 0:
        return (10.0 if actual <= 0 else 0.0, r"$score = 10$ if $actual \leq 0$")

    score = 10.0 * (1.0 - abs(actual) / usl)
    score = max(0.0, min(10.0, score))

    formula = r"$score = 10 \times \left(1 - \frac{|actual|}{USL}\right)$"
    return (score, formula)


def _calculate_freq_score(
    usl: float | None,
    lsl: float | None,
    target: float,
    actual: float,
) -> tuple[float, str]:
    """Calculate frequency score (closer to target is better)."""
    if usl is None or lsl is None:
        return (10.0, r"$score = 10$ (no limits)")

    deviation = abs(actual - target)
    max_allowed_deviation = max(abs(target - lsl), abs(usl - target))

    if max_allowed_deviation == 0:
        score = 10.0 if deviation == 0 else 0.0
    else:
        score = 10.0 * (1.0 - deviation / max_allowed_deviation)
        score = max(0.0, min(10.0, score))

    formula = r"$score = 10 \times \left(1 - \frac{|actual - target|}{\max(|target - LSL|, |USL - target|)}\right)$"
    return (score, formula)


def _calculate_per_score(usl: float | None, actual: float) -> tuple[float, str]:
    """Calculate PER score (lower is better, 0 is ideal)."""
    if usl is None or usl == 0:
        return (10.0 if actual == 0 else 0.0, r"$score = 10$ if $actual = 0$")

    score = 10.0 * (1.0 - actual / usl)
    score = max(0.0, min(10.0, score))

    formula = r"$score = 10 \times \left(1 - \frac{actual}{USL}\right)$"
    return (score, formula)


def _calculate_pa_adjusted_power_score(actual: float, threshold: float = 5.0) -> tuple[float, str]:
    """Calculate PA adjusted power score (closer to 0 is better)."""
    deviation = abs(actual)

    if deviation <= threshold:
        score = 10.0 - (deviation / threshold) * 2.0
    else:
        score = 8.0 - ((deviation - threshold) / threshold) * 8.0

    score = max(0.0, min(10.0, score))

    formula = r"$score = \begin{cases} 10 - \frac{|actual|}{5} \times 2 & \text{if } |actual| \leq 5 \\ 8 - \frac{|actual| - 5}{5} \times 8 & \text{otherwise} \end{cases}$"
    return (score, formula)


def _calculate_pa_pow_dif_abs_score(actual: float, usl: float | None = None) -> tuple[float, str]:
    """Calculate PA power difference absolute score (0 is ideal)."""
    threshold = usl if usl is not None else 1.0

    if actual <= threshold:
        score = 10.0 - (actual / threshold) * 2.0
    else:
        score = 8.0 - ((actual - threshold) / threshold) * 8.0

    score = max(0.0, min(10.0, score))

    formula = r"$score = \begin{cases} 10 - \frac{actual}{threshold} \times 2 & \text{if } actual \leq threshold \\ 8 - \frac{actual - threshold}{threshold} \times 8 & \text{otherwise} \end{cases}$"
    return (score, formula)


def _calculate_bounded_measurement_score(
    usl: float,
    lsl: float,
    target: float,
    actual: float,
) -> tuple[float, str]:
    """Calculate bounded measurement score with USL and LSL."""
    # Check if within bounds
    if actual < lsl or actual > usl:
        score = 0.0
    else:
        # Calculate score based on distance from target
        deviation = abs(actual - target)
        max_allowed_deviation = max(abs(target - lsl), abs(usl - target))

        if max_allowed_deviation == 0:
            score = 10.0 if deviation == 0 else 0.0
        else:
            score = 10.0 * (1.0 - deviation / max_allowed_deviation)
            score = max(0.0, min(10.0, score))

    formula = r"$score = \begin{cases} 0 & \text{if } actual < LSL \text{ or } actual > USL \\ 10 \times \left(1 - \frac{|actual - target|}{\max(|target - LSL|, |USL - target|)}\right) & \text{otherwise} \end{cases}$"
    return (score, formula)
