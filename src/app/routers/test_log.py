"""
Router for test log parsing and comparison endpoints.
"""

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from ..models.test_log import (
    CompareResponse,
    CompareResponseEnhanced,
    TestLogParseResponse,
    TestLogParseResponseEnhanced,
)
from ..services.test_log_parser import TestLogParser, _parse_test_log_criteria_file

router = APIRouter(prefix="/api/test-log", tags=["Test_Log_Processing"])


def validate_test_log_pattern(content: str) -> bool:
    """
    Check if file content contains valid test log patterns.
    
    Valid patterns:
    1. << START TESTING >> repeated 4 times
    2. =========[Start SFIS Test Result]====== format with TEST/UCL/LCL/VALUE
    3. *** Test flow *** format with Duration/Status/Result/Counter
    
    Args:
        content: File content as string
        
    Returns:
        True if file contains at least one valid pattern, False otherwise
    """
    # Pattern 1: << START TESTING >> repeated 4 times
    if "<< START TESTING >>  << START TESTING >>  << START TESTING >>  << START TESTING >>" in content:
        return True
    
    # Pattern 2: SFIS Test Result format
    if "=========[Start SFIS Test Result]======" in content and '"TEST" <"UCL","LCL">  ===> "VALUE"' in content:
        return True
    
    # Pattern 3: Test flow format
    if "*** Test flow ***" in content and "[Result]=" in content:
        return True
    
    return False

# Directory for temporary file storage
UPLOAD_DIR = Path("data/uploads/test_logs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/parse",
    response_model=None,
    summary="Parse test log file or archive (with optional criteria filtering)",
    description="""
    Upload and parse test log file (.txt) or archive (.zip, .rar, .7z).

    **Optional criteria filtering:**
    - Upload .ini criteria file to filter and score test items
    - Criteria format: [Test_Items] section with "TEST" <USL,LSL> ===> "Target"
    - Enhanced response includes metadata, classification, scoring with LaTeX formulas

    **Parser behavior:**
    - Flexible markers (any number of '=' characters)
    - USL/LSL spec limits extracted
    - FAIL items included for tracking
    - PASS/VALUE items excluded
    - ISN extraction from filename pattern: [Station]_[ISN]_[Date]
    """,
)
async def parse_test_log(
    file: Annotated[UploadFile, File(description="Test log file (.txt) or archive (.zip, .rar, .7z) to parse")],
    criteria_file: Annotated[UploadFile | None, File(description="Optional .ini criteria file for filtering")] = None,
    show_only_criteria: Annotated[bool, Form(description="If true, only show items matching criteria")] = False,
) -> TestLogParseResponse | TestLogParseResponseEnhanced | JSONResponse:
    """
    Parse a test log file or archive and extract test items.

    Args:
        file: Uploaded .txt file or archive
        criteria_file: Optional .ini criteria file for filtering
        show_only_criteria: If True, only return items matching criteria

    Returns:
        TestLogParseResponse (basic) or TestLogParseResponseEnhanced (with criteria)
        or JSONResponse (for archives)

    Raises:
        HTTPException 400: If file type is invalid
        HTTPException 500: If parsing fails
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    filename_lower = file.filename.lower()

    # Check if file is archive
    is_archive = filename_lower.endswith((".zip", ".rar", ".7z"))
    is_txt = filename_lower.endswith(".txt")

    if not is_archive and not is_txt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only .txt, .zip, .rar, and .7z files are accepted.")

    # Parse criteria file if provided
    criteria_rules = None
    if criteria_file:
        if not criteria_file.filename or not criteria_file.filename.lower().endswith(".ini"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Criteria file must be .ini format.")

        try:
            criteria_content = await criteria_file.read()
            criteria_rules = _parse_test_log_criteria_file(criteria_content)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse criteria file: {str(e)}") from e

    # Save uploaded file temporarily
    temp_file_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if is_archive:
            # Extract archive to get .txt files
            extracted_files = TestLogParser.extract_archive(str(temp_file_path))

            if len(extracted_files) == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No .txt files found in archive.")

            # Validate patterns in extracted files
            valid_files = []
            for txt_file in extracted_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if validate_test_log_pattern(content):
                            valid_files.append(txt_file)
                except Exception:
                    # Skip files that can't be read
                    continue
            
            if len(valid_files) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid test log files found in archive. Files must contain one of the required patterns: "
                           "'<< START TESTING >>' (repeated 4 times), "
                           "'=========[Start SFIS Test Result]======', or "
                           "'*** Test flow ***'"
                )

            # If criteria provided, use enhanced parsing on first valid file
            # (archive mode with criteria currently processes first file only)
            if criteria_rules:
                result = TestLogParser.parse_file_enhanced(valid_files[0], criteria_rules=criteria_rules, show_only_criteria=show_only_criteria)
                response = TestLogParseResponseEnhanced(**result)
                return response
            else:
                # Parse archive (returns all valid files)
                # Note: We need to filter the archive parser to only process valid files
                result = TestLogParser.parse_archive(str(temp_file_path))
                # Filter results to only include valid files
                if isinstance(result, dict) and 'files' in result:
                    valid_file_names = [Path(f).name for f in valid_files]
                    result['files'] = [f for f in result['files'] if f.get('filename') in valid_file_names]
                return JSONResponse(content=result)
        else:
            # Parse single .txt file - validate pattern first
            with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if not validate_test_log_pattern(content):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File does not contain valid test log format. Files must contain one of the required patterns: "
                               "'<< START TESTING >>' (repeated 4 times), "
                               "'=========[Start SFIS Test Result]======', or "
                               "'*** Test flow ***'"
                    )
            
            # Always use enhanced parsing (with or without criteria)
            result = TestLogParser.parse_file_enhanced(str(temp_file_path), criteria_rules=criteria_rules, show_only_criteria=show_only_criteria)
            response = TestLogParseResponseEnhanced(**result)
            return response

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to parse file: {str(e)}") from e

    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
        await file.close()
        if criteria_file:
            await criteria_file.close()


@router.post(
    "/compare",
    response_model=None,
    summary="Compare test logs (with optional criteria filtering and deviation analysis)",
    description="""
    Compare multiple test log files (.txt) or archives (.zip, .rar, .7z).

    **Enhanced with criteria support:**
    - Upload .ini criteria file for filtering and scoring
    - Per-ISN deviation from median baseline (or criteria target)
    - Separated value-type items (numeric) from non-value items
    - Aggregate statistics: avg deviation, avg score, median score

    **Response includes:**
    - File summaries with ISNs
    - Per-ISN measurements, deviations, and scores
    - Baseline values (median or criteria target)
    - Value vs non-value item separation
    """,
)
async def compare_test_logs(
    files: Annotated[list[UploadFile], File(description="Test log files (.txt) or archives (.zip, .rar, .7z) to compare")],
    criteria_file: Annotated[UploadFile | None, File(description="Optional .ini criteria file for filtering")] = None,
    show_only_criteria: Annotated[bool, Form(description="If true, only show items matching criteria")] = False,
) -> CompareResponse | CompareResponseEnhanced:
    """
    Compare test items across multiple test log files or archives.

    Args:
        files: List of uploaded files (minimum 1 archive or 2 .txt files)
        criteria_file: Optional .ini criteria file for filtering
        show_only_criteria: If True, only return items matching criteria

    Returns:
        CompareResponse (basic) or CompareResponseEnhanced (with criteria)

    Raises:
        HTTPException 400: If insufficient .txt files or invalid file type
        HTTPException 500: If comparison fails
    """
    if len(files) < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least 1 file is required.")

    # Validate all files
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File with no name provided.")
        filename_lower = file.filename.lower()
        if not filename_lower.endswith((".txt", ".zip", ".rar", ".7z")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type for {file.filename}. Only .txt, .zip, .rar, and .7z files are accepted.")

    # Parse criteria file if provided
    criteria_rules = None
    if criteria_file:
        if not criteria_file.filename or not criteria_file.filename.lower().endswith(".ini"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Criteria file must be .ini format.")

        try:
            criteria_content = await criteria_file.read()
            criteria_rules = _parse_test_log_criteria_file(criteria_content)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse criteria file: {str(e)}") from e

    temp_file_paths = []
    txt_file_paths = []

    try:
        # Save all uploaded files temporarily
        for file in files:
            temp_path = UPLOAD_DIR / file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(temp_path)

            # Check if archive
            if file.filename.lower().endswith((".zip", ".rar", ".7z")):
                # Extract archive and get .txt files
                extracted_files = TestLogParser.extract_archive(str(temp_path))
                
                # Validate patterns in extracted files
                for txt_file in extracted_files:
                    try:
                        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if validate_test_log_pattern(content):
                                txt_file_paths.append(txt_file)
                    except Exception:
                        # Skip files that can't be read
                        continue
            else:
                # Validate .txt file pattern before adding
                try:
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if validate_test_log_pattern(content):
                            txt_file_paths.append(str(temp_path))
                except Exception:
                    # Skip files that can't be read
                    pass

        if len(txt_file_paths) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"No valid test log files found for comparison. Files must contain one of the required patterns: "
                       "'<< START TESTING >>' (repeated 4 times), "
                       "'=========[Start SFIS Test Result]======', or "
                       "'*** Test flow ***'"
            )

        # Always use enhanced comparison for BY UPLOAD LOG feature
        result = TestLogParser.compare_files_enhanced(txt_file_paths, criteria_rules=criteria_rules, show_only_criteria=show_only_criteria)
        response = CompareResponseEnhanced(**result)

        return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to compare files: {str(e)}") from e

    finally:
        # Clean up temporary files
        for temp_path in temp_file_paths:
            path = Path(temp_path)
            if path.exists():
                path.unlink()

        for file in files:
            await file.close()

        if criteria_file:
            await criteria_file.close()


@router.get("/health", summary="Health check for test log parser", description="Check if the test log parsing service is operational.")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status message
    """
    return JSONResponse(content={"status": "healthy", "service": "Test Log Parser", "upload_dir": str(UPLOAD_DIR.absolute()), "upload_dir_exists": UPLOAD_DIR.exists()})
