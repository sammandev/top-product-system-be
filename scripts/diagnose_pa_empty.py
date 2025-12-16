"""
Diagnostic script to understand why adjusted_power_trend is empty.

The /api/dut/pa/adjusted-power/ endpoint ALREADY auto-fills test items from srom_test_items.

Here's the flow:
1. Fetches DUT records for the ISN
2. For each station, gets nonvalue records containing PA SROM test items
3. Extracts PA_SROM_OLD and PA_SROM_NEW test items automatically
4. Calls external PA trend API with these test items
5. Pairs OLD/NEW items by base name
6. Calculates adjusted power

Why adjusted_power_trend might be EMPTY:
═══════════════════════════════════════════════════════════════════════

ISSUE 1: No PA SROM Test Items Found
─────────────────────────────────────────────────────────────────────
- Check: Do nonvalue records contain PA1/PA2/PA3/PA4_SROM_OLD/NEW items?
- Log: "No PA SROM test items found in any station for the provided DUT"
- Solution: Verify the DUT has PA test items in its test data

ISSUE 2: External PA Trend API Returns Empty
─────────────────────────────────────────────────────────────────────
- Check: Is the time window appropriate?
- Log: "No PA trend data returned for station X (empty response from external API)"
- Common causes:
  a) Time window doesn't overlap with test data
  b) External API has no historical data for these test items
  c) Test items names don't match exactly in the trend database
- Solution: 
  - Check the time window being used (logged in detail)
  - Verify test items exist in external trend API

ISSUE 3: OLD/NEW Pairing Fails
─────────────────────────────────────────────────────────────────────
- Check: Do we have matching OLD and NEW pairs?
- Log: "Incomplete pair for BASE_NAME: old=..., new=..."
- Log: "Station X: Found Y potential base name pairs"
- Log: "Station X: Completed 0 pairs, Z unpaired items"
- Common causes:
  a) Only OLD items exist, no NEW items (or vice versa)
  b) Base name extraction fails for some items
  c) Test item naming doesn't follow PA{1-4}_SROM_OLD/NEW pattern
- Solution: Check that both OLD and NEW variants exist for each test pattern

ISSUE 4: Time Window Too Narrow
─────────────────────────────────────────────────────────────────────
- Default: 24-hour window ending at latest test_date
- If test_date is very recent and trend API needs time to aggregate data
- Solution: Widen the time window (use start_time/end_time params)

ISSUE 5: Station Filtering
─────────────────────────────────────────────────────────────────────
- If station_identifiers parameter filters out all stations with PA items
- Solution: Remove station filter or adjust it

═══════════════════════════════════════════════════════════════════════

HOW TO DEBUG:
═══════════════════════════════════════════════════════════════════════

1. Check Backend Logs for These Key Messages:
   ─────────────────────────────────────────────────────────────────
   - "PA trend time window for DUT {isn}: start=..., end=..."
     → Shows the time window being used
   
   - "Fetching PA trend data for station {id} with {n} test items in time window..."
     → Shows how many test items were extracted
   
   - "PA trend data for station {id}: {n} items returned"
     → Shows if external API returned data
   
   - "Station {id}: Found {n} potential base name pairs"
     → Shows pairing attempts
   
   - "Station {id}: Completed {n} pairs, {m} unpaired items"
     → Shows successful pairs

2. Example Successful Flow:
   ─────────────────────────────────────────────────────────────────
   [INFO] PA trend time window for DUT ABC123: start=2025-11-23T10:00:00Z, end=2025-11-24T10:00:00Z
   [INFO] Fetching PA trend data for station 145 with 4 test items in time window ...
   [DEBUG] PA trend payload: {...}
   [INFO] PA trend data for station 145: 4 items returned
   [DEBUG] PA trend data keys: ['WiFi_PA1_SROM_OLD_...', 'WiFi_PA1_SROM_NEW_...', ...]
   [DEBUG] Station 145: Found 2 potential base name pairs
   [INFO] Station 145: Completed 2 pairs, 0 unpaired items

3. Test with Different Parameters:
   ─────────────────────────────────────────────────────────────────
   # Try widening the time window
   GET /api/dut/pa/adjusted-power/?dut_isn=ABC123&start_time=2025-11-01T00:00:00Z&end_time=2025-11-24T23:59:59Z
   
   # Try specific station
   GET /api/dut/pa/adjusted-power/?dut_isn=ABC123&station_identifiers=145

4. Verify Test Items in Database:
   ─────────────────────────────────────────────────────────────────
   - Check if the DUT's nonvalue records contain PA_SROM items
   - Verify naming follows: WiFi_PA{1-4}_SROM_{OLD|NEW}_<pattern>_{MID|MEAN}

═══════════════════════════════════════════════════════════════════════

CURRENT IMPLEMENTATION STATUS:
═══════════════════════════════════════════════════════════════════════

✅ AUTO-FILLS test_items from srom_test_items (Lines 6490-6499 in external_api_client.py)
✅ Comprehensive logging added for debugging
✅ Proper time window handling (default 24h, max 7 days)
✅ Station filtering support
✅ Automatic OLD/NEW pairing by base name

The endpoint DOES NOT need manual test_items parameter - it extracts them automatically!

If you're still seeing empty results, follow the debug steps above and check the logs.
"""

print(__doc__)
