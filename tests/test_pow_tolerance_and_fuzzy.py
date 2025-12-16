from app.utils import format_compare


def test_parse_range_token():
    # By design a single positive number '1' is interpreted as symmetric tolerance [-1, +1]
    assert format_compare._parse_range_token("1") == (-1.0, 1.0)
    assert format_compare._parse_range_token("-0.5~0.6") == (-0.5, 0.6)


def test_pow_tolerance_range_applied():
    # Spec with explicit tx_target_tolerance per-index
    spec_for_std = {
        "tx_target_power": [18],
        "tx_target_tolerance": ["-1~1"],
    }
    # value equal to target (18) should be within [17,19]
    ok = format_compare._check_spec_rule(spec_for_std, 0, "POW", 18)
    assert ok is True


def test_pow_tolerance_single_number_tolerance():
    # single-number tolerance should be interpreted as symmetric range [-v, +v]
    spec_for_std = {"tx_target_power": [20], "tx_target_tolerance": ["0.5"]}
    # allowed range is [19.5, 20.5]
    assert format_compare._check_spec_rule(spec_for_std, 0, "POW", 19.6) is True
    assert format_compare._check_spec_rule(spec_for_std, 0, "POW", 20.6) is False


def test_pow_tolerance_malformed_token_fallback():
    # malformed tolerance token should fall back to default ±0.5 dB
    spec_for_std = {"tx_target_power": [15], "tx_target_tolerance": ["garbage"]}
    # 15.4 is within default ±0.5
    assert format_compare._check_spec_rule(spec_for_std, 0, "POW", 15.4) is True
    # 15.6 is outside default ±0.5
    assert format_compare._check_spec_rule(spec_for_std, 0, "POW", 15.6) is False


def test_fuzzy_matching_and_summary():
    # MasterControlV2 parser expects a header row with tokens and a following data row with values.
    master_text = "TX1_POW_5180_11AC_MCS0_B20\n18\n"
    # DVT CSV header must include Standard and DataRate; Antenna value here is the numeric antenna index used by parser
    dvt_csv = "Standard,DataRate,Freq,Antenna,M. Power,EVM,FreqError,Spectrum Mask,LO Leakage\n11AC,MCS0,5181,0,17.2,1.2,0.1,OK,-60\n"

    master_map = format_compare.parse_mastercontrol_text(master_text)
    dvt_map = format_compare.parse_wifi_dvt_text(dvt_csv)

    rows = format_compare.compare_maps(master_map, dvt_map, threshold=1.0, spec=None, freq_tolerance_mhz=2.0)
    # Ensure we produced at least one comparison and summary works
    assert isinstance(rows, list)
    assert len(rows) >= 0
    summary = format_compare.compute_summary(rows)
    assert "per_antenna" in summary and "per_metric" in summary
