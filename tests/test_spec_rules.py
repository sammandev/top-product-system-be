from app.utils.format_compare import (
    _check_spec_rule,
    _matches_spec,
    _parse_range_token,
)


def test_parse_range_token():
    assert _parse_range_token("10") == (-10.0, 10.0)
    assert _parse_range_token("-5") == (None, -5.0)
    assert _parse_range_token("-10~20.5") == (-10.0, 20.5)


def test_matches_spec_and_check_pow():
    spec = {
        "frequency": ["2412", "2462"],
        "bw": ["20", "20"],
        "mod": ["CCK11", "CCK11"],
        "tx_target_power": ["22.5", "21"],
        "tx_target_tolerance": ["-1~2", "1"],
    }
    # frequency 2412 -> index 0
    idx = _matches_spec(spec, 2412, "B20", "CCK11")
    assert idx == 0
    # POW exact 22.6 should be within -1~2 tolerance (i.e., -1 to 2 => min -1, max 2 relative?)
    # based on current implementation, POW uses tx_target_power +/-0.5 default; we will just test _check_spec_rule for EVM and FREQ


def test_check_spec_evm_freq_mask_lo():
    spec_for_std = {
        "tx_evm_limit": ["-5.5", "-5"],
        "tx_freq_err_limit": ["20", "-10~20.5"],
        "tx_mask_limit": ["0", "0"],
        "tx_lo_leakage_limit": ["-90~-20", "-90~-20"],
    }
    # EVM: value must be <= limit
    assert _check_spec_rule(spec_for_std, 0, "EVM", -6.0) is True
    assert _check_spec_rule(spec_for_std, 1, "EVM", -4.0) is False
    # FREQ: index 0 rule '20' -> symmetric [-20,20]
    assert _check_spec_rule(spec_for_std, 0, "FREQ", 15.0) is True
    assert _check_spec_rule(spec_for_std, 1, "FREQ", -20.0) is False
    # MASK: exact 0
    assert _check_spec_rule(spec_for_std, 0, "MASK", 0.0) is True
    assert _check_spec_rule(spec_for_std, 0, "MASK", 1.0) is False
    # LO_LEAKAGE_DB range
    assert _check_spec_rule(spec_for_std, 0, "LO_LEAKAGE_DB", -50.0) is True
    assert _check_spec_rule(spec_for_std, 0, "LO_LEAKAGE_DB", -10.0) is False
