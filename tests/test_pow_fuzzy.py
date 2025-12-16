from app.utils.format_compare import (
    _datarate_similarity,
    compare_maps,
    parse_mastercontrol_text,
    parse_wifi_dvt_text,
)


def test_freq_tolerance_and_datarate_similarity():
    # create a simple master_map and dvt_map with slightly different freq and datarate naming
    master_text = "TX1_POW_2412_11B_CCK11_B20,TX1_EVM_2412_11B_CCK11_B20\n24.4,-6.0\n"
    dvt_text = "Standard,DataRate,Freq,Ant, M. Power,EVM\n11B,CCK11,2413,0,24.0,-5.9\n"
    master_map = parse_mastercontrol_text(master_text)
    dvt_map = parse_wifi_dvt_text(dvt_text)
    # with default freq tolerance 2 MHz, compare_maps should match 2412 with 2413
    rows = compare_maps(master_map, dvt_map, threshold=None, spec=None, freq_tolerance_mhz=2.0)
    assert any(r["metric"] == "POW" for r in rows)
    # datarate similarity test
    assert _datarate_similarity("CCK11", "CCK11") > 0.9


def test_pow_tolerance_range():
    spec = {
        "frequency": ["2412"],
        "bw": ["20"],
        "mod": ["CCK11"],
        "tx_target_power": ["22.5"],
        "tx_target_tolerance": ["-1~2"],
    }
    # master has 22.6, dvt has 22.0 -> diff 0.6 -> within -1~2 tolerance
    master_text = "TX1_POW_2412_11B_CCK11_B20\n22.6\n"
    dvt_text = "Standard,DataRate,Freq,Ant, M. Power\n11B,CCK11,2412,0,22.0\n"
    master_map = parse_mastercontrol_text(master_text)
    dvt_map = parse_wifi_dvt_text(dvt_text)
    rows = compare_maps(master_map, dvt_map, threshold=None, spec={"11B": spec}, freq_tolerance_mhz=1.0)
    # find POW row and check spec_applied true and flag false
    pow_rows = [r for r in rows if r["metric"] == "POW"]
    assert pow_rows, "POW row not found"
    assert pow_rows[0]["spec_applied"] is True
    assert pow_rows[0]["flag"] is False
