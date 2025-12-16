from app.utils.format_compare import augment_for_human, compare_maps


def test_pow_mc2_mpass_dvt_pass():
    # Construct a minimal master and dvt map for the POW metric where
    # target=24.5 and tolerance '-3~0' -> USL=24.5, LSL=21.5
    # MC2 value is 24.41 (within 0.5 dB of USL -> M.PASS)
    # DVT value is 23.64 (0.86 dB from USL -> PASS under 0.5 threshold)
    master_map = {(1, "POW", 2412, "11B", "CCK11", "B20"): 24.41}
    dvt_map = {(0, 2412, "11B", "CCK11", "B20"): {"POW": 23.64}}

    spec = {"11B": {"tx_target_power": [24.5], "tx_target_tolerance": ["-3~0"]}}

    rows = compare_maps(master_map, dvt_map, spec=spec)
    human = augment_for_human(rows, spec=spec)
    assert len(human) == 1
    r = human[0]
    assert r["metric"] == "POW"
    assert r["mc2_result"] == "M.PASS"
    assert r["dvt_result"] == "PASS"
