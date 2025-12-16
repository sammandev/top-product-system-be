from app.utils.format_compare import (
    augment_for_human,
    compare_maps,
    parse_mastercontrol_text,
    parse_wifi_dvt_text,
)


def test_mc2_dvt_result_parity_pow(tmp_path):
    # create a tiny spec where target=10 and tolerance +/-0.5 for one index
    spec = {
        "11B": {
            "frequency": [2412],
            "bw": ["B20"],
            "mod": ["CCK11"],
            "tx_target_power": [10],
            "tx_target_tolerance": ["-0.5~0.5"],
        }
    }

    # MasterControl: antenna 1 POW 2412 value 10 (mc2)
    # header line then value line
    master_text = "TX1_POW_2412_11B_CCK11_B20\n10\n"
    # DVT: antenna 0 POW 2412 value 10 (dvt)
    dvt_text = "Standard,Antenna,Freq,DataRate,BandWidth,M. Power\n11B,0,2412,CCK11,B20,10\n"

    master_map = parse_mastercontrol_text(master_text)
    dvt_map = parse_wifi_dvt_text(dvt_text)
    rows = compare_maps(master_map, dvt_map, spec=spec)
    human = augment_for_human(rows, spec=spec)
    assert len(human) == 1
    r = human[0]
    # both mc2_result and dvt_result should be identical and acceptable (PASS or M.PASS)
    assert r["mc2_result"] == r["dvt_result"]
    assert r["mc2_result"] in ("PASS", "M.PASS")
