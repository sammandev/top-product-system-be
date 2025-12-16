from app.utils.format_compare import (
    augment_for_human,
    format_minimal_number,
)


def test_format_minimal_number_examples():
    assert format_minimal_number(None) == "N/A"
    assert format_minimal_number(0) == "0"
    assert format_minimal_number(-5) == "-5"
    assert format_minimal_number(21.5) == "21.5"
    assert format_minimal_number(21.5000) == "21.5"
    assert format_minimal_number(21.000) == "21"


def _spec_for_pow():
    # minimal spec with one frequency entry and target + tolerance
    return {
        "11AC": {
            "frequency": [2412],
            "bw": ["B20"],
            "mod": ["CCK11"],
            "tx_target_power": [20],
            "tx_target_tolerance": ["-0.5~0.5"],
        }
    }


def test_augment_for_human_pow_display_and_margin():
    spec = _spec_for_pow()
    # create a comparison row where mc2_value is 19.6 (within 0.4 dB of target 20)
    rows = [
        {
            "antenna_dvt": 0,
            "antenna_mc2": 1,
            "metric": "POW",
            "freq": 2412,
            "standard": "11AC",
            "datarate": "CCK11",
            "bandwidth": "B20",
            "mc2_value": 19.6,
            "dvt_value": 19.4,
            "diff": 0.2,
            "flag": False,
            "spec_applied": True,
        }
    ]

    out = augment_for_human(rows, spec=spec)
    assert len(out) == 1
    r = out[0]
    # USL/LSL relative to target 20 with tolerance [-0.5,0.5] -> usl = 20+0.5=20.5, lsl = 20-0.5=19.5
    assert r["usl"] == "20.5"
    assert r["lsl"] == "19.5"
    # mc2_spec_diff should show value - target => 19.6 - 20 = -0.4 represented minimally
    assert r["mc2_spec_diff"] == "-0.4"
    assert r["dvt_spec_diff"] == "-0.6"
    # mc2 is 0.1 away from lsl (19.6 - 19.5 = 0.1) -> margin within 1 -> M.PASS
    assert r["mc2_result"] == "M.PASS"
    # dvt is 0.1 below lsl (19.4 < 19.5) -> out-of-spec but within 1 -> M.FAIL
    assert r["dvt_result"] == "M.FAIL"


def test_augment_for_human_evm_and_freq():
    # build spec where EVM USL is -30 and FREQ range is -10~10
    spec = {
        "11B": {
            "frequency": [5180],
            "bw": ["B20"],
            "mod": ["OFDM6"],
            "tx_evm_limit": [-30],
            "tx_freq_err_limit": ["-10~10"],
        }
    }
    rows = [
        {
            "antenna_dvt": 1,
            "antenna_mc2": 2,
            "metric": "EVM",
            "freq": 5180,
            "standard": "11B",
            "datarate": "OFDM6",
            "bandwidth": "B20",
            "mc2_value": -29.5,
            "dvt_value": -30.5,
            "diff": 1.0,
            "flag": False,
            "spec_applied": True,
        },
        {
            "antenna_dvt": 1,
            "antenna_mc2": 2,
            "metric": "FREQ",
            "freq": 5180,
            "standard": "11B",
            "datarate": "OFDM6",
            "bandwidth": "B20",
            "mc2_value": 2.0,
            "dvt_value": -3.0,
            "diff": 5.0,
            "flag": False,
            "spec_applied": True,
        },
    ]

    out = augment_for_human(rows, spec=spec)
    # two outputs expected
    assert len(out) == 2
    evm_row = next(r for r in out if r["metric"] == "EVM")
    freq_row = next(r for r in out if r["metric"] == "FREQ")
    # EVM usl should display '-30'
    assert evm_row["usl"] == "-30"
    # mc2_spec_diff: mc2(-29.5) - usl(-30) = 0.5 -> minimal '0.5'
    assert evm_row["mc2_spec_diff"] == "0.5"
    # dvt_spec_diff: -30.5 - -30 = -0.5 -> '-0.5'
    assert evm_row["dvt_spec_diff"] == "-0.5"

    # FREQ should render pair diffs; ensure usl/lsl are set and formatted minimally
    assert freq_row["usl"] == "10"
    assert freq_row["lsl"] == "-10"
    # pair string contains ' | '
    assert " | " in freq_row["mc2_spec_diff"]
