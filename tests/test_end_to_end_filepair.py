from app.utils.format_compare import (
    augment_for_human,
    compare_maps,
    parse_mastercontrol_text,
    parse_wifi_dvt_text,
    write_human_xlsx,
)


def test_end_to_end_filepair(tmp_path):
    # create minimal MasterControlV2-like content
    # MasterControlV2-like output expects a header token such as 'TX1_...'
    mc2_text = "TX1_POW_2412_11B_CCK11_B20\n24.41\n"
    # create minimal DVT content with header 'Standard' present so parser picks it up
    dvt_text = "Antenna,Freq,Standard,DataRate,BandWidth,M. Power\n0,2412,11B,CCK11,B20,23.64\n"

    # parse
    master_map = parse_mastercontrol_text(mc2_text)
    dvt_map = parse_wifi_dvt_text(dvt_text)

    # compare and augment
    rows = compare_maps(master_map, dvt_map, spec=None)
    human = augment_for_human(rows, spec=None)

    # basic assertions about structure
    assert isinstance(human, list)
    assert len(human) == 1
    r = human[0]
    assert r["metric"] == "POW"
    # ensure human fields are present
    for k in (
        "usl",
        "lsl",
        "mc2_spec_diff",
        "dvt_spec_diff",
        "mc2_result",
        "dvt_result",
        "mc2_dvt_diff",
    ):
        assert k in r

    # also verify write_human_xlsx accepts a BytesIO-like object
    from io import BytesIO

    bio = BytesIO()
    write_human_xlsx(human, bio, provenance={"generated": "test"}, provenance_position="bottom")
    bio.seek(0)
    data = bio.read()
    assert data and len(data) > 10
