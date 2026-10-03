"""tools/thor/hc_check_summary.py with thor_hc2 (THOR-HC-FUSE-2): the warp-specialized kernels' byte flags and times
per prompt and M, in a second table after the first."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hc_check_summary as hs  # noqa: E402

BASE = {"pass": 0, "layer": 0, "hc": "attn", "mode": 0, "R": 4096, "equal_inject": True, "equal_mixed": True,
        "equal_act": True}
ROWS = [
    {**BASE, "prompt": "nat32k", "M": 4096, "equal_act2": True, "equal_mixed2": True, "ms_r25": 4.0, "ms_new": 2.0,
     "ms_new2": 1.25, "ms_new2_down_act": 0.5, "ms_new2_upmix": 0.625},
    {**BASE, "prompt": "nat32k", "M": 4096, "equal_act2": True, "equal_mixed2": False, "ms_r25": 4.5, "ms_new": 2.5,
     "ms_new2": 1.75, "ms_new2_down_act": 0.75, "ms_new2_upmix": 0.875},
    {**BASE, "prompt": "nat32k", "M": 4096, "equal_act2": None, "ms_r25": 5.0, "ms_new": 3.0, "ms_new2": "Err('x')"},
    {**BASE, "prompt": "nat32k", "M": 128, "equal_act2": True, "equal_mixed2": True},
    {**BASE, "prompt": "syn128k", "M": 4096, "ms_r25": 4.0, "ms_new": 2.0},
]


def test_summarize_adds_the_second_paths_flags_and_speedup():
    s = hs.summarize(ROWS)
    a = s["nat32k"]["4096"]
    assert a["equal_act2"] is True and a["equal_mixed2"] is False        # None ignored; any False is False
    assert a["ms_new2"] == 1.5 and a["ms_new2_down_act"] == 0.625 and a["ms_new2_upmix"] == 0.75    # medians
    assert a["speedup2"] == a["ms_new"] / a["ms_new2"]                   # median ms_new / median ms_new2
    assert s["nat32k"]["128"]["equal_mixed2"] is True and "speedup2" not in s["nat32k"]["128"]
    c = s["syn128k"]["4096"]
    assert c["equal_act2"] is None and c["equal_mixed2"] is None and "speedup2" not in c


def test_the_second_table_lists_only_rows_with_the_second_path():
    s = hs.summarize(ROWS)
    text = hs.markdown2(s)
    lines = text.strip().splitlines()
    assert lines[0] == ("| prompt | M | calls | act2 equal | mixed2 equal | new ms | new2 ms | new2 down ms "
                        "| new2 upmix ms | speedup2 |")
    assert lines[1] == "|" + "---|" * 10
    assert lines[2] == f"| nat32k | 4096 | 3 | yes | no | 2.500 | 1.500 | 0.625 | 0.750 | {2.5 / 1.5:.2f} |"
    assert lines[3] == "| nat32k | 128 | 1 | yes | yes | - | - | - | - | - |"
    assert len(lines) == 4                                                # syn128k has no thor_hc2 data
    assert hs.markdown2(hs.summarize([ROWS[-1]])) == ""


def test_main_appends_the_second_table_and_records_fuse2(tmp_path, capsys):
    src = tmp_path / "hc_check.json"
    src.write_text(json.dumps({"upmix": True, "fuse2": True, "rows": ROWS}))
    out, md = tmp_path / "s.json", tmp_path / "s.md"
    assert hs.main([str(src), "--json", str(out), "--markdown", str(md)]) == 0
    got = json.loads(out.read_text())
    assert got["upmix"] is True and got["fuse2"] is True
    assert got["summary"]["nat32k"]["4096"]["equal_mixed2"] is False
    text = md.read_text()
    first = hs.markdown(hs.summarize(ROWS))
    assert text == first + "\n" + hs.markdown2(hs.summarize(ROWS))
    assert text in capsys.readouterr().out
    plain = tmp_path / "p.json"
    plain.write_text(json.dumps({"upmix": False, "rows": [ROWS[-1]]}))
    assert hs.main([str(plain), "--markdown", str(md)]) == 0
    assert md.read_text() == hs.markdown(hs.summarize([ROWS[-1]]))      # no second table without thor_hc2 data
