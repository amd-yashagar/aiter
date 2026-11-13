# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every result row is one well-formed CSV record: a field with quotes, commas, newlines or
JSON survives a round trip, a resume repairs what a kill or a column change left, and a
reader flags a row whose field count is wrong."""

import csv
import json
import os
import subprocess
import sys

import pytest

from op_tests.moe_bench import bar_runner, runner, schema

NASTY = [
    'impl "x", cold\nsecond line',
    json.dumps({"a": [1, 2], "q": 'say "hi", ok', "n": "line\nbreak"}),
    'trailing quote"',
    "a,b,c",
    "\r\nwindows",
    "",
    "plain",
]


def row_for(cols, vals):
    return {c: vals[i % len(vals)] for i, c in enumerate(cols)}


def test_a_field_with_quotes_commas_newlines_and_json_round_trips(tmp_path):
    p = str(tmp_path / "t.csv")
    cols = ["case_id", "mode", "detail", "numerics", "kernels", "x"]
    out = schema.Appender(p, cols, resume=False)
    rows = [row_for(cols, NASTY[i:] + NASTY[:i]) for i in range(len(NASTY))]
    for r in rows:
        out.add_row(r)
    out.close()
    assert schema.shape_problems(p) == []
    back = schema.read_csv(p, strict=True)
    assert back == [{c: r[c] for c in cols} for r in rows]
    with open(p, newline="") as f:  # a plain csv.reader agrees: one record per row
        assert all(len(r) == len(cols) for r in csv.reader(f))


def test_bar_and_result_tables_write_through_the_appender(tmp_path):
    t = bar_runner.Table(str(tmp_path / "bar.csv"), bar_runner.BAR_COLS, False)
    r = {c: NASTY[i % len(NASTY)] for i, c in enumerate(bar_runner.BAR_COLS)}
    r["case_id"], r["mode"] = "tier1-00088", "cold"
    t.add(r)
    t.close()
    assert schema.shape_problems(str(tmp_path / "bar.csv")) == []
    got = schema.read_csv(str(tmp_path / "bar.csv"), strict=True)
    assert len(got) == 1 and got[0]["case_id"] == "tier1-00088"
    assert got[0]["detail"] == r["detail"]
    res = runner.Results(str(tmp_path / "res.csv"), False)
    rr = {c: NASTY[i % len(NASTY)] for i, c in enumerate(runner.RESULT_COLS)}
    rr["case_id"], rr["mode"] = "tier1-00088", "cold"
    res.add(rr)
    res.close()
    got = schema.read_csv(str(tmp_path / "res.csv"), strict=True)
    assert got[0]["detail"] == rr["detail"] and ("tier1-00088", "cold") in res.done


def test_the_reader_flags_a_row_with_the_wrong_column_count(tmp_path, capsys):
    p = tmp_path / "bad.csv"
    p.write_text('a,b,c\r\n1,2,3\r\n4,5\r\n6,"7,8",9\r\n10,11,12,13\r\n')
    assert schema.shape_problems(str(p)) == [(2, 3, 2), (4, 5, 4)]
    with pytest.raises(schema.CsvShapeError, match="2 rows"):
        schema.read_csv(str(p), strict=True)
    rows = schema.read_csv(str(p))  # non-strict: still returned, and said on stderr
    assert len(rows) == 4 and "WARNING" in capsys.readouterr().err
    r = subprocess.run(
        [sys.executable, "-m", "op_tests.moe_bench.schema", str(p)],
        capture_output=True, text=True, check=False,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )  # fmt: skip
    assert r.returncode == 1 and "malformed rows" in r.stdout


def test_a_multiline_quoted_field_is_not_a_malformed_row(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text('a,b\r\n1,"x\r\ny, ""z"""\r\n2,3\r\n')
    assert schema.shape_problems(str(p)) == []
    assert schema.read_csv(str(p), strict=True)[0]["b"] == 'x\r\ny, "z"'


def test_a_resume_after_a_kill_mid_row_drops_the_half_row_not_the_next_one(tmp_path):
    p = str(tmp_path / "t.csv")
    cols = ["case_id", "mode", "detail", "x"]
    out = schema.Appender(p, cols, False)
    out.add_row(row_for(cols, ["c1", "warm", "d", "x"]))
    out.close()
    with open(p, "a", newline="") as f:  # killed inside the second row
        f.write('c2,cold,"half, a quo')
    out = schema.Appender(p, cols, True)
    out.add_row(row_for(cols, ["c3", "cold", "d,e", "x"]))
    out.close()
    rows = schema.read_csv(p, strict=True)
    assert [r["case_id"] for r in rows] == ["c1", "c3"]
    assert rows[1]["detail"] == "d,e"
    with open(p + ".rejected", newline="") as f:
        assert len(list(csv.reader(f))) == 1


def test_a_resume_after_a_column_was_added_keeps_every_row_in_its_own_column(tmp_path):
    p = str(tmp_path / "t.csv")
    old = ["case_id", "mode", "detail"]
    out = schema.Appender(p, old, False)
    out.add_row({"case_id": "c1", "mode": "warm", "detail": 'a "q"'})
    out.close()
    new = ["case_id", "mode", "max_abs", "detail"]  # a column inserted mid-list
    out = schema.Appender(p, new, True)
    out.add_row({"case_id": "c2", "mode": "cold", "max_abs": "0.5", "detail": "b"})
    out.close()
    rows = schema.read_csv(p, strict=True)
    assert rows[0] == {
        "case_id": "c1",
        "mode": "warm",
        "max_abs": "",
        "detail": 'a "q"',
    }
    assert rows[1] == {"case_id": "c2", "mode": "cold", "max_abs": "0.5", "detail": "b"}


def test_two_writers_appending_to_one_file_never_interleave_inside_a_row(tmp_path):
    p = str(tmp_path / "t.csv")
    cols = ["case_id", "mode", "detail"]
    schema.Appender(p, cols, False).close()
    code = (
        "import sys\n"
        "from op_tests.moe_bench import schema\n"
        "a = schema.Appender(sys.argv[1], ['case_id', 'mode', 'detail'], True)\n"
        "for i in range(300):\n"
        "    a.add_row({'case_id': sys.argv[2] + str(i), 'mode': 'warm', 'detail': (sys.argv[2] * 2000) + '\"\\n,'})\n"
    )
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ps = [
        subprocess.Popen([sys.executable, "-c", code, p, w], cwd=here, env=dict(os.environ, PYTHONPATH=here))
        for w in "AB"
    ]  # fmt: skip
    assert all(x.wait(timeout=120) == 0 for x in ps)
    assert schema.shape_problems(p) == []
    rows = schema.read_csv(p, strict=True)
    assert len(rows) == 600
    assert all(set(r["detail"][:-3]) == {r["case_id"][0]} for r in rows)
