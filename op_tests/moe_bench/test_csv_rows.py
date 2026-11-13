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


def test_rows_of_an_earlier_head_are_mapped_by_that_head_not_rejected(tmp_path):
    """A file made on one head and resumed on others (columns added between them) holds
    rows of several widths: each is mapped by the column list of the head that wrote it, so
    nothing is read under another head's header (a shifted harness_sha) or dropped."""
    p = str(tmp_path / "bar.csv")
    h1 = ["case_id", "bar_us", "new_us", "harness_sha"]  # first head
    h2 = [
        "case_id",
        "bar_us",
        "bar_incomplete",
        "new_us",
        "harness_sha",
    ]  # one column added
    cur = [
        "case_id",
        "bar_us",
        "dispatch_rounds_more",
        "bar_incomplete",
        "new_us",
        "harness_sha",
    ]
    out = schema.Appender(p, h1, False)
    out.add_row(
        {"case_id": "a", "bar_us": "1", "new_us": "5.5", "harness_sha": "aaaaaaaaaaaa"}
    )
    out.close()
    with open(
        p, "a", newline=""
    ) as f:  # the second head appended its 5-wide row, no migration
        csv.writer(f).writerow(["b", "2", "inc", "6.5", "bbbbbbbbbbbb"])
    assert schema.shape_problems(
        p
    )  # the old reader sees a 5-wide row under a 4-wide header
    out = schema.Appender(p, cur, True, history=[h1, h2])
    out.add_row({"case_id": "c", "bar_us": "3", "harness_sha": "cccccccccccc"})
    out.close()
    rows = {r["case_id"]: r for r in schema.read_csv(p, strict=True)}
    assert rows["a"]["new_us"] == "5.5" and rows["a"]["harness_sha"] == "aaaaaaaaaaaa"
    assert rows["b"]["bar_incomplete"] == "inc" and rows["b"]["new_us"] == "6.5"
    assert (
        rows["b"]["harness_sha"] == "bbbbbbbbbbbb"
        and rows["b"]["dispatch_rounds_more"] == ""
    )
    assert rows["c"]["harness_sha"] == "cccccccccccc" and len(rows) == 3
    assert not os.path.exists(p + ".rejected")
    # without the history the 5-wide row is moved out, not misread
    p2 = str(tmp_path / "bar2.csv")
    out = schema.Appender(p2, h1, False)
    out.close()
    with open(p2, "a", newline="") as f:
        csv.writer(f).writerow(["b", "2", "inc", "6.5", "bbbbbbbbbbbb"])
    schema.Appender(p2, cur, True).close()
    assert os.path.exists(p2 + ".rejected") and schema.read_csv(p2, strict=True) == []


def test_the_history_covers_the_widths_the_smci355_file_was_written_with():
    import json

    from op_tests.moe_bench import bar_runner

    widths = {len(h) for h in bar_runner._history("bar")}
    assert {102, 107, 108, 117} <= widths
    assert all(
        h[:37] == bar_runner._history("bar")[0][:37]
        for h in bar_runner._history("bar")[:2]
    )
    with open(
        os.path.join(os.path.dirname(bar_runner.__file__), "col_history.json")
    ) as f:
        assert json.load(f)["cand"]


def _row(header, **vals):
    base = {c: "" for c in header}
    base.update(vals)
    return [base[c] for c in header]


PROV = {
    "gfx": "gfx950", "cu_num": "256", "rocm": "7.2.53211", "torch": "2.13.0+rocm7.2",
    "flydsl": "0.5.0", "aiter_sha": "27faf0a57b4e", "aiter_dirty": "0",
    "harness_sha": "50c0d1b74f94", "date": "2026-11-12T18:00:00Z",
}  # fmt: skip


def test_repair_maps_a_row_by_the_header_whose_values_fit_and_rejects_the_rest(
    tmp_path,
):
    """The smci355 tier-2 file: header of 102 columns, 107-wide rows from a later head cut
    to 102 fields (so harness_sha held the ROCm version), clean 102-wide rows, and a row no
    header fits."""
    import json

    from op_tests.moe_bench import bar_runner

    hist = {len(h): h for h in bar_runner._history("bar")}
    h102, h107 = hist[102], hist[107]
    p = str(tmp_path / "bar.csv")
    clean = _row(h102, case_id="c1", mode="warm", status="ok", bar_us="12.5", dispatch_us="14.0",
                 dispatch_path_rel_l2_mean="0.01", **PROV)  # fmt: skip
    wide = _row(h107, case_id="c2", mode="warm", status="ok", bar_us="20.5", dispatch_us="21.0",
                bar_incomplete="fused_moe enumerated zero", dispatch_path_rel_l2_mean="0.02", **PROV)  # fmt: skip
    cut = wide[:102]  # what a rewrite against the 102-column header kept
    garbage = ["c9", "warm"] + ["x"] * 100  # right width, nothing plausible
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows([h102, clean, cut, garbage])
    assert schema.row_problems(
        p
    )  # the file as it is: the cut row holds the rocm version in harness_sha
    kept = bar_runner.repair_bar_csv(p)
    assert kept == 2
    rows = {r["case_id"]: r for r in schema.read_csv(p, strict=True)}
    assert (
        rows["c1"]["harness_sha"] == "50c0d1b74f94" and rows["c1"]["bar_us"] == "12.5"
    )
    # the cut row is read under the 107-column header: its columns land where they belong,
    # the cut tail (aiter_sha .. date) is empty, not another column's value
    c2 = rows["c2"]
    assert (
        c2["bar_incomplete"] == "fused_moe enumerated zero"
        and c2["rocm"] == "7.2.53211"
    )
    assert c2["dispatch_path_rel_l2_mean"] == "0.02" and c2["bar_us"] == "20.5"
    assert c2["harness_sha"] == "" and c2["date"] == ""
    with open(p + ".rejected", newline="") as f:
        rej = list(csv.reader(f))
    assert rej[0] == h102 and [r[0] for r in rej[1:]] == ["c9"]
    assert schema.row_problems(p) == []
    assert json.dumps(sorted(rows)) == '["c1", "c2"]'


def test_a_row_two_headers_fit_equally_is_rejected_not_guessed():
    a = ["case_id", "x_us", "tail"]
    b = ["case_id", "x_us", "other"]
    assert schema.map_row(["c", "1.5", "t"], [a, b]) is None
    assert schema.map_row(["c", "1.5", "t"], [a, a]) == (a, False)
    assert (
        schema.map_row(["c", "oops", "t"], [a]) is None
    )  # a number column holding text


def test_row_problems_flags_a_shifted_or_short_row_in_a_file_with_one_header(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "case_id,dispatch_path_rel_l2_mean,harness_sha\r\nc1,0.5,abcdef012345\r\nc2,7.2.53211,x\r\nc3,0.5\r\n"
    )
    got = schema.row_problems(str(p))
    assert [(n, why.split(" ")[0]) for n, _, why in got] == [
        (2, "implausible"),
        (3, "2"),
    ]
