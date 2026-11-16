# SPDX-License-Identifier: MIT
"""Kimi-K3 TPOT investigation scripts on aiperf output cut from CI runs.

The fixtures hold, for CI Slurm 11899 (ROCm 10.1) and 12390 (ROCm 7.2.4),
the results JSON's ITL fields and the ITL rows of the timeslices CSV around
900 to 930 s of the profiling phase, plus slice 0's row.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "kimi_k3_tpot"
sys.path.insert(0, str(SCRIPTS))
import itl

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kimi_k3_tpot"


@pytest.fixture
def out_dir(tmp_path):
    for run in FIXTURES.iterdir():
        shutil.copytree(run, tmp_path / run.name)
    return tmp_path


def add_run(out_dir, src, dst, mean_itl_ms):
    shutil.copytree(out_dir / src, out_dir / dst)
    (results,) = (out_dir / dst / "benchmark_results").glob("pd-atom-*.json")
    figures = json.loads(results.read_text())
    figures["mean_itl_ms"] = mean_itl_ms
    results.write_text(json.dumps(figures))


def test_ci_runs_give_the_ci_table_gap_and_window_means(out_dir):
    out = out_dir / "itl.json"
    printed = subprocess.run(
        [sys.executable, SCRIPTS / "itl.py", out_dir, "--json", out],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "ROCm_10.1.0: mean ITL 14.95 ms (min-max 14.95-14.95, 1 runs)" in printed
    assert "nightly_202611021458: mean ITL 14.68 ms" in printed
    assert "median ITL 15.24 ms" in printed
    assert "+0.28 ms (+1.9%)" in printed
    figures = json.loads(out.read_text())
    new, base = figures["ROCm_10.1.0"]["r1"], figures["nightly_202611021458"]["r1"]
    assert new["window_mean_itl_ms"] == pytest.approx(14.5696, abs=1e-4)
    assert base["window_mean_itl_ms"] == pytest.approx(13.9822, abs=1e-4)
    assert new["median_itl_ms"] == pytest.approx(15.2424, abs=1e-4)


def test_window_averages_only_the_slices_aiperf_wrote(out_dir):
    # 11899 has no slice at 903, 908 or 911-913 s: 23 slices in [900, 930),
    # not 30. The slices at 899 and 930 s, other metrics and other stats
    # stay out.
    (csv,) = (out_dir / "ROCm_10.1.0-r1").glob("*/*/*timeslices.csv")
    rows = [line.split(",") for line in csv.read_text().splitlines()[1:]]
    start0 = int(rows[0][1])
    expected = [
        float(r[6])
        for r in rows
        if r[3] == "Inter Token Latency"
        and r[5] == "avg"
        and 900 <= (int(r[1]) - start0) / 1e9 < 930
    ]
    assert len(expected) == 23
    assert itl.window_mean(csv) == pytest.approx(sum(expected) / 23)


def test_gap_and_min_max_over_runs(out_dir):
    add_run(out_dir, "ROCm_10.1.0-r1", "ROCm_10.1.0-r2", 15.25)
    add_run(out_dir, "nightly_202611021458-r1", "nightly_202611021458-r2", 14.38)
    images = itl.collect(out_dir)
    assert sorted(images["ROCm_10.1.0"]) == ["r1", "r2"]
    printed = itl.report(images)
    assert "ROCm_10.1.0: mean ITL 15.10 ms (min-max 14.95-15.25, 2 runs)" in printed
    assert "nightly_202611021458: mean ITL 14.53 ms (min-max 14.38-14.68" in printed
    assert "+0.57 ms (+3.9%)" in printed


def test_run_missing_its_results_is_a_named_error(out_dir):
    (results,) = (out_dir / "nightly_202611021458-r1").glob("*/pd-atom-*.json")
    results.unlink()
    with pytest.raises(itl.MissingResults, match="pd-atom"):
        itl.collect(out_dir)
    failed = subprocess.run(
        [sys.executable, SCRIPTS / "itl.py", out_dir],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "MissingResults" in failed.stderr


def test_image_without_runs_is_a_named_error(out_dir):
    shutil.rmtree(out_dir / "ROCm_10.1.0-r1")
    with pytest.raises(itl.MissingResults, match="ROCm_10.1.0"):
        itl.collect(out_dir)
