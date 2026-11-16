# SPDX-License-Identifier: MIT
"""Kimi-K3 TPOT investigation scripts on aiperf output cut from CI runs.

The fixtures hold, for CI Slurm 11899 (ROCm 10.1) and 12390 (ROCm 7.2.4),
the results JSON's ITL fields and the ITL rows of the timeslices CSV around
900 to 930 s of the profiling phase, plus slice 0's row. summarize.py and
breakdown.py run on synthetic kineto traces, times in us.
"""

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_engine.run_labels import build_run_label

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "kimi_k3_tpot"
sys.path.insert(0, str(SCRIPTS))
import breakdown
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


# summarize.py and breakdown.py on synthetic kineto traces, times in us.


def label(bs, running_bs=None, prefill=False, prefill_seqs=0, dummy=False):
    batch = None
    if prefill_seqs:
        batch = SimpleNamespace(
            total_tokens_num=bs,
            total_seqs_num_prefill=prefill_seqs,
            total_seqs_num_decode=bs - prefill_seqs,
            num_spec_step=0,
        )
    return build_run_label(
        is_prefill=prefill,
        use_cudagraph=not prefill,
        is_dummy=dummy,
        tbo_on=False,
        scheduled_bs=bs,
        batch=batch,
        running_bs=running_bs or bs,
    )


def event(cat, name, ts, dur, stream=3):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "tid": stream}


def rank_trace(mla_start):
    """Steps [0, 100) bs 4, [100, 200) bs 3 padded to 4, [400, 500) bs 4.

    The first window repeats on stream 7; a prefill and a decode holding a
    prefill run between the second and third steps, a dummy after the third.
    """
    windows = [
        (0, 80, label(4)),
        (100, 80, label(3, running_bs=4)),
        (200, 50, label(2, prefill=True)),
        (300, 50, label(4, prefill_seqs=1)),
        (400, 50, label(4)),
        (500, 50, label(4, dummy=True)),
    ]
    events = [event("gpu_user_annotation", n, ts, d) for ts, d, n in windows]
    events.append(event("gpu_user_annotation", label(4), 1, 78, stream=7))
    kernels = [
        ("gemm_a", 10, 40, 3),
        ("mla_decode", mla_start, 70 - mla_start, 7),
        ("ncclKernel_x", 90, 20, 3),  # crosses into the second step
        ("foo_unknown", 120, 20, 3),
        ("gemm_a", 210, 80, 3),  # prefill window
        ("moe_x", 310, 80, 3),  # decode window holding a prefill
        ("moe_x", 400, 50, 3),
        ("moe_x", 510, 10, 3),  # dummy window
    ]
    events += [event("kernel", n, ts, d, s) for n, ts, d, s in kernels]
    events.append({"ph": "M", "name": "process_name", "pid": 0, "tid": 0})
    return {"traceEvents": events}


@pytest.fixture
def summary(tmp_path):
    run = tmp_path / "run"
    (run / "rank_0").mkdir(parents=True)
    (run / "rank_1").mkdir()
    # rank 0's mla overlaps gemm_a over [30, 50); rank 1's does not.
    with gzip.open(run / "rank_0" / "a.pt.trace.json.gz", "wt") as f:
        json.dump(rank_trace(mla_start=30), f)
    (run / "rank_1" / "b.pt.trace.json").write_text(json.dumps(rank_trace(50)))
    subprocess.run(
        [sys.executable, SCRIPTS / "summarize.py", "case-x", itl.NEW, run]
        + ["--out", tmp_path / "out"],
        check=True,
    )
    return tmp_path / "out" / f"case-x.{itl.NEW}.trace.json"


def test_summary_steps_by_real_batch_size_with_overlap_shared(summary):
    (run,) = json.loads(summary.read_text())["runs"]
    assert run["ranks"] == 2
    bs = run["batch_sizes"]
    assert sorted(bs) == ["3", "4"]
    assert (bs["4"]["steps"], bs["3"]["steps"]) == (2, 1)
    assert bs["4"]["step_us"] == bs["3"]["step_us"] == 100
    # Rank 0 splits [30, 50) between gemm_a and mla: 30 + 40 over 4 steps.
    assert bs["4"]["kernels"] == {
        "gemm_a": 17.5,
        "mla_decode": 12.5,
        "ncclKernel_x": 5,
        "moe_x": 25,
    }
    assert bs["4"]["idle_us"] == 40
    assert bs["3"]["kernels"] == {"ncclKernel_x": 10, "foo_unknown": 20}
    for s in bs.values():
        assert sum(s["kernels"].values()) + s["idle_us"] == s["step_us"]


def test_breakdown_of_one_summary_as_both_images(summary):
    figures = json.loads(summary.read_text())
    figures["image"] = itl.BASE
    summary.with_name(f"case-x.{itl.BASE}.trace.json").write_text(json.dumps(figures))
    printed = subprocess.run(
        [sys.executable, SCRIPTS / "breakdown.py", summary.parent],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # Pooled weights 2:1 over bs 4 and 3.
    assert "unmatched kernels" in printed
    rows = {line[:20].strip(): line.split()[-3:] for line in printed.splitlines()}
    assert rows["GEMM"] == ["0.012", "0.012", "+0.000"]
    assert rows["unmatched kernels"][0] == "0.007"
    assert rows["step"][0] == "0.100"
    assert "traced gap: +0.000 ms" in printed
    assert "explained share: n/a" in printed
    assert "A1 gap" not in printed


def runs(*batch_sizes):
    return [{"batch_sizes": {bs: s for bs, s in batch_sizes}}]


def bs_summary(steps, step_us, idle_us, kernels):
    return {"steps": steps, "step_us": step_us, "idle_us": idle_us, "kernels": kernels}


def test_breakdown_weights_both_images_by_their_pooled_batch_sizes(tmp_path, out_dir):
    # NEW runs mostly bs 8, BASE mostly bs 4: pooled, 4 steps each, so each
    # size weighs a half. bs 16 has no BASE step and is left out.
    new = runs(
        ("4", bs_summary(1, 100, 40, {"mla_x": 60})),
        ("8", bs_summary(3, 200, 50, {"mla_x": 120, "weird": 30})),
        ("16", bs_summary(5, 900, 0, {"mla_x": 900})),
    )
    base = runs(
        ("4", bs_summary(3, 90, 35, {"mla_x": 55})),
        ("8", bs_summary(1, 190, 45, {"mla_x": 115, "weird": 30})),
    )
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    for image, r in ((itl.NEW, new), (itl.BASE, base)):
        (summaries / f"c.{image}.trace.json").write_text(
            json.dumps({"case": "c", "image": image, "runs": r})
        )
    subprocess.run(
        [
            sys.executable,
            SCRIPTS / "itl.py",
            out_dir,
            "--json",
            summaries / "c.itl.json",
        ],
        check=True,
        capture_output=True,
    )
    printed = subprocess.run(
        [sys.executable, SCRIPTS / "breakdown.py", summaries],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = {line[:20].strip(): line.split()[-3:] for line in printed.splitlines()}
    assert rows["MLA attention"] == ["0.090", "0.085", "+0.005"]
    assert rows["unmatched kernels"] == ["0.015", "0.015", "+0.000"]
    assert rows["idle"] == ["0.045", "0.040", "+0.005"]
    assert rows["step"] == ["0.150", "0.140", "+0.010"]
    assert "traced gap: +0.010 ms (+7.1%)" in printed
    assert "A1 gap, whole hour: +0.277 ms (+1.9%)" in printed
    assert "A1 gap, 900-930 s: +0.587 ms (+4.2%)" in printed
    assert "explained share: 50%" in printed


def test_kernels_take_their_first_matching_component():
    assert breakdown.component("aiter::cross_device_reduce_2stage<bf16>") == (
        "communication"
    )
    assert breakdown.component("gemm1_a4w4_port_fp4_moe") == "MoE"
    assert breakdown.component("aiter::f4gemm_bf16_per1x32Fp4") == "GEMM"
    assert breakdown.component("_replayssm_kda_fwd_kernel") == "KDA"
    assert breakdown.component("_attn_res_fused_kernel") == breakdown.UNMATCHED
