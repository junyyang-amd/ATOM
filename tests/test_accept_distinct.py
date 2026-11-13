"""The red/blue distinction check (tests/accept/atomesh_mm/checks/distinct.py)
against fake replies and a fake ATOMesh log: its verdict on the answers, the
routing key hashes and the cached prompt tokens."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "accept" / "atomesh_mm" / "checks"))
import distinct  # noqa: E402


def run(tmp_path, monkeypatch, cached, hashes=None):
    """Run the check with each request's (color, cached tokens) reply from
    `cached`, keyed by request name, logging one decision event per request."""
    for name, value in {
        "ACCEPT_MODEL": "k31",
        "ACCEPT_TOPOLOGY": "single",
        "ACCEPT_CHECK": "distinct",
        "RUN_DIR": str(tmp_path),
        "RESULTS_DIR": str(tmp_path / "results"),
        "LIMIT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    hashes = hashes or {"red": "aa", "blue": "bb"}
    log = tmp_path / "mesh.log"
    log.write_text("")
    names = iter(name for name, _ in distinct.REQUESTS)

    def ask(color):
        name = next(names)
        with log.open("a") as f:
            f.write(f"routing decision key_hash={hashes[color]} policy=cache_aware\n")
        return f"{color.title()}.", cached[name]

    monkeypatch.setattr(distinct, "ask", ask)
    with pytest.raises(SystemExit) as exit_:
        distinct.main()
    return exit_.value.code, json.loads((tmp_path / "result.json").read_text())


def test_passes_when_only_the_red_repeat_reuses_the_image(tmp_path, monkeypatch):
    code, result = run(tmp_path, monkeypatch, {"red": 0, "blue": 0, "red_again": 2048})
    assert code == 0, result["reason"]
    assert result["red_again"]["cached_tokens"] == 2048


@pytest.mark.parametrize("blue", [0, 2048])
def test_fails_when_the_red_repeat_reuses_no_more_than_blue(
    tmp_path, monkeypatch, blue
):
    # 0 / 0: the image fills no whole hash block, so nothing shows the seed
    # mattered; 2048 / 2048: the blue request reused the red image's KV.
    code, result = run(
        tmp_path, monkeypatch, {"red": 0, "blue": blue, "red_again": blue}
    )
    assert code == 1
    assert "second red request reused" in result["reason"]


def test_fails_without_cached_tokens(tmp_path, monkeypatch):
    code, result = run(
        tmp_path, monkeypatch, {"red": None, "blue": None, "red_again": None}
    )
    assert code == 1
    assert "cached_tokens" in result["reason"]


def test_fails_when_both_colors_share_a_key_hash(tmp_path, monkeypatch):
    code, result = run(
        tmp_path,
        monkeypatch,
        {"red": 0, "blue": 0, "red_again": 2048},
        hashes={"red": "aa", "blue": "aa"},
    )
    assert code == 1
    assert "both colors routed on key hash aa" in result["reason"]
