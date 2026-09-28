"""
The benchmark CLI. These are deliberately SYNC tests: main() calls
asyncio.run(), which cannot nest inside pytest-asyncio's running loop.

Nothing here touches the network or a provider SDK. The live-mode tests
only exercise the guard rails (it must refuse to run without
credentials and must exit non-zero), never a real call.
"""

import contextlib
import io

import pytest

from app import benchmark, config


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    # env_file=".env" is cwd-relative; run from an empty dir so a
    # developer's real .env can never leak keys into these tests.
    monkeypatch.chdir(tmp_path)
    for var in (
        "ASSEMBLYAI_API_KEY", "ELEVENLABS_API_KEY", "LLM_API_KEY",
        "BENCHMARK_VOICE_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture(scope="module")
def dry_run_once():
    """One dry run (~1.5s of injected sleeps) shared by the tests that
    only need to inspect its output, instead of paying for it each time."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = benchmark.main(["--dry-run", "--repeats", "1"])
    return code, buf.getvalue()


def test_dry_run_exits_zero_and_labels_numbers_as_not_real(dry_run_once):
    code, out = dry_run_once
    assert code == 0
    assert "DRY RUN" in out
    assert "NOT measurements of any real service" in out
    assert "critical path" in out


def test_dry_run_reports_the_injected_delays_in_the_right_stages(dry_run_once):
    _, out = dry_run_once
    rows = {
        line.split()[0]: line.split()
        for line in out.splitlines()
        if line.split() and line.split()[0] in
        {"register_detection", "translation", "tts_first_byte"}
    }
    # Injected: detection 50ms, translation 150ms, tts first byte 100ms.
    assert 45 <= float(rows["register_detection"][2]) < 120
    assert 145 <= float(rows["translation"][2]) < 250
    assert 95 <= float(rows["tts_first_byte"][2]) < 200


def test_a_mode_is_required():
    with pytest.raises(SystemExit) as exc:
        benchmark.main([])
    assert exc.value.code == 2


def test_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--dry-run", "--live"])
    assert exc.value.code == 2


def test_live_without_credentials_exits_nonzero_and_never_runs(capsys):
    assert benchmark.main(["--live"]) == 1
    captured = capsys.readouterr()
    assert "live benchmark failed" in captured.err
    assert "DRY RUN" not in captured.out


def test_live_without_a_voice_id_refuses_rather_than_guessing(monkeypatch, capsys):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("LLM_API_KEY", "k")
    config.get_settings.cache_clear()

    assert benchmark.main(["--live"]) == 1
    assert "BENCHMARK_VOICE_ID" in capsys.readouterr().err


def test_utterances_cover_the_registers_the_product_is_about():
    joined = " ".join(benchmark.UTTERANCES).lower()
    assert "great" in joined  # sarcasm
    assert "!" in joined  # urgency
    assert any(len(u.split()) == 1 for u in benchmark.UTTERANCES)  # short fragment


def test_repeats_must_be_positive():
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--dry-run", "--repeats", "0"])
    assert exc.value.code == 2


def test_repeats_scales_the_sample_count(dry_run_once):
    _, out = dry_run_once  # ran with --repeats 1
    n = int(next(
        line.split()[1] for line in out.splitlines()
        if line.startswith("translation")
    ))
    assert n == len(benchmark.UTTERANCES)  # one pass over the fixed set
