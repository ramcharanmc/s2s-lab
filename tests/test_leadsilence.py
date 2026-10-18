"""
M38: leading silence across TTS engines (s2slab.leadsilence), on made-up renders: the measures against a known
silence and a quiet onset, the trim, the first sound's class, Kokoro's and Piper's adapters on fakes of their
results, and the analysis over a folder of renders with a fake listener.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from s2slab import leadsilence as ls

SR = 24_000


def render(silence_ms=300.0, quiet_ms=0.0, quiet_db=-45.0, seconds=1.0):
    """Digital silence, then optionally a quiet hiss quiet_db below the tone, then a tone."""
    rng = np.random.default_rng(0)
    n0, nq = int(silence_ms / 1000 * SR), int(quiet_ms / 1000 * SR)
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * SR)) / SR)
    hiss = rng.normal(0, 0.5 / np.sqrt(2) * 10 ** (quiet_db / 20), nq)
    return np.concatenate([np.zeros(n0), hiss, tone]).astype(np.float32)


def test_the_measures_find_a_known_silence_at_every_resolution():
    m = ls.measures(render(300.0), SR)
    assert m["rung_ms"] == 300.0 and m["fine30_ms"] == 300.0 and m["fine50_ms"] == 300.0
    assert np.isnan(ls.onset_ms(np.zeros(SR, np.float32), SR))


def test_a_quiet_onset_counts_as_sound_at_50_db_and_as_silence_at_30():
    m = ls.measures(render(200.0, quiet_ms=100.0, quiet_db=-40.0), SR)       # a hiss 40 dB down, 100 ms long
    assert m["fine50_ms"] == 200.0 and m["fine30_ms"] == 300.0


def test_a_trim_keeps_20_ms_before_the_first_sound_and_never_cuts_into_it():
    audio = render(300.0, quiet_ms=100.0, quiet_db=-40.0)
    cut, cut_ms = ls.trim(audio, SR)
    assert cut_ms == pytest.approx(300.0 - 20.0) and ls.onset_ms(cut, SR, 5, 50) == 20.0   # the hiss is the onset
    short, none = ls.trim(render(10.0), SR)
    assert none == 0.0 and len(short) == len(render(10.0))                    # less than 20 ms: nothing to cut


@pytest.mark.parametrize("phonemes,cls", [("ðə kˈæt", "voiced obstruent"), ("ʃˈi", "voiceless fricative"),
                                          ("kˈæn", "voiceless stop"), ("ˈaɪ", "vowel"), ("A wˈʌn", "vowel"),
                                          ("mˈaɪ", "sonorant"), ("hˈaʊ", "voiceless fricative"), ("ʧˈɛk", "voiceless fricative"),
                                          ("", "unknown"), (None, "unknown")])
def test_the_first_sound_is_classed_from_its_phonemes(phonemes, cls):
    assert ls.sound_class(phonemes) == cls


def test_the_first_clauses_are_r1s_first_chunk_of_each_reply(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"turns": {
        "a": {"reply_text": "Sure, I can help with that. What do you need?"},
        "b": {"reply_text": ""}, "c": {"reply_text": "Yes."}}}), encoding="utf-8")
    got = ls.first_clauses("run.json", root=tmp_path)
    assert list(got) == ["a", "c"] and got["a"].startswith("Sure") and got["c"] == "Yes."


def test_kokoros_padding_is_its_first_duration_in_samples():
    audio = np.zeros(6000, np.float32)                    # 10 + 15 + 25 frames over 6000 samples: 120 a frame
    result = SimpleNamespace(audio=audio, phonemes="hˈɛlO", output=SimpleNamespace(pred_dur=np.array([10, 15, 25])))
    row = ls.kokoro_row([result], {"h": 1, "ɛ": 2, "l": 3, "O": 4}, SR)
    assert row["pad_ms"] == pytest.approx(10 * 120 / SR * 1000) and row["ids"] == [[1, 2, 3, 4]]
    assert ls.kokoro_row([SimpleNamespace(audio=audio, phonemes="a", output=None)], {}, SR)["pad_ms"] is None


def test_piper_is_read_through_either_of_its_apis():
    new = SimpleNamespace(synthesize=lambda text: [SimpleNamespace(audio_float_array=np.ones(4, np.float32))] * 2,
                          phonemize=lambda text: [["h", "ə"]])
    assert ls.piper_row(new, "hi", 22_050)["audio"].tolist() == [1.0] * 8 and ls.piper_row(new, "hi", 22_050)["phonemes"] == "hə"
    old = SimpleNamespace(synthesize_stream_raw=lambda text: [np.array([16384, -16384], "<i2").tobytes()],
                          phonemize=lambda text: 1 / 0)
    row = ls.piper_row(old, "hi", 22_050)
    assert row["audio"].tolist() == [0.5, -0.5] and row["phonemes"] is None and row["pad_ms"] is None


def test_the_analysis_measures_trims_and_hears_every_saved_render(tmp_path):
    out = tmp_path / "m38"
    texts = {"c0": "Sure, I can help.", "c1": "Hello there."}
    for engine, silence in (("kokoro", 320.0), ("piper", 120.0)):
        ls.save_renders(engine, {clip: {"audio": render(silence), "sr": SR, "pad_ms": 250.0 if engine == "kokoro" else None,
                                        "phonemes": "sˈʊɹ" if clip == "c0" else "lˈO", "text": text, "render_s": 0.1}
                                 for clip, text in texts.items()}, out)
    run = tmp_path / "r4.json"
    run.write_text(json.dumps({"turns": {"x": {"leading_silence_ms": 80.0}}}), encoding="utf-8")
    heard = {"c0": "sure i can help", "c1": "hello there"}
    lengths = {}

    def hear(audio):                                        # the trimmed render loses its first word: a clipped onset
        lengths.setdefault("n", []).append(len(audio))
        return "i can help" if len(audio) < 16_000 * 1.2 else "sure i can help"

    blob = ls.analyze(out, hear=hear, root=tmp_path, r4_run="r4.json")
    s = blob["summary"]
    assert s["kokoro"]["rung_ms"] == (320.0, 320.0) and s["kokoro"]["pad_ms"] == (250.0, 250.0) and "pad_ms" not in s["piper"]
    assert s["kokoro"]["cut_ms"][0] == pytest.approx(300.0) and s["r4-omni"] == {"n": 1, "rung_ms": (80.0, 80.0)}
    assert s["kokoro"]["by_class"] == {"sonorant": (1, 320.0), "voiceless fricative": (1, 320.0)}
    assert s["kokoro"]["first_heard"][0] >= s["kokoro"]["first_heard"][1]
    text = ls.format_report(blob)
    assert "| kokoro | 2 | 320 / 320 |" in text and "r4-omni" in text and "By the clause's first sound" in text
    side = json.loads((out / "kokoro" / "renders.json").read_text(encoding="utf-8"))
    assert side["note"].startswith("SYNTHETIC SPEECH") and set(side["renders"]["c0"]) == {"text", "sr", "pad_ms", "phonemes", "render_s"}
    saved = ls.save(blob, tmp_path / "ls.json")
    with pytest.raises(Exception):
        ls.save(blob, saved)                                                   # never overwritten
