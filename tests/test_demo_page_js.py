"""
The page's stream helpers (ndjsonLines, pcm16ToFloat, wavFrom), run in node, and the page's whole
script checked for syntax there: the parts of M32a's streaming that live in the browser. Skipped
where node is not installed.
"""

import json
import pathlib
import re
import shutil
import subprocess

import numpy as np
import pytest

from s2slab import demo

PAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "s2slab" / "demo_page.html"
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")


def run_js(code: str, tmp_path) -> str:
    f = tmp_path / "t.mjs"
    f.write_text(code, encoding="utf-8")
    out = subprocess.run([node, str(f)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout


def helpers() -> str:
    html = PAGE.read_text(encoding="utf-8")
    m = re.search(r"// <stream-helpers>.*?\n(.*?)// </stream-helpers>", html, re.S)
    assert m, "the page has no stream helpers"
    return m.group(1)


def test_the_page_script_parses(tmp_path):
    html = PAGE.read_text(encoding="utf-8")
    script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    f = tmp_path / "page.js"
    f.write_text(script, encoding="utf-8")
    out = subprocess.run([node, "--check", str(f)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


def test_lines_split_across_reads_come_out_whole(tmp_path):
    js = helpers() + """
const s = {buf: ""}, got = [];
for (const part of ['{"a": 1}\\n{"b"', ': 2}\\n', '\\n{"c": 3}']) got.push(...ndjsonLines(s, part));
got.push(...ndjsonLines(s, "\\n"));
console.log(JSON.stringify(got));
"""
    assert json.loads(run_js(js, tmp_path)) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_the_servers_pcm_comes_back_as_the_same_floats_and_a_valid_wav(tmp_path):
    x = np.array([0.0, 0.5, -0.5, 0.999, -1.0, 0.25], dtype=np.float32)
    js = helpers() + f"""
const f = pcm16ToFloat({json.dumps(demo.pcm16(x))});
const blob = wavFrom([f.slice(0, 3), f.slice(3)], 24000), b = new Uint8Array(await blob.arrayBuffer()), v = new DataView(b.buffer);
const tag = (o) => String.fromCharCode(...b.slice(o, o + 4));
console.log(JSON.stringify({{floats: Array.from(f), size: b.length, riff: tag(0), wave: tag(8), sr: v.getUint32(24, true),
                             data: v.getUint32(40, true), first: v.getInt16(46, true)}}));
"""
    out = json.loads(run_js(js, tmp_path))
    assert out["floats"] == pytest.approx(x.tolist(), abs=1e-4)
    assert (out["size"], out["riff"], out["wave"], out["sr"], out["data"]) == (44 + 12, "RIFF", "WAVE", 24000, 12)
    assert out["first"] == 16384                                              # 0.5, as 16-bit


def test_the_page_counts_the_gaps_its_listener_would_hear(tmp_path):
    js = helpers() + """
const p = {next: 0, dry: 0}, at = [];
at.push(schedule(p, 0.0, 0.46));              // the first block: on arrival plus 150 ms
at.push(schedule(p, 0.40, 1.92));             // the second, before the first has ended: straight after it
at.push(schedule(p, 3.00, 1.92));             // the third, 0.47 s after the second ended, plus 50 ms: a gap
console.log(JSON.stringify({at, dry: p.dry, next: p.next}));
"""
    out = json.loads(run_js(js, tmp_path))
    assert out["at"] == pytest.approx([0.15, 0.61, 3.05]) and out["dry"] == pytest.approx(0.52)
    assert out["next"] == pytest.approx(3.05 + 1.92)
