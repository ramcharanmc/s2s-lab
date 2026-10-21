"""
The harness's partial flicker rate (v1 §2): how often a streamed transcript takes back words it had already shown.
Counted from what is committed, with nothing run: the streamed partials of PR-003's two runs, the only partials
anything here streams (the fast tier, R0's and R1's recognizer, run again and again on the growing audio while the
caller speaks, for barge-in). The plan sets no bar, so this is a measurement, its definition written first
(docs/m45-log.md).

    PYTHONPATH=src python -m s2slab.flicker      writes runs/bargein/flicker.json once

THE DEFINITION. Words as the barge-in policies read them (bargein_policies.words_of: lower case, punctuation
dropped, so "Oh no!" and "oh no" are the same). Within one sound event, each partial after the first is an UPDATE.
An update is a REVISION when the previous partial's words are not a prefix of its own: something already shown
changed or went. A TAIL revision changes only the previous partial's last word ("nobody wi" -> "nobody will"); a
DEEP revision changes an earlier one.
  flicker rate              revisions / updates, tail and deep apart
  withdrawn per word shown  words taken back / the previous partials' words, over every update
  settled                   a settled partial (its audio ended in quiet) against the unsettled one before it
Per run, and per kind of item: should stop (stop-long, stop-short) and should keep going (continue-short,
continue-long).

WHAT IT CANNOT SHOW: twenty short items in one voice, the barge-in probe set, not the eval set; no rung's reply
reads partials, so this is the flicker the barge-in policy reads, nothing else.
"""

from __future__ import annotations

import csv
import json
import pathlib

from .bargein_policies import words_of

RUNS = {"measured": pathlib.Path("runs/bargein/pr003-measured.json"),
        "70ms": pathlib.Path("runs/bargein/pr003-70ms.json")}
SCRIPT = pathlib.Path("probes/bargein/script.tsv")
OUT = pathlib.Path("runs/bargein/flicker.json")
GROUPS = {"stop-long": "stop", "stop-short": "stop", "continue-short": "keep going", "continue-long": "keep going"}


def compare(prev: list[str], cur: list[str]) -> dict:
    """One update: was it a revision, how deep, and how many words it took back."""
    if cur[:len(prev)] == prev:
        return {"revision": False, "deep": False, "withdrawn": 0}
    k = 0
    while k < min(len(prev), len(cur)) and prev[k] == cur[k]:
        k += 1
    return {"revision": True, "deep": k < len(prev) - 1, "withdrawn": len(prev) - k}


def events(partials) -> dict[float, list]:
    """Partials (at_s, text, final, event_s, ...) grouped by the sound event they transcribe, in delivery order."""
    out: dict[float, list] = {}
    for p in sorted(partials, key=lambda p: p[0]):
        out.setdefault(p[3], []).append(p)
    return out


def count(partials) -> dict:
    """The counts for one item's partials."""
    c = {"events": 0, "partials": 0, "updates": 0, "revisions": 0, "tail": 0, "deep": 0, "withdrawn": 0,
         "shown": 0, "settled": 0, "settled_revised": 0}
    for ev in events(partials).values():
        c["events"] += 1
        c["partials"] += len(ev)
        for prev, cur in zip(ev, ev[1:]):
            pw, cw = words_of(prev[1]), words_of(cur[1])
            r = compare(pw, cw)
            c["updates"] += 1
            c["shown"] += len(pw)
            c["withdrawn"] += r["withdrawn"]
            if r["revision"]:
                c["revisions"] += 1
                c["deep" if r["deep"] else "tail"] += 1
            if cur[2] and not prev[2]:
                c["settled"] += 1
                c["settled_revised"] += r["revision"]
    return c


def add(total: dict, c: dict) -> dict:
    return {k: total.get(k, 0) + v for k, v in c.items()}


def rates(c: dict) -> dict:
    u, s = c["updates"], c["shown"]
    return {**c, "flicker": round(c["revisions"] / u, 4) if u else None,
            "flicker_tail": round(c["tail"] / u, 4) if u else None,
            "flicker_deep": round(c["deep"] / u, 4) if u else None,
            "withdrawn_per_word": round(c["withdrawn"] / s, 4) if s else None}


def measure(runs: dict[str, dict], cells: dict[str, str]) -> dict:
    """Takes {run name: its file's contents} and {item: its cell}; returns per run, overall and per kind of item."""
    out = {}
    for name, blob in runs.items():
        by: dict[str, dict] = {}
        whole: dict = {}
        for item, partials in blob["partials"].items():
            c = count(partials)
            whole = add(whole, c)
            g = GROUPS[cells[item]]
            by[g] = add(by.get(g, {}), c)
        out[name] = {"all": rates(whole), **{g: rates(v) for g, v in sorted(by.items())}}
    return out


def read_cells(path: pathlib.Path = SCRIPT) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"]: row["cell"] for row in csv.DictReader(f, delimiter="\t")}


def format_report(result: dict) -> str:
    lines = ["The fast tier's streamed partials, PR-003's runs: revisions / updates", "",
             f"{'run':10}{'items':12}{'updates':>8}{'flicker':>9}{'tail':>7}{'deep':>7}{'withdrawn/word':>16}"
             f"{'settled revised':>17}"]
    for name, parts in result.items():
        for g, c in parts.items():
            f = lambda v: "—" if v is None else f"{100 * v:.1f}%"
            lines.append(f"{name:10}{g:12}{c['updates']:>8}{f(c['flicker']):>9}{f(c['flicker_tail']):>7}"
                         f"{f(c['flicker_deep']):>7}{f(c['withdrawn_per_word']):>16}"
                         f"{str(c['settled_revised']) + ' of ' + str(c['settled']):>17}")
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse

    from .sweep import _git_commit
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the count is recorded once")
    runs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in RUNS.items()}
    result = measure(runs, read_cells())
    blob = {"run": "M45, the partial flicker rate", "commit": _git_commit(),
            "from": {k: str(p) for k, p in RUNS.items()}, "result": result}
    OUT.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    print(format_report(result))


if __name__ == "__main__":
    main()
