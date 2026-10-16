"""
M36: the compile cache (s2slab.compilecache), with torch.compiler's save and load replaced by recorders:
nothing compiles here. What it loads, when it writes, and that a broken file only costs the cache.
"""

import pytest

torch = pytest.importorskip("torch")

from s2slab import compilecache  # noqa: E402


@pytest.fixture
def compiler(monkeypatch):
    calls = {"loaded": [], "blob": b""}

    def load_cache_artifacts(data):
        if data == b"broken":
            raise ValueError("not a cache")
        calls["loaded"].append(data)

    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", load_cache_artifacts)
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (calls["blob"], {"inductor": []}) if calls["blob"] else None)
    return calls


def test_no_file_loads_nothing_and_the_first_wake_writes_what_it_compiled(tmp_path, compiler):
    path = tmp_path / "cache" / "l40s.bin"
    loaded = compilecache.load(path)
    assert loaded == {"loaded": False, "bytes": 0} and compiler["loaded"] == []
    compiler["blob"] = b"x" * 100
    assert compilecache.save(path, loaded) == {"saved": True, "bytes": 100, "was": 0}
    assert path.read_bytes() == b"x" * 100 and not (tmp_path / "cache" / "l40s.bin.tmp").exists()


def test_a_later_wake_loads_the_file_and_writes_only_when_it_compiled_more(tmp_path, compiler):
    path = tmp_path / "l40s.bin"
    path.write_bytes(b"y" * 100)
    loaded = compilecache.load(path)
    assert loaded["loaded"] and loaded["bytes"] == 100 and compiler["loaded"] == [b"y" * 100]
    compiler["blob"] = b"y" * 100                                     # the same graphs: nothing new
    assert compilecache.save(path, loaded) == {"saved": False, "bytes": 100, "kept": 100}
    compiler["blob"] = b"y" * 140                                     # a changed rung compiled more
    assert compilecache.save(path, loaded)["saved"] and path.stat().st_size == 140


def test_a_broken_file_costs_only_the_cache_and_is_replaced(tmp_path, compiler):
    path = tmp_path / "l40s.bin"
    path.write_bytes(b"broken")
    loaded = compilecache.load(path)
    assert not loaded["loaded"] and "not a cache" in loaded["error"]
    compiler["blob"] = b"z" * 3                                       # smaller than the file, but the file was no use
    assert compilecache.save(path, loaded)["saved"] and path.read_bytes() == b"z" * 3


def test_nothing_compiled_writes_nothing(tmp_path, compiler):
    path = tmp_path / "l40s.bin"
    assert compilecache.save(path, compilecache.load(path)) == {"saved": False, "bytes": 0} and not path.exists()
