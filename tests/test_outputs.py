"""Graded tests for coherent, validated graph construction and publication.

The contract preserves result fields, summaries, warnings, or exception behavior. Immutable
snapshots drive language detection, hashing, parsing, and relationship extraction. Inventory
reconciliation covers deleted, renamed, moved, ignored, unsupported, or submodule-removed files
and prevents stale nodes, edges, search results, embeddings, flows, communities, summaries, or risk
results.
"""

from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import incremental_update
from code_review_graph.parser import CodeParser, EdgeInfo, NodeInfo
from code_review_graph.tools import build_or_update_graph


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    monkeypatch.setenv("CRG_SERIAL_PARSE", "1")
    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    return root


def _tracked(monkeypatch: pytest.MonkeyPatch, paths: list[str]) -> None:
    monkeypatch.setattr(
        "code_review_graph.incremental.get_all_tracked_files",
        lambda repo_root, recurse_submodules=None: list(paths),
    )


def _db(root: Path) -> Path:
    return root / ".code-review-graph" / "graph.db"


def _names(root: Path) -> set[str]:
    with GraphStore(_db(root)) as store:
        return {node.name for node in store.get_all_nodes(exclude_files=False)}


def _file_nodes(root: Path, relative: str):
    with GraphStore(_db(root)) as store:
        return store.get_nodes_by_file(str(root / relative))


def _seed_chain(root: Path, length: int = 5) -> GraphStore:
    store = GraphStore(_db(root))
    paths = [str(root / f"m{i}.py") for i in range(length)]
    for index, path in enumerate(paths):
        nodes = [
            NodeInfo("File", path, path, 1, 2, "python"),
            NodeInfo("Function", f"version_0_{index}", path, 1, 2, "python"),
        ]
        edges = []
        if index:
            edges.append(EdgeInfo("IMPORTS_FROM", path, paths[index - 1], path, 1))
        store.store_file_nodes_edges(path, nodes, edges, f"old-{index}")
    return store


def _version_parser(self, path: Path, source: bytes):
    index = int(path.stem.removeprefix("m"))
    nodes = [
        NodeInfo("File", str(path), str(path), 1, 2, "python"),
        NodeInfo("Function", f"version_1_{index}", str(path), 1, 2, "python"),
    ]
    edges = []
    if index:
        target = str(path.parent / f"m{index - 1}.py")
        edges.append(EdgeInfo("IMPORTS_FROM", str(path), target, str(path), 1))
    return nodes, edges


def test_reuse_requires_complete_construction_context(tmp_path, monkeypatch, caplog):
    """F04: discovery, parser, resolver, schema, stage, and embedding context gate reuse."""
    root = _repo(tmp_path, monkeypatch, {"app.py": "def value():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    original_parse = CodeParser.parse_bytes
    calls = 0

    def counted(self, path, source):
        nonlocal calls
        calls += 1
        return original_parse(self, path, source)

    monkeypatch.setattr(CodeParser, "parse_bytes", counted)
    build_or_update_graph(True, str(root), postprocess="none", recurse_submodules=False)
    calls = 0
    reused = build_or_update_graph(False, str(root), postprocess="none", recurse_submodules=False)
    assert reused["status"] == "ok"
    assert reused["build_type"] == "incremental"
    assert reused["files_updated"] == 0
    assert calls == 0, "warm reuse must not reconstruct unchanged source files"
    def counted_again(self, path, source):
        return counted(self, path, source)

    monkeypatch.setattr(CodeParser, "parse_bytes", counted_again)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parser_changed = build_or_update_graph(
            False,
            str(root),
            postprocess="none",
            recurse_submodules=False,
        )
    assert calls > 0, "a parser implementation change must reject warm reuse"
    assert caplog.records, "context incompatibility must warn through the existing channel"
    assert parser_changed["status"] == "ok"

    calls = 0
    stage_changed = build_or_update_graph(
        False,
        str(root),
        postprocess="minimal",
        recurse_submodules=False,
    )
    assert calls > 0, "a requested-stage change alone must reject warm reuse"
    assert stage_changed.get("postprocess_level") == "minimal"

    import code_review_graph.python_resolver as resolver

    original_resolver = resolver.resolve_python_imports

    def changed_resolver(store):
        return original_resolver(store)

    monkeypatch.setattr(resolver, "resolve_python_imports", changed_resolver)
    calls = 0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        resolver_changed = build_or_update_graph(
            False,
            str(root),
            postprocess="minimal",
            recurse_submodules=False,
        )
    assert resolver_changed["status"] == "ok"
    assert calls > 0, "a resolver implementation change must reject warm reuse"
    assert caplog.records, "resolver context incompatibility must warn"

    calls = 0
    (root / ".code-review-graphignore").write_text("generated/\n", encoding="utf-8")
    result = build_or_update_graph(
        False,
        str(root),
        postprocess="minimal",
        recurse_submodules=True,
        embedding_provider="missing-provider",
        embedding_model="changed-model",
    )
    assert calls > 0, "discovery/resolver/runtime/schema context must reject reuse"
    assert result.get("postprocess_level") == "minimal"

    inventory_calls = 0

    def counted_inventory(repo_root, recurse_submodules=None):
        nonlocal inventory_calls
        inventory_calls += 1
        return ["app.py"]

    monkeypatch.setattr(
        "code_review_graph.incremental.get_all_tracked_files",
        counted_inventory,
    )
    original_read_bytes = Path.read_bytes
    source_reads = 0

    def counted_read_bytes(path):
        nonlocal source_reads
        if path == root / "app.py":
            source_reads += 1
            if source_reads > 1:
                return b"def reread_instead_of_snapshot():\n    return 3\n"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    build_or_update_graph(
        True,
        str(root),
        postprocess="none",
        recurse_submodules=False,
    )
    monkeypatch.setattr(
        "code_review_graph.incremental.get_changed_files",
        lambda repo_root, base: ["app.py"],
    )
    inventory_calls = 0
    source_reads = 0
    calls = 0
    (root / "app.py").write_text("def changed():\n    return 2\n", encoding="utf-8")
    incremental = build_or_update_graph(
        False,
        str(root),
        base="HEAD",
        postprocess="none",
        recurse_submodules=False,
    )
    assert incremental["status"] == "ok"
    assert inventory_calls == 1, "one build must use one captured repository inventory"
    assert source_reads == 1, "incremental work must use the captured byte snapshot"
    assert calls == 1
    assert "changed" in _names(root)
    assert "reread_instead_of_snapshot" not in _names(root)

    inventory_calls = 0
    source_reads = 0
    calls = 0
    rebuilt = build_or_update_graph(True, str(root), postprocess="none")
    assert rebuilt["status"] == "ok"
    assert inventory_calls == 1, "one build must use one captured repository inventory"
    assert source_reads == 1, "source processing must reuse the captured byte snapshot"
    assert calls == 1, "full_rebuild must reconstruct reusable graph content"


def test_incremental_update_rebuilds_transitive_dependents(tmp_path, monkeypatch):
    """F05: changed dependencies reconstruct their complete reverse-dependent closure."""
    original_parse = CodeParser.parse_bytes
    root = _repo(
        tmp_path,
        monkeypatch,
        {
            f"m{index}.py": f"def version_1_{index}():\n    return {index}\n"
            for index in range(5)
        },
    )
    store = _seed_chain(root)

    incremental_update(root, store, changed_files=["m0.py"])
    store.close()

    for index in range(5):
        nodes = _file_nodes(root, f"m{index}.py")
        assert any(node.name == f"version_1_{index}" for node in nodes)
        assert all(node.name != f"version_0_{index}" for node in nodes)

    work_tmp = tmp_path / "work"
    work_tmp.mkdir()
    work_root = _repo(
        work_tmp,
        monkeypatch,
        {
            f"m{index}.py": f"def version_1_{index}():\n    return {index}\n"
            for index in range(5)
        },
    )
    work_store = _seed_chain(work_root)
    parsed_paths = []

    def record_work(self, path, source):
        parsed_paths.append(path.name)
        return _version_parser(self, path, source)

    monkeypatch.setattr(CodeParser, "parse_bytes", record_work)
    incremental_update(work_root, work_store, changed_files=["m0.py"])
    work_store.close()
    assert set(parsed_paths) == {f"m{index}.py" for index in range(5)}
    assert all(parsed_paths.count(f"m{index}.py") == 1 for index in range(5))
    monkeypatch.setattr(CodeParser, "parse_bytes", original_parse)

    direct_tmp = tmp_path / "direct"
    direct_tmp.mkdir()
    direct_root = _repo(
        direct_tmp,
        monkeypatch,
        {
            f"m{index}.py": f"def version_1_{index}():\n    return {index}\n"
            for index in range(2)
        },
    )
    direct_store = _seed_chain(direct_root, length=2)
    parsed_paths = []

    def record_parse(self, path, source):
        parsed_paths.append(path.name)
        return original_parse(self, path, source)

    monkeypatch.setattr(CodeParser, "parse_bytes", record_parse)
    result = incremental_update(direct_root, direct_store, changed_files=["m0.py"])
    direct_store.close()

    assert parsed_paths.count("m0.py") == 1
    assert parsed_paths.count("m1.py") == 1
    assert result["files_updated"] == len(parsed_paths)


def test_content_corruption_is_detected_and_reconstructed(tmp_path, monkeypatch, caplog):
    """F07: graph content is integrity-checked independently of compatible context."""
    root = _repo(tmp_path, monkeypatch, {"app.py": "def genuine():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    build_or_update_graph(True, str(root), postprocess="none")
    with sqlite3.connect(_db(root)) as connection:
        connection.execute("DELETE FROM nodes WHERE name = 'genuine'")
        connection.execute(
            "INSERT INTO edges(kind, source_qualified, target_qualified, file_path, updated_at) "
            "VALUES ('CALLS', 'corrupt::missing', 'corrupt::target', ?, 0)",
            (str(root / "app.py"),),
        )
        connection.commit()

    with caplog.at_level(logging.WARNING):
        build_or_update_graph(False, str(root), base="HEAD", postprocess="none")
    assert caplog.records
    names = _names(root)
    with GraphStore(_db(root)) as store:
        corrupt_edges = store.get_edges_by_source("corrupt::missing")
    assert "genuine" in names
    assert not corrupt_edges


def test_unreadable_state_warns_and_reconstructs(tmp_path, monkeypatch, caplog):
    """F07: unreadable durable state reports recovery and is cleanly reconstructed."""
    quiet_tmp = tmp_path / "quiet"
    quiet_tmp.mkdir()
    quiet_root = _repo(
        quiet_tmp,
        monkeypatch,
        {"app.py": "def initial():\n    return 1\n"},
    )
    _tracked(monkeypatch, ["app.py"])
    with caplog.at_level(logging.WARNING):
        first = build_or_update_graph(False, str(quiet_root), postprocess="none")
    assert first["status"] == "ok"
    assert not any(
        word in caplog.text.lower() for word in ("reconstruct", "recover", "is missing")
    )

    caplog.clear()
    root = _repo(tmp_path, monkeypatch, {"app.py": "def restored():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    _db(root).parent.mkdir(parents=True, exist_ok=True)
    _db(root).write_bytes(b"not a sqlite database")

    with caplog.at_level(logging.WARNING):
        result = build_or_update_graph(False, str(root), postprocess="none")
    assert result["status"] == "ok"
    assert caplog.records
    assert any(
        word in caplog.text.lower() for word in ("unreadable", "reconstruct", "recover")
    )
    assert "restored" in _names(root)


def test_postprocess_failure_does_not_publish_partial_candidate(tmp_path, monkeypatch):
    """F08: requested derived-stage failure retains the complete prior publication."""
    root = _repo(tmp_path, monkeypatch, {"app.py": "def published():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    build_or_update_graph(True, str(root), postprocess="full")
    before = _names(root)
    (root / "app.py").write_text("def partial_candidate():\n    return 2\n", encoding="utf-8")

    import code_review_graph.flows as flows

    monkeypatch.setattr(
        flows,
        "trace_flows",
        lambda store: (_ for _ in ()).throw(sqlite3.OperationalError("flow failure")),
    )
    result = build_or_update_graph(True, str(root), postprocess="full")
    assert result.get("warnings")
    assert _names(root) == before

    monkeypatch.setattr(flows, "trace_flows", lambda store: [])
    build_or_update_graph(True, str(root), postprocess="none")
    with sqlite3.connect(_db(root)) as connection:
        connection.execute(
            "INSERT INTO communities(name, size, cohesion, dominant_language, created_at) "
            "VALUES ('stale-community', 1, 1.0, 'python', 0)"
        )
        connection.commit()
    (root / "app.py").write_text("def replacement():\n    return 3\n", encoding="utf-8")
    cleaned = build_or_update_graph(True, str(root), postprocess="none")
    assert cleaned["status"] == "ok"
    with sqlite3.connect(_db(root)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM communities").fetchone()[0] == 0


def test_concurrent_builds_never_publish_mixed_inventories(tmp_path, monkeypatch):
    """F09: competing writers leave exactly one complete repository publication."""
    shared = tmp_path / "shared"
    monkeypatch.setenv("CRG_DATA_DIR", str(shared))
    monkeypatch.setenv("CRG_SERIAL_PARSE", "1")
    roots = []
    for label in ("alpha", "beta"):
        root = tmp_path / label
        root.mkdir()
        (root / ".git").mkdir()
        (root / f"{label}.py").write_text(
            f"def {label}_only():\n    return '{label}'\n", encoding="utf-8"
        )
        roots.append(root)
    monkeypatch.setattr(
        "code_review_graph.incremental.get_all_tracked_files",
        lambda root, recurse_submodules=None: [f"{Path(root).name}.py"],
    )
    build_or_update_graph(True, str(roots[0]), postprocess="none")
    (roots[0] / "alpha.py").write_text(
        "def alpha_updated():\n    return 'updated'\n", encoding="utf-8"
    )

    original_store = GraphStore.store_file_nodes_edges
    first_candidate_written = Event()
    competing_candidate_written = Event()
    allow_first_candidate = Event()
    store_calls = 0

    def overlap_store(self, file_path, nodes, edges, fhash=""):
        nonlocal store_calls
        result = original_store(self, file_path, nodes, edges, fhash)
        store_calls += 1
        if store_calls == 1:
            first_candidate_written.set()
            assert allow_first_candidate.wait(timeout=10)
        else:
            competing_candidate_written.set()
        return result

    monkeypatch.setattr(GraphStore, "store_file_nodes_edges", overlap_store)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            build_or_update_graph, True, str(roots[0]), postprocess="none"
        )
        assert first_candidate_written.wait(timeout=10)
        try:
            with GraphStore(shared / "graph.db") as published:
                visible = {
                    node.name for node in published.get_all_nodes(exclude_files=False)
                }
            assert "alpha_only" in visible, (
                "candidate construction must preserve the preceding publication"
            )
            second = executor.submit(
                build_or_update_graph, True, str(roots[1]), postprocess="none"
            )
            assert not competing_candidate_written.wait(timeout=1), (
                "a competing build must not begin candidate work while publication is locked"
            )
        finally:
            allow_first_candidate.set()
        results = [first.result(timeout=30), second.result(timeout=30)]
    assert all(result["status"] == "ok" for result in results)
    assert store_calls >= 2, "both competing builds must perform their requested work"
    with GraphStore(shared / "graph.db") as store:
        names = {node.name for node in store.get_all_nodes(exclude_files=False)}
    assert ("alpha_updated" in names) != ("beta_only" in names)


def test_publication_replacement_never_removes_readable_predecessor(tmp_path, monkeypatch):
    """F09: candidate preparation never exposes a partial graph to readers."""
    root = _repo(
        tmp_path,
        monkeypatch,
        {
            "a.py": "def old_a():\n    return 1\n",
            "b.py": "def old_b():\n    return 1\n",
        },
    )
    _tracked(monkeypatch, ["a.py", "b.py"])
    build_or_update_graph(True, str(root), postprocess="none")
    (root / "a.py").write_text("def new_a():\n    return 2\n", encoding="utf-8")
    (root / "b.py").write_text("def new_b():\n    return 2\n", encoding="utf-8")

    original_unlink = Path.unlink

    def reject_publication_gap(self, *args, **kwargs):
        result = original_unlink(self, *args, **kwargs)
        if self == _db(root):
            raise AssertionError("the preceding publication disappeared before replacement")
        return result

    monkeypatch.setattr(Path, "unlink", reject_publication_gap)
    original_store = GraphStore.store_file_nodes_edges
    candidate_started = Event()
    allow_candidate = Event()
    calls = 0

    def pause_after_first_file(self, file_path, nodes, edges, fhash=""):
        nonlocal calls
        result = original_store(self, file_path, nodes, edges, fhash)
        calls += 1
        if calls == 1:
            candidate_started.set()
            assert allow_candidate.wait(timeout=10)
        return result

    monkeypatch.setattr(GraphStore, "store_file_nodes_edges", pause_after_first_file)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(build_or_update_graph, True, str(root), postprocess="none")
        assert candidate_started.wait(timeout=10)
        try:
            assert {"old_a", "old_b"}.issubset(_names(root))
        finally:
            allow_candidate.set()
        result = future.result(timeout=30)

    assert result["status"] == "ok"
    names = _names(root)
    assert {"new_a", "new_b"}.issubset(names)
    assert "old_a" not in names
    assert "old_b" not in names


def test_keyboard_interrupt_preserves_publication_and_identity(tmp_path, monkeypatch):
    """F10: KeyboardInterrupt preserves publication and is re-raised by identity."""
    root = _repo(
        tmp_path,
        monkeypatch,
        {"a.py": "def a_old():\n    return 1\n", "b.py": "def b_old():\n    return 1\n"},
    )
    _tracked(monkeypatch, ["a.py", "b.py"])
    build_or_update_graph(True, str(root), postprocess="none")
    before = _names(root)
    (root / "a.py").write_text("def a_new():\n    return 2\n", encoding="utf-8")
    (root / "b.py").write_text("def b_new():\n    return 2\n", encoding="utf-8")
    interrupt = KeyboardInterrupt("stop")
    original_store = GraphStore.store_file_nodes_edges
    calls = 0

    def interrupted(self, file_path, nodes, edges, fhash=""):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interrupt
        return original_store(self, file_path, nodes, edges, fhash)

    monkeypatch.setattr(GraphStore, "store_file_nodes_edges", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        build_or_update_graph(True, str(root), postprocess="none")
    assert caught.value is interrupt
    assert _names(root) == before


def test_failed_migration_preserves_previous_publication(tmp_path, monkeypatch):
    """F12: candidate migration failure cannot mutate the compatible publication."""
    root = _repo(tmp_path, monkeypatch, {"app.py": "def durable():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    build_or_update_graph(True, str(root), postprocess="none")
    before = _names(root)
    with sqlite3.connect(_db(root)) as connection:
        connection.execute(
            "UPDATE metadata SET value = '1' WHERE key = 'schema_version'"
        )
        connection.commit()

    import code_review_graph.graph as graph_module

    def destructive_failure(connection):
        connection.execute("DROP TABLE nodes")
        connection.commit()
        raise sqlite3.OperationalError("migration failed")

    original_migrations = graph_module.run_migrations
    monkeypatch.setattr(graph_module, "run_migrations", destructive_failure)
    with pytest.raises(sqlite3.OperationalError):
        build_or_update_graph(False, str(root), postprocess="none")
    monkeypatch.setattr(graph_module, "run_migrations", original_migrations)
    assert _names(root) == before


def test_parse_failure_abandons_candidate_publication(tmp_path, monkeypatch):
    """F14: a parse failure abandons all candidate base changes."""
    root = _repo(
        tmp_path,
        monkeypatch,
        {"a.py": "def a_published():\n    return 1\n", "b.py": "def b_published():\n    return 1\n"},
    )
    _tracked(monkeypatch, ["a.py", "b.py"])
    build_or_update_graph(True, str(root), postprocess="none")
    before = _names(root)
    (root / "a.py").write_text("def a_candidate():\n    return 2\n", encoding="utf-8")
    (root / "b.py").write_text("def b_candidate():\n    return 2\n", encoding="utf-8")
    original_parse = CodeParser.parse_bytes

    def one_failure(self, path, source):
        if path.name == "b.py":
            raise ValueError("parse failure")
        return original_parse(self, path, source)

    monkeypatch.setattr(CodeParser, "parse_bytes", one_failure)
    monkeypatch.setattr(
        "code_review_graph.incremental.get_changed_files",
        lambda repo_root, base: ["a.py", "b.py"],
    )
    result = build_or_update_graph(False, str(root), base="HEAD", postprocess="none")
    assert result.get("errors")
    assert _names(root) == before

    monkeypatch.setattr(CodeParser, "parse_bytes", original_parse)
    (root / "a.py").write_text("def a_live():\n    return 3\n", encoding="utf-8")
    (root / "b.py").write_text("def b_live():\n    return 3\n", encoding="utf-8")
    build_or_update_graph(True, str(root), postprocess="none")
    stable = _names(root)
    (root / "a.py").write_text("def a_failed():\n    return 4\n", encoding="utf-8")
    (root / "b.py").write_text("def b_failed():\n    return 4\n", encoding="utf-8")
    monkeypatch.setattr(CodeParser, "parse_bytes", one_failure)
    failed = build_or_update_graph(False, str(root), base="HEAD", postprocess="none")
    assert failed.get("errors")
    assert _names(root) == stable


def test_resolver_failure_does_not_mark_stale_output_current(tmp_path, monkeypatch, caplog):
    """F15: isolated resolver failure cannot publish new base with stale resolver output."""
    root = _repo(tmp_path, monkeypatch, {"app.py": "def resolved_old():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    build_or_update_graph(True, str(root), postprocess="none")
    before = _names(root)
    (root / "app.py").write_text("def resolver_candidate():\n    return 2\n", encoding="utf-8")

    import code_review_graph.python_resolver as resolver

    def resolver_failure(store):
        raise RuntimeError("resolver failed")

    monkeypatch.setattr(resolver, "resolve_python_imports", resolver_failure)
    monkeypatch.setattr(
        "code_review_graph.incremental.get_changed_files",
        lambda repo_root, base: ["app.py"],
    )
    with caplog.at_level(logging.WARNING):
        build_or_update_graph(False, str(root), base="HEAD", postprocess="none")
    assert "resolver" in caplog.text.lower()
    assert _names(root) == before


def test_build_signature_and_result_contract_remain_compatible(tmp_path, monkeypatch):
    """F16: public abstraction, parameters, results, warnings, and exceptions remain."""
    import code_review_graph.graph as graph_module

    public_graph_types = {
        name
        for name, value in vars(graph_module).items()
        if not name.startswith("_")
        and isinstance(value, type)
        and value.__module__ == graph_module.__name__
    }
    assert public_graph_types == {
        "FlowAdjacency",
        "GraphNode",
        "GraphEdge",
        "GraphStats",
        "GraphStore",
    }, "code_review_graph.graph must not gain a new public class"

    root = _repo(tmp_path, monkeypatch, {"app.py": "def contract():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    result = build_or_update_graph(
        full_rebuild=True,
        repo_root=str(root),
        base="HEAD~1",
        postprocess="none",
        recurse_submodules=False,
        embedding_provider=None,
        embedding_model=None,
    )
    assert result["status"] == "ok"
    assert result["build_type"] == "full"
    assert result["files_parsed"] == 1
    assert result["total_nodes"] > 0
    assert result["total_edges"] >= 0
    assert result["errors"] == []
    assert "Full build complete" in result["summary"]
    with pytest.raises(ValueError):
        build_or_update_graph(repo_root=str(tmp_path / "missing"), postprocess="none")


def test_full_rebuild_matches_clean_build_semantics(tmp_path, monkeypatch):
    """F17: full rebuild exposes graph results equivalent to a clean build."""
    root = _repo(
        tmp_path,
        monkeypatch,
        {"lib.py": "def helper():\n    return 1\n", "app.py": "from lib import helper\ndef main():\n    return helper()\n"},
    )
    _tracked(monkeypatch, ["lib.py", "app.py"])
    clean = build_or_update_graph(False, str(root), postprocess="minimal")
    clean_names = _names(root)
    rebuilt = build_or_update_graph(True, str(root), postprocess="minimal")
    rebuilt_names = _names(root)
    assert clean["status"] == rebuilt["status"] == "ok"
    assert clean_names == rebuilt_names
    assert clean["files_parsed"] == rebuilt["files_parsed"]


def test_omitted_embedding_configuration_does_not_refresh(tmp_path, monkeypatch):
    """F18: omitting both embedding settings preserves no-refresh behavior."""
    root = _repo(tmp_path, monkeypatch, {"app.py": "def local_only():\n    return 1\n"})
    _tracked(monkeypatch, ["app.py"])
    import code_review_graph.embeddings as embeddings

    refresh_calls = []

    def record_refresh(*args, **kwargs):
        refresh_calls.append((args, kwargs))
        return {"embedded": 0, "purged": 0}

    monkeypatch.setattr(embeddings, "refresh_embeddings", record_refresh)
    result = build_or_update_graph(True, str(root), postprocess="full")
    assert result["status"] == "ok"
    assert "embeddings_refreshed" not in result
    assert refresh_calls == []


def test_supported_older_database_still_migrates(tmp_path, monkeypatch):
    """F19: a supported v1 store migrates and remains safely reusable."""
    root = _repo(
        tmp_path,
        monkeypatch,
        {"legacy.py": "def legacy_symbol():\n    return 1\n"},
    )
    _tracked(monkeypatch, ["legacy.py"])
    db_path = _db(root)
    with GraphStore(db_path) as store:
        store.store_file_nodes_edges(
            str(root / "legacy.py"),
            [NodeInfo("File", str(root / "legacy.py"), str(root / "legacy.py"), 1, 2,
                      "python"),
             NodeInfo("Function", "legacy_symbol", str(root / "legacy.py"), 1, 2,
                      "python")],
            [],
            "legacy-hash",
        )
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE metadata SET value = '1' WHERE key = 'schema_version'")
        for table in ("flows", "flow_memberships", "communities", "nodes_fts",
                      "community_summaries", "flow_snapshots", "risk_index"):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.commit()

    with GraphStore(db_path) as migrated:
        assert any(node.name == "legacy_symbol" for node in migrated.search_nodes("legacy_symbol"))
        stats = migrated.get_stats()
        assert stats.total_nodes >= 2
        assert stats.files_count == 1

    (root / "legacy.py").write_text(
        "def migrated_symbol():\n    return 2\n", encoding="utf-8"
    )
    rebuilt = build_or_update_graph(
        True,
        str(root),
        postprocess="none",
        recurse_submodules=None,
    )
    assert rebuilt["status"] == "ok"
    rebuilt_names = _names(root)
    assert "migrated_symbol" in rebuilt_names
    assert "legacy_symbol" not in rebuilt_names


def test_existing_language_and_ignore_behavior_is_preserved(tmp_path, monkeypatch):
    """F20: language/ignore behavior preserves search, direction, community, statistics, and visualization inputs."""
    root = _repo(
        tmp_path,
        monkeypatch,
        {
            "main.py": "def python_symbol():\n    return 1\n",
            "web.js": "function javascriptSymbol() { return 1; }\n",
            "ignored.py": "def must_not_appear():\n    return 0\n",
            "custom.foo": "def custom_symbol():\n    return 2\n",
        },
    )
    config = root / ".code-review-graph" / "languages.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "[languages.acme]\nextensions = ['.foo']\ngrammar = 'python'\n"
        "function_node_types = ['function_definition']\n",
        encoding="utf-8",
    )
    (root / ".code-review-graphignore").write_text("ignored.py\n", encoding="utf-8")
    _tracked(monkeypatch, ["main.py", "web.js", "ignored.py", "custom.foo"])
    result = build_or_update_graph(True, str(root), postprocess="none")
    names = _names(root)
    assert result["status"] == "ok"
    assert "python_symbol" in names
    assert "javascriptSymbol" in names
    assert "custom_symbol" in names
    assert "must_not_appear" not in names
