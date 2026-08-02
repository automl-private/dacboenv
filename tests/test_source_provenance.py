"""Dirty-worktree source revision tests."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from dacboenv.experiment.source_provenance import current_source_revision


def _git(repository: Path, *arguments: str) -> None:
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run(  # noqa: S603
        [executable, *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def test_source_revision_distinguishes_clean_tracked_and_untracked_states(tmp_path: Path) -> None:
    _git(tmp_path, "init", "--quiet")
    (tmp_path / "dacboenv").mkdir()
    tracked = tmp_path / "dacboenv" / "tracked.py"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "dacboenv/tracked.py")
    _git(
        tmp_path,
        "-c",
        "user.name=DACBO test",
        "-c",
        "user.email=dacbo-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )

    clean_revision = current_source_revision(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{40}", clean_revision)

    tracked.write_text("dirty\n", encoding="utf-8")
    tracked_revision = current_source_revision(tmp_path)
    assert re.fullmatch(rf"{clean_revision}\+dirty\.sha256\.[0-9a-f]{{64}}", tracked_revision)
    assert current_source_revision(tmp_path) == tracked_revision

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "result.json").write_text("{}\n", encoding="utf-8")
    assert current_source_revision(tmp_path) == tracked_revision

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "untracked.py").write_text("new\n", encoding="utf-8")
    untracked_revision = current_source_revision(tmp_path)
    assert untracked_revision != tracked_revision
    assert re.fullmatch(rf"{clean_revision}\+dirty\.sha256\.[0-9a-f]{{64}}", untracked_revision)
