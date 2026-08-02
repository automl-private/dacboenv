"""Git source-revision provenance that does not hide dirty worktrees."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

_SOURCE_ROOTS = frozenset({"dacboenv", "scripts", "tests"})
_SOURCE_FILES = frozenset({"Makefile", "pyproject.toml", "uv.lock"})


def _git(repository: Path | None, *arguments: str) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git")
    result = subprocess.run(  # noqa: S603
        [executable, *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return result.stdout


def current_source_revision(repository: Path | None = None) -> str:
    """Return HEAD, augmented by a reproducible dirty-source digest.

    A bare commit identifies the source only for a clean checkout. For a dirty
    tree, the digest covers tracked and untracked implementation/config/test
    inputs. Generated reports, run directories, and result artifacts are
    deliberately excluded so repeated evaluation does not change its own
    source identity.
    """
    try:
        head = _git(repository, "rev-parse", "HEAD").decode().strip()
        root = Path(_git(repository, "rev-parse", "--show-toplevel").decode().strip())
        tracked_diff = _git(
            repository,
            "diff",
            "--binary",
            "HEAD",
            "--",
            *sorted(_SOURCE_ROOTS),
            *sorted(_SOURCE_FILES),
        )
        untracked = [
            path
            for path in _git(repository, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
            if path and _is_source_path(path)
        ]
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return "unavailable"

    if not tracked_diff and not untracked:
        return head

    digest = hashlib.sha256()
    digest.update(tracked_diff)
    for encoded_path in sorted(untracked):
        digest.update(b"\0path\0")
        digest.update(encoded_path)
        path = root / encoded_path.decode(errors="surrogateescape")
        if path.is_file():
            digest.update(b"\0content\0")
            digest.update(path.read_bytes())
    return f"{head}+dirty.sha256.{digest.hexdigest()}"


def _is_source_path(encoded_path: bytes) -> bool:
    """Return whether an untracked path can affect protocol execution."""
    path = Path(encoded_path.decode(errors="surrogateescape"))
    return bool(path.parts) and (path.parts[0] in _SOURCE_ROOTS or path.as_posix() in _SOURCE_FILES)


__all__ = ["current_source_revision"]
