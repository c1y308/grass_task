from __future__ import annotations

import re
import sys


_CONDA_FORGE_SYS_VERSION_RE = re.compile(
    r"^(?P<version>\d+\.\d+(?:\.\d+)?)\s+\|\s+packaged by conda-forge\s+\|\s+"
    r"\((?P<buildno>[^,]+),\s+(?P<builddate>[A-Za-z]{3}\s+\d{1,2}\s+\d{4}),\s+"
    r"(?P<buildtime>\d{2}:\d{2}:\d{2})\)\s+\[(?P<compiler>[^\]]+)\]$"
)


def patch_conda_forge_sys_version_for_isaaclab() -> None:
    """Teach Isaac Sim's bundled platform.py about conda-forge sys.version strings."""
    import platform

    try:
        platform._sys_version(sys.version)  # type: ignore[attr-defined]
        return
    except ValueError:
        pass

    match = _CONDA_FORGE_SYS_VERSION_RE.match(sys.version)
    if match is None:
        return

    if hasattr(sys, "_git"):
        _, branch, revision = sys._git
    elif hasattr(sys, "_mercurial"):
        _, branch, revision = sys._mercurial
    else:
        branch = ""
        revision = ""

    version = match.group("version")
    if len(version.split(".")) == 2:
        version = f"{version}.0"

    parsed = (
        "CPython",
        version,
        branch,
        revision,
        match.group("buildno"),
        f"{match.group('builddate')} {match.group('buildtime')}",
        match.group("compiler"),
    )
    platform._sys_version_cache[sys.version] = parsed  # type: ignore[attr-defined]
