"""Runoff adapter identity, sharing the pinned EPA source checks with soil tooling."""

from pathlib import Path

from swmm_soil_sources import sha256


def adapter_hashes():
    directory = Path(__file__).parent
    paths = sorted(directory.glob("swmm_runoff*"))
    paths.append(directory / "swmm_soil_sources.py")
    return {
        path.name: sha256(path)
        for path in paths
        if path.is_file() and path.suffix in (".py", ".c", ".h")
    }
