"""Build the pinned EPA SWMM v5.2.4 drainage worker against untouched solver sources."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from swmm_soil_sources import (
    INCLUDE_ENVIRONMENT,
    clean_environment,
    fresh,
    sha256,
    source_identity,
)

INCLUDED_SOURCES = (
    ("SWMM_SWMM5_SOURCE", "swmm5.c"),
    ("SWMM_ROUTING_SOURCE", "routing.c"),
    ("SWMM_DYNWAVE_SOURCE", "dynwave.c"),
    ("SWMM_CONTROLS_SOURCE", "controls.c"),
    ("SWMM_MASSBAL_SOURCE", "massbal.c"),
)


def build(source_root: Path, archive: Path, build_dir: Path, cc: str = "gcc") -> Path:
    source = source_root.resolve(strict=True)
    archive_path = archive.resolve(strict=True)
    identity = source_identity(source, archive_path)
    executable = shutil.which(cc)
    if executable is None:
        raise ValueError(f"Compiler={cc}: supply an available C compiler executable")
    compiler = Path(executable).resolve(strict=True)
    compiler_hash = sha256(compiler)
    environment = clean_environment()
    version = subprocess.check_output(
        [str(compiler), "--version"], env=environment, text=True
    )
    directory = fresh(build_dir)
    directory.mkdir(parents=True)
    binary = directory / "swmm-drainage-worker"
    driver = Path(__file__).with_name("swmm_drainage_driver.c").resolve(strict=True)
    solver = source / "src/solver"
    included_names = {name for _, name in INCLUDED_SOURCES}
    linked = sorted(
        path for path in solver.glob("*.c") if path.name not in included_names
    )
    if not linked or any(
        str(path.relative_to(source)) not in identity["source_sha256"]
        for path in linked
    ):
        raise ValueError(
            "Solver sources differ from the pinned archive: restore original files"
        )
    command = [
        str(compiler),
        "-std=c11",
        "-D_GNU_SOURCE",
        "-O2",
        "-ffp-contract=off",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    for macro, filename in INCLUDED_SOURCES:
        command.append(f"-D{macro}={json.dumps(str(solver / filename))}")
    command.extend(["-I", str(solver), "-I", str(solver / "include"), str(driver)])
    command.extend(str(path) for path in linked)
    command.extend(["-Wl,--gc-sections", "-lm", "-o", str(binary)])
    with (directory / "build.log").open("x") as stream:
        process = subprocess.run(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
    if source_identity(source, archive_path) != identity or sha256(compiler) != compiler_hash:
        raise ValueError("Build inputs changed during compilation")
    if process.returncode != 0:
        log_text = (directory / "build.log").read_text()
        raise ValueError(
            f"Native drainage worker build exit={process.returncode}:\n{log_text}"
        )
    manifest = {
        "scope": "Pinned EPA SWMM v5.2.4 dynamic-wave drainage worker with complete hidden-state snapshot/restore",
        "source": identity,
        "compiler": str(compiler),
        "compiler_sha256": compiler_hash,
        "compiler_version": version,
        "driver_sha256": sha256(driver),
        "binary_sha256": sha256(binary),
        "removed_environment_keys": INCLUDE_ENVIRONMENT,
        "included_original_sources": [f"src/solver/{n}" for _, n in INCLUDED_SOURCES],
        "linked_original_sources": [str(p.relative_to(source)) for p in linked],
        "exit_code": 0,
    }
    with (directory / "build-manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cc", default="gcc")
    args = parser.parse_args()
    binary = build(args.source_root, args.archive, args.build_dir, args.cc)
    print(f"Built pinned SWMM v5.2.4 drainage worker at {binary}")


if __name__ == "__main__":
    main()
