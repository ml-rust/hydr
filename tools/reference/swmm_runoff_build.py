"""Build the untouched EPA runoff component against its original solver dependencies."""

import json
import shutil
import subprocess
from pathlib import Path

from swmm_runoff_sources import adapter_hashes
from swmm_soil_sources import (
    INCLUDE_ENVIRONMENT,
    clean_environment,
    fresh,
    sha256,
    source_identity,
)


def build(args):
    source = args.source_root.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    identity = source_identity(source, archive)
    executable = shutil.which(args.cc)
    if executable is None:
        raise ValueError(
            f"Compiler={args.cc}: supply an available C compiler executable"
        )
    compiler = Path(executable).resolve(strict=True)
    compiler_hash = sha256(compiler)
    adapters = adapter_hashes()
    environment = clean_environment()
    version = subprocess.check_output(
        [str(compiler), "--version"], env=environment, text=True
    )
    directory = fresh(args.build_dir)
    directory.mkdir(parents=True)
    binary = directory / "swmm-runoff-reference"
    driver = Path(__file__).with_name("swmm_runoff_driver.c").resolve(strict=True)
    solver = source / "src/solver"
    included = {"subcatch.c", "infil.c"}
    linked = sorted(path for path in solver.glob("*.c") if path.name not in included)
    if not linked or any(
        str(path.relative_to(source)) not in identity["source_sha256"]
        for path in linked
    ):
        raise ValueError(
            "Solver sources differ from the pinned archive: restore original files"
        )
    # Original CMake leaves GNU C extensions enabled. Expose its POSIX interfaces under strict C11.
    command = [
        str(compiler),
        "-std=c11",
        "-D_GNU_SOURCE",
        "-O2",
        "-ffp-contract=off",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    command.extend(
        "-D" + macro + "=" + json.dumps(str(solver / filename))
        for macro, filename in (
            ("SWMM_SUBCATCH_SOURCE", "subcatch.c"),
            ("SWMM_INFIL_SOURCE", "infil.c"),
            ("SWMM_ODE_HEADER", "odesolve.h"),
        )
    )
    command.extend(["-I", str(solver / "include"), str(driver)])
    command.extend(str(path) for path in linked)
    command.extend(["-Wl,--gc-sections", "-lm", "-o", str(binary)])
    with (directory / "build.log").open("x") as stream:
        process = subprocess.run(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
    if (
        source_identity(source, archive) != identity
        or sha256(compiler) != compiler_hash
        or adapter_hashes() != adapters
    ):
        raise ValueError(
            "Build inputs changed during compilation: preserve a stable source/compiler/adapter envelope"
        )
    if process.returncode != 0:
        raise ValueError(
            f"Native runoff build exit={process.returncode}: read {directory / 'build.log'}"
        )
    manifest = {
        "scope": "Standalone original subcatchment/soil/ODE component, not the full SWMM driver",
        "source": identity,
        "compiler": str(compiler),
        "compiler_sha256": compiler_hash,
        "compiler_version": version,
        "command": command,
        "removed_environment_keys": INCLUDE_ENVIRONMENT,
        "adapter_sha256": adapters,
        "driver_sha256": sha256(driver),
        "binary_sha256": sha256(binary),
        "compiled_adapter_sha256": {
            name: digest
            for name, digest in adapters.items()
            if Path(name).suffix in (".c", ".h")
        },
        "included_original_sources": ["src/solver/subcatch.c", "src/solver/infil.c"],
        "linked_original_sources": [str(path.relative_to(source)) for path in linked],
        "unit_bridge": "Only included infil.c UCF calls map to a native-input identity helper; original engine UCF remains unchanged elsewhere",
        "native_factors": {
            "InfilFactor": 1.0,
            "Adjust.hydconFactor": 1.0,
            "Evap.recoveryFactor": 1.0,
        },
        "ode_provenance": "Pinned odesolve.c attributes fifth-order adaptive Runge-Kutta/controller code to Numerical Recipes in C, Cambridge University Press, 1992",
        "exit_code": 0,
    }
    with (directory / "build-manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Built untouched EPA runoff reference at {binary}")
