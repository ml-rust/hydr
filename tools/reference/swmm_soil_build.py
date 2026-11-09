"""Build only the native soil adapter against unchanged EPA source."""

import json
import shutil
import subprocess
from pathlib import Path

from swmm_soil_sources import (
    INCLUDE_ENVIRONMENT,
    adapter_hashes,
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
    binary = directory / "swmm-soil-reference"
    driver = Path(__file__).with_name("swmm_soil_driver.c").resolve(strict=True)
    command = [
        str(compiler),
        "-std=c11",
        "-O2",
        "-ffp-contract=off",
        "-ffunction-sections",
        "-fdata-sections",
        "-DSWMM_SOIL_SOURCE=" + json.dumps(str(source / "src/solver/infil.c")),
        str(driver),
        "-Wl,--gc-sections",
        "-lm",
        "-o",
        str(binary),
    ]
    with (directory / "build.log").open("x") as stream:
        subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    after = source_identity(source, archive)
    if (
        after != identity
        or sha256(compiler) != compiler_hash
        or adapter_hashes() != adapters
    ):
        raise ValueError(
            "Build inputs changed during compilation: preserve a stable source/compiler envelope"
        )
    manifest = {
        "source": identity,
        "compiler": str(compiler),
        "compiler_sha256": compiler_hash,
        "compiler_version": version,
        "command": command,
        "removed_environment_keys": INCLUDE_ENVIRONMENT,
        "adapter_sha256": adapters,
        "binary_sha256": sha256(binary),
        "driver_sha256": sha256(driver),
        "exit_code": 0,
        "native_factors": {"UCF": 1.0, "InfilFactor": 1.0, "recoveryFactor": 1.0},
    }
    with (directory / "build-manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Built untouched EPA soil reference at {binary}")
