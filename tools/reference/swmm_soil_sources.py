"""Check the pinned EPA archive and original source tree before native execution."""

import hashlib
import os
import tarfile
from pathlib import Path

COMMIT = "7952ca837988b1c32f791812eccc9fd64547e093"
RELEASE = "v5.2.4"
ARCHIVE_SHA256 = "0c882c315b64d5236b3569f891e9ad71421bc5b7d92a5710ceb1ae3b7f34c6cf"
ARCHIVE_URL = (
    f"https://codeload.github.com/USEPA/Stormwater-Management-Model/tar.gz/{COMMIT}"
)
INCLUDE_ENVIRONMENT = (
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "INCLUDE",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
    "LIBRARY_PATH",
    "DEPENDENCIES_OUTPUT",
    "SUNPRO_DEPENDENCIES",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(source, archive):
    source = source.resolve(strict=True)
    archive = archive.resolve(strict=True)
    actual_archive = sha256(archive)
    if actual_archive != ARCHIVE_SHA256:
        raise ValueError(
            f"Archive={archive} SHA256={actual_archive}: require pinned {ARCHIVE_SHA256}"
        )
    hashes = {}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            if not member.isfile():
                continue
            prefix, relative = member.name.split("/", 1)
            if prefix != f"Stormwater-Management-Model-{COMMIT}":
                raise ValueError(
                    f"Archive member={member.name}: require the pinned source prefix"
                )
            current = source / relative
            if not current.is_file() or current.is_symlink():
                raise ValueError(
                    f"Source={current}: require the original regular archive file"
                )
            native = stream.extractfile(member)
            if native is None:
                raise ValueError(
                    f"Archive member={member.name}: require readable source bytes"
                )
            expected = hashlib.sha256(native.read()).hexdigest()
            actual = sha256(current)
            if actual != expected:
                raise ValueError(
                    f"Source={current} SHA256={actual}: restore pinned bytes {expected}"
                )
            hashes[relative] = actual
    if "src/solver/infil.c" not in hashes or "README.md" not in hashes:
        raise ValueError(
            f"Source={source}: require original infiltration code and provenance README"
        )
    return {
        "release": RELEASE,
        "commit": COMMIT,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": actual_archive,
        "source_root": str(source),
        "source_file_count": len(hashes),
        "source_sha256": dict(sorted(hashes.items())),
        "reuse_statement": "Pinned EPA README states that SWMM source is released in the Public Domain",
        "reuse_statement_file": "README.md",
    }


def adapter_hashes():
    return {
        path.name: sha256(path)
        for path in sorted(Path(__file__).parent.glob("swmm_soil*"))
        if path.is_file() and path.suffix in (".py", ".c")
    }


def clean_environment():
    environment = os.environ.copy()
    for key in INCLUDE_ENVIRONMENT:
        environment.pop(key, None)
    return environment


def fresh(path):
    path = path.absolute()
    if os.path.lexists(path):
        raise FileExistsError(
            f"Destination={path}: choose a fresh path without existing evidence"
        )
    return path
