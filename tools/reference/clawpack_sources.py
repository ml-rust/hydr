"""Load original Clawpack modules and record their source and binary identities."""

import contextlib
import hashlib
import importlib
import sys
import types
from pathlib import Path


def load_solver(pyclaw_root, riemann_root, compiled_root, case):
    for path in (pyclaw_root / "src/pyclaw", riemann_root / "riemann"):
        if not path.is_dir():
            raise ValueError(
                f"Source directory missing at {path}: supply extracted upstream sources"
            )
    namespace = types.ModuleType("clawpack")
    namespace.__path__ = [str(pyclaw_root / "src")]
    riemann = types.ModuleType("clawpack.riemann")
    riemann.__path__ = [str(riemann_root / "riemann")]
    namespace.riemann = riemann
    sys.modules["clawpack"] = namespace
    sys.modules["clawpack.riemann"] = riemann
    with contextlib.redirect_stdout(sys.stderr):
        importlib.import_module("clawpack.riemann.static")
        pyclaw = importlib.import_module("clawpack.pyclaw")
        if case.allows_dry:
            if compiled_root is None:
                raise ValueError(
                    "dry-ritter lacks --compiled-root: supply original classic2 and sw_aug_2D extensions"
                )
            classic = importlib.import_module("clawpack.pyclaw.classic")
            classic.__path__.append(str(compiled_root))
            riemann.__path__.append(str(compiled_root))
            solver = importlib.import_module("clawpack.riemann.sw_aug_2D")
        else:
            solver = importlib.import_module(
                "clawpack.riemann.shallow_1D_py"
            ).shallow_hll_1D
    return pyclaw, solver


def source_hashes(roots, dry):
    paths = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        for label, root in roots.items():
            if path.is_relative_to(root) and path.is_file():
                paths[f"{label}/{path.relative_to(root)}"] = path
    for label in ("pyclaw", "riemann"):
        root = roots[label]
        for name in ("LICENSE", "LICENSE.txt"):
            path = root / name
            if path.is_file():
                paths[f"{label}/{name}"] = path
    if dry:
        for name in ("limiter", "philim", "flux2", "step2ds", "step2"):
            relative = f"src/pyclaw/classic/{name}.f90"
            paths[f"pyclaw/{relative}"] = roots["pyclaw"] / relative
        for name in ("rpn2_sw_aug", "rpt2_sw_aug"):
            relative = f"src/{name}.f90"
            paths[f"riemann/{relative}"] = roots["riemann"] / relative
        manifest = roots["compiled"] / "build-manifest.json"
        if manifest.is_file():
            paths["compiled/build-manifest.json"] = manifest
        original = roots["compiled"] / "original-source-sha256.json"
        if original.is_file():
            paths["compiled/original-source-sha256.json"] = original
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    }


def adapter_hashes(entry):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(entry.parent.glob("clawpack_*.py"))
    }
