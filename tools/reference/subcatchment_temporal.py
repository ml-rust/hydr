"""Freeze and check independent smooth routed references against public hydr observations."""

import argparse
import json
import platform
import sys
from pathlib import Path

from subcatchment_temporal_cases import frozen_specification
from subcatchment_temporal_compare import (
    compare_smooth,
    history_refinement,
    native_departures,
    read_observer,
)
from subcatchment_temporal_reference import (
    check_reference_convergence,
    integrate,
    write_references,
)
from swmm_soil_sources import fresh, sha256


def tool_identity():
    return {
        path.name: sha256(path)
        for path in sorted(Path(__file__).parent.glob("subcatchment_temporal*.py"))
    }


def freeze(args):
    output = fresh(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(
            {"specification": frozen_specification(), "tools_sha256": tool_identity()},
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(f"Wrote pre-execution physical specification to {output}")


def run(args):
    frozen = json.loads(args.frozen_spec.read_text())
    current = json.loads(json.dumps(frozen_specification()))
    if frozen["specification"] != current or frozen["tools_sha256"] != tool_identity():
        raise ValueError(
            "Frozen physical inputs, gates, or reference tools differ. Freeze before execution"
        )
    before = {
        "tools": tool_identity(),
        "frozen_spec_sha256": sha256(args.frozen_spec),
        "observer_sha256": sha256(args.observer),
        "native_sha256": None if args.native_csv is None else sha256(args.native_csv),
    }
    rows = read_observer(args.observer)
    directory = fresh(args.output_dir)
    directory.mkdir(parents=True)
    references, execution = integrate()
    write_references(directory, references)
    reference_difference = check_reference_convergence(references)
    smooth = compare_smooth(rows, references)
    histories = history_refinement(rows)
    native = native_departures(rows, args.native_csv)
    after = {
        "tools": tool_identity(),
        "frozen_spec_sha256": sha256(args.frozen_spec),
        "observer_sha256": sha256(args.observer),
        "native_sha256": None if args.native_csv is None else sha256(args.native_csv),
    }
    if after != before:
        raise ValueError(
            "Reference or observation inputs changed during comparison. Retain output and use stable inputs"
        )
    report = {
        "scope": "Smooth routed physical accuracy and separate nonlinear-history/native departures",
        "identity_before": before,
        "identity_after": after,
        "specification": current,
        "python": sys.version,
        "platform": platform.platform(),
        "reference_execution": execution,
        "reference_maximum_normalized_differences": reference_difference,
        "smooth_refinement": smooth,
        "historical_refinement": histories,
        "native_departures": native,
        "generated_sha256": {
            path.name: sha256(path) for path in sorted(directory.glob("*.csv"))
        },
        "limits": "No arbitrary nonlinear-soil-history, field-accuracy, or cross-platform bitwise claim",
    }
    with (directory / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        f"Checked twelve smooth grids and {len(native)} retained native departures in {directory}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(function=freeze)
    compare = commands.add_parser("run")
    compare.add_argument("--frozen-spec", type=Path, required=True)
    compare.add_argument("--observer", type=Path, required=True)
    compare.add_argument("--native-csv", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(function=run)
    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
