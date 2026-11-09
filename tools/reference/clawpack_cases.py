"""Case definitions and accepted-state checks for external Clawpack comparisons."""

from dataclasses import dataclass, field

import numpy as np

STEP = 0.01
END = 4.0
GRAVITY = 9.81


@dataclass(frozen=True)
class Case:
    name: str
    downstream: float
    method: str
    allows_dry: bool = False

    def metadata(self, cells):
        return {
            "case": self.name,
            "cells": cells,
            "time": END,
            "gravity": GRAVITY,
            "left_depth": 1.0,
            "right_depth": self.downstream,
            "xmin": -100.0,
            "xmax": 100.0,
            "max_step": STEP,
            "method": self.method,
        }


CASES = {
    "wet-quarter": Case("wet-quarter", 0.25, "pyclaw-python-hll-order1"),
    "wet-tenth": Case("wet-tenth", 0.1, "pyclaw-python-hll-order1"),
    "dry-ritter": Case("dry-ritter", 0.0, "pyclaw-fortran-augmented-order1", True),
}


@dataclass
class AcceptedStates:
    case: Case
    times: list = field(default_factory=list)
    min_depth: float = float("inf")
    max_abs_qy: float = 0.0

    def observe(self, state):
        if state.q.dtype != np.float64:
            raise ValueError(
                f"PyClaw dtype={state.q.dtype}: require numpy.float64 fields"
            )
        if not np.isfinite(state.q).all() or not (state.q[0] >= 0.0).all():
            raise ValueError(
                "PyClaw state contains nonfinite fields or negative water: reject this reference run"
            )
        dry = state.q[0] == 0.0
        if not self.case.allows_dry and dry.any():
            raise ValueError(
                f"PyClaw case={self.case.name} contains dry cells: require strictly positive wet-case depth"
            )
        if not (state.q[1:, dry] == 0.0).all():
            raise ValueError(
                "PyClaw zero-depth state has discharge: reject this reference run"
            )
        qy = float(np.max(np.abs(state.q[2]))) if state.num_eqn == 3 else 0.0
        if qy > 1e-12:
            raise ValueError(f"PyClaw |qy|={qy}: require at most 1e-12 m²/s")
        self.max_abs_qy = max(self.max_abs_qy, qy)
        self.min_depth = min(self.min_depth, float(np.min(state.q[0])))
        time = float(state.t)
        if self.times and time < self.times[-1]:
            raise ValueError(
                f"PyClaw callback time={time}: require nondecreasing accepted-state times"
            )
        if not self.times or time > self.times[-1]:
            self.times.append(time)

    def intervals(self, status):
        intervals = np.diff(self.times)
        if len(intervals) != status["numsteps"] or not (intervals > 0.0).all():
            raise ValueError(
                f"PyClaw accepted interval count={len(intervals)}: require status numsteps={status['numsteps']}"
            )
        if float(np.max(intervals)) > STEP + 1e-12:
            raise ValueError(
                f"PyClaw accepted timestep={np.max(intervals)}: require max_step={STEP}"
            )
        return intervals
