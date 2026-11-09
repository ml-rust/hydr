"""Construct and advance original Clawpack solvers for matched dam cases."""

import json
import math
import os

import numpy as np
from clawpack_cases import CASES, END, GRAVITY, STEP, AcceptedStates
from clawpack_output import prepare_output, write_output
from clawpack_sources import load_solver


def setup(pyclaw, riemann_solver, case, cells):
    if case.allows_dry:
        solver = pyclaw.ClawSolver2D(riemann_solver)
        solver.kernel_language = "Fortran"
        solver.dimensional_split = True
        solver.fwave = True
        solver.num_eqn = 3
        solver.num_waves = 3
        solver.transverse_waves = 0
        solver.aux_bc_lower = [pyclaw.BC.extrap] * 2
        solver.aux_bc_upper = [pyclaw.BC.extrap] * 2
        domain = pyclaw.Domain(
            [
                pyclaw.Dimension(-100.0, 100.0, cells, name="x"),
                pyclaw.Dimension(0.0, 1.0, 1, name="y"),
            ]
        )
        state = pyclaw.State(domain, 3, 1)
        state.aux[:] = 0.0
        state.q[0, :, 0] = np.where(state.grid.x.centers < 0.0, 1.0, case.downstream)
        state.q[1:] = 0.0
    else:
        solver = pyclaw.ClawSolver1D(riemann_solver)
        solver.kernel_language = "Python"
        solver.num_eqn = 2
        solver.num_waves = 2
        solver.fwave = False
        domain = pyclaw.Domain(pyclaw.Dimension(-100.0, 100.0, cells, name="x"))
        state = pyclaw.State(domain, 2)
        state.q[0] = np.where(state.grid.x.centers < 0.0, 1.0, case.downstream)
        state.q[1] = 0.0
    solver.order = 1
    solver.all_bcs = pyclaw.BC.wall
    solver.cfl_desired = 0.4
    solver.cfl_max = 0.45
    solver.dt_initial = STEP
    solver.dt = STEP
    solver.dt_max = STEP
    state.problem_data["grav"] = GRAVITY
    return solver, state, domain


def run(args, entry):
    case = CASES[args.case]
    roots = {
        "pyclaw": args.pyclaw_root.resolve(strict=True),
        "riemann": args.riemann_root.resolve(strict=True),
    }
    compiled = args.compiled_root.resolve(strict=True) if args.compiled_root else None
    if case.allows_dry and compiled is not None:
        roots["compiled"] = compiled
    output = prepare_output(args.output)
    # Upstream package logging creates its file in the current directory.
    os.chdir(output.parent)
    pyclaw, riemann_solver = load_solver(
        roots["pyclaw"], roots["riemann"], compiled, case
    )
    np.seterr(divide="raise", invalid="raise", over="raise")
    solver, state, domain = setup(pyclaw, riemann_solver, case, args.cells)
    monitor = AcceptedStates(case)
    monitor.observe(state)
    solver.before_step = lambda _, accepted: monitor.observe(accepted)
    initial_storage = float(np.sum(state.q[0]) * 200.0 / args.cells)
    solution = pyclaw.Solution(state, domain)
    backend = {"kernel_language": solver.kernel_language}
    if case.allows_dry:
        build_manifest = compiled / "build-manifest.json"
        backend["build_manifest"] = json.loads(build_manifest.read_text())
        solver.setup(solution)
        actual_gravity = float(riemann_solver.cparam.grav)
        if actual_gravity != GRAVITY:
            raise ValueError(
                f"Fortran cparam gravity={actual_gravity}: require {GRAVITY}"
            )
        backend["fortran_gravity_m_per_s2"] = actual_gravity
    status = solver.evolve_to_time(solution, END)
    monitor.observe(solution.state)
    if not math.isclose(solution.t, END, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"PyClaw endpoint={solution.t}: require completed horizon {END}"
        )
    monitor.intervals(status)
    write_output(
        output,
        entry,
        case,
        args.cells,
        pyclaw,
        solution,
        monitor,
        status,
        initial_storage,
        roots,
        backend,
    )
