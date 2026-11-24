"""Verify pinned SWMM v5.2.4 drainage worker: hidden-state rollback, cold restart, surcharge, and continuity."""

import struct
import subprocess
import tempfile
from pathlib import Path

from swmm_drainage_build import build

SWMM_ROOT = Path("resource/reference-engines/swmm")
SOURCE_ROOT = SWMM_ROOT / "Stormwater-Management-Model-7952ca837988b1c32f791812eccc9fd64547e093"
ARCHIVE = SWMM_ROOT / "swmm-v5.2.4.tar.gz"
BUILD_DIR = SWMM_ROOT / "build-drainage-worker"


def f64_hex(val: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', val))[0]:016x}"


def hex_f64(h: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def make_inp(surcharge_method: str, inflow_cfs: float, with_controls: bool) -> str:
    controls = """
[CONTROLS]
RULE R1
IF NODE J1 DEPTH > 2.5
THEN ORIFICE OR1 SETTING = 0.5
ELSE ORIFICE OR1 SETTING = 1.0
PRIORITY 1

RULE R2
IF LINK OR1 TIMEOPEN >= 0.005
AND NODE J1 DEPTH > 1.0
THEN PUMP P1 STATUS = ON
ELSE PUMP P1 STATUS = OFF
PRIORITY 2
""" if with_controls else ""
    pump_section = """
[PUMPS]
P1 J2 OUT2 PC1 ON 0.5 0.1

[CURVES]
PC1 PUMP2 0.0 0.0
PC1 1.0 2.0
PC1 5.0 5.0
""" if with_controls else ""
    outfall2 = "OUT2 6.0 FREE NO\n" if with_controls else ""
    orifice_section = """
[ORIFICES]
OR1 J1 J2 SIDE 0.0 0.65 NO 0.002
""" if with_controls else ""
    xsect_orifice = "OR1 CIRCULAR 1.5 0 0 0\n" if with_controls else ""
    return f"""[TITLE]
Pinned SWMM v5.2.4 drainage rollback and surcharge check

[OPTIONS]
FLOW_UNITS           CFS
INFILTRATION         MODIFIED_GREEN_AMPT
FLOW_ROUTING         DYNWAVE
SURCHARGE_METHOD     {surcharge_method}
START_DATE           01/01/2026
START_TIME           00:00:00
END_DATE             01/01/2026
END_TIME             01:00:00
REPORT_STEP          00:01:00
ROUTING_STEP         0.5
VARIABLE_STEP        0.75
MINIMUM_STEP         0.05
MAX_TRIALS           20
HEAD_TOLERANCE       0.001
RULE_STEP            00:00:02
IGNORE_RAINFALL      YES
IGNORE_SNOWMELT      YES
IGNORE_GROUNDWATER   YES
IGNORE_QUALITY       YES
ALLOW_PONDING        NO
MIN_SURFAREA         12.566

[JUNCTIONS]
J1 10.0 10.0 0.5 0.0 0.0
J2 8.0 10.0 0.5 0.0 0.0

[OUTFALLS]
OUT1 6.0 FREE NO
{outfall2}

[CONDUITS]
C1 J1 J2 200.0 0.013 0.0 0.0 0.0 0.0
C2 J2 OUT1 200.0 0.013 0.0 0.0 0.0 0.0

[XSECTIONS]
C1 CIRCULAR 2.0 0 0 0 1
C2 CIRCULAR 2.0 0 0 0 1
{xsect_orifice}
{orifice_section}
{pump_section}
[INFLOWS]
J1 FLOW "" FLOW 1.0 1.0 {inflow_cfs}
{controls}
"""


class Worker:
    def __init__(self, binary: Path):
        self.proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def cmd(self, line: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().rstrip("\n")
        if resp.startswith("ERR"):
            raise RuntimeError(f"Worker error on {line!r}: {resp}")
        return resp

    def state(self) -> tuple[str, list[str]]:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write("STATE\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            ln = self.proc.stdout.readline().rstrip("\n")
            if ln == "END_STATE":
                break
            lines.append(ln)
        return ("\n".join(lines), lines)

    def close(self) -> None:
        self.cmd("CLOSE")
        self.proc.wait(timeout=5)


def run_verification() -> None:
    binary = BUILD_DIR / "swmm-drainage-worker"
    if not binary.exists():
        binary = build(SOURCE_ROOT, ARCHIVE, BUILD_DIR)

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        inp = td / "controls.inp"
        inp.write_text(make_inp("EXTRAN", 12.0, True))

        # 1. Uninterrupted reference run (0 -> 120 s in 2 s steps).
        w_ref = Worker(binary)
        w_ref.cmd(f"OPEN {inp} {td / 'ref.rpt'}")
        ref_states = {}
        snap_40 = None
        for step_idx in range(1, 61):
            w_ref.cmd(f"STEP {f64_hex(2.0)}")
            st_text, _ = w_ref.state()
            ref_states[step_idx] = st_text
            if step_idx == 20:
                snap_40 = w_ref.cmd("SNAPSHOT").split(" ", 1)[1]
        w_ref.close()
        assert snap_40 is not None

        # 2. In-process rollback proof: advance to 40 s, snapshot, inject huge perturbing inflow
        #    and advance 15 steps (triggering surcharge, control switching, orifice travel),
        #    then RESTORE t = 40 s and continue to 120 s. Must match ref bit-for-bit!
        w_rb = Worker(binary)
        w_rb.cmd(f"OPEN {inp} {td / 'rb.rpt'}")
        for _ in range(20):
            w_rb.cmd(f"STEP {f64_hex(2.0)}")
        snap = w_rb.cmd("SNAPSHOT").split(" ", 1)[1]
        assert snap == snap_40, "snapshot at t=40 s must be deterministic"
        w_rb.cmd(f"SET_INFLOW 0 {f64_hex(80.0)}")
        for _ in range(15):
            w_rb.cmd(f"STEP {f64_hex(2.0)}")
        perturbed_text, _ = w_rb.state()
        assert perturbed_text != ref_states[35], "perturbation must visibly alter state"
        w_rb.cmd(f"RESTORE {snap}")
        for step_idx in range(21, 61):
            w_rb.cmd(f"STEP {f64_hex(2.0)}")
            st_text, _ = w_rb.state()
            assert st_text == ref_states[step_idx], f"rollback mismatch at step {step_idx}"
        w_rb.close()

        # 3. Cold-process restart proof: fresh worker process restores snap_40 and runs to 120 s.
        w_cold = Worker(binary)
        w_cold.cmd(f"OPEN {inp} {td / 'cold.rpt'}")
        w_cold.cmd(f"RESTORE {snap_40}")
        for step_idx in range(21, 61):
            w_cold.cmd(f"STEP {f64_hex(2.0)}")
            st_text, _ = w_cold.state()
            assert st_text == ref_states[step_idx], f"cold restart mismatch at step {step_idx}"
        w_cold.close()

        # 4. Surcharge treatment comparison (EXTRAN vs SLOT) under pressurizing inflow (23.5 cfs > 22.6 cfs full-pipe).
        surcharge_results = {}
        for method in ("EXTRAN", "SLOT"):
            s_inp = td / f"surcharge_{method.lower()}.inp"
            s_inp.write_text(make_inp(method, 23.5, False))
            w = Worker(binary)
            w.cmd(f"OPEN {s_inp} {td / f'surcharge_{method.lower()}.rpt'}")
            max_depth = 0.0
            for _ in range(180):
                w.cmd(f"STEP {f64_hex(2.0)}")
                _, lines = w.state()
                j1_depth = hex_f64(lines[1].split()[5])
                max_depth = max(max_depth, j1_depth)
            _, lines = w.state()
            pct_err = hex_f64(lines[0].split()[10])
            w.close()
            assert max_depth > 2.0, f"{method}: expected pressurized depth > 2.0 ft, got {max_depth}"
            assert abs(pct_err) < 5.0, f"{method}: continuity error too large: {pct_err}%"
            surcharge_results[method] = (max_depth, pct_err)

        # 5. External coupled inflow injection (SET_INFLOW) continuity check.
        ext_inp = td / "ext_inflow.inp"
        ext_inp.write_text(make_inp("SLOT", 0.0, False))
        w_ext = Worker(binary)
        w_ext.cmd(f"OPEN {ext_inp} {td / 'ext_inflow.rpt'}")
        w_ext.cmd(f"SET_INFLOW 0 {f64_hex(10.0)}")
        for _ in range(300):
            w_ext.cmd(f"STEP {f64_hex(2.0)}")
        _, lines = w_ext.state()
        ext_in = hex_f64(lines[0].split()[4])
        ext_err = hex_f64(lines[0].split()[10])
        w_ext.close()
        assert abs(ext_in - 6000.0) < 10.0, f"expected ~6000 ft3 injected, got {ext_in}"
        assert abs(ext_err) < 3.0, f"SET_INFLOW continuity error too large: {ext_err}%"

        print("D1 VERIFICATION PASSED:")
        print("  - In-process rollback (hydraulics + Xnode + VariableStep + controls + orifice): BIT-IDENTICAL")
        print("  - Cold-process restart from snapshot: BIT-IDENTICAL")
        for method, (md, pe) in surcharge_results.items():
            print(f"  - Surcharge {method}: max_depth={md:.4f} ft (> 2.0 ft crown), continuity_err={pe:.4f}%")
        print(f"  - Coupled SET_INFLOW: injected={ext_in:.2f} ft3, continuity_err={ext_err:.4f}%")


if __name__ == "__main__":
    run_verification()
