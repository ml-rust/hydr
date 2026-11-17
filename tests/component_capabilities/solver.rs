//! Surface solver declaration checked against its admission code.

use crate::fixture::{create, interval, through, wet};
use crate::runtime::Runtime;
use hydr::capability::{
    BoundaryKind, EndpointId, Regime, RegimeScope, RegimeSupport, UnsupportedReason,
};
use hydr::model::CellState;
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryFace, BoundaryForcing, DischargeBoundary, EngineError,
    ExteriorBoundary, ExteriorState, FluxPort, Solver, SolverConfig, StageBoundary,
    surface_solver_capabilities,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Case = Result<Solver<Runtime>, Box<dyn std::error::Error>>;

/// Right exterior face of a one-cell grid. Its outward normal is positive x.
const RIGHT: BoundaryFace = BoundaryFace {
    axis: Axis::X,
    column: 1,
    row: 0,
};

const GRAVITY: f64 = 9.81;

fn cell(h: f64, qx: f64) -> CellState {
    CellState { h, qx, qy: 0.0 }
}

fn config() -> SolverConfig {
    SolverConfig {
        max_step: 0.01,
        ..SolverConfig::default()
    }
}

/// One cell with state `state` and `boundary` on its right face. `None` leaves every face a wall.
fn one_cell(state: CellState, boundary: Option<BoundaryForcing>) -> Case {
    create(
        1,
        1,
        vec![0.0],
        vec![state],
        vec![interval(0.0, 1.0, 1, boundary.into_iter().collect())],
        config(),
    )
}

fn stage(elevation: f64) -> BoundaryForcing {
    BoundaryForcing::Stage(StageBoundary {
        face: RIGHT,
        elevation,
        inflow_tangential_velocity: Some(0.0),
    })
}

fn discharge(outward_rate: f64) -> BoundaryForcing {
    BoundaryForcing::Discharge(DischargeBoundary {
        face: RIGHT,
        outward_rate,
        inflow_tangential_velocity: Some(0.0),
    })
}

fn dry_exterior() -> BoundaryForcing {
    BoundaryForcing::Exterior(ExteriorBoundary {
        face: RIGHT,
        state: ExteriorState::Dry,
    })
}

/// Resting unit-depth lake. The port carries only hydrostatic pressure `g*h²/2`.
fn resting_port() -> BoundaryForcing {
    BoundaryForcing::Flux(FluxPort {
        face: RIGHT,
        water_rate: 0.0,
        momentum_rate: [GRAVITY / 2.0, 0.0],
        signal_speed: 4.0,
        provenance: "Resting h=1 lake, pressure g*h^2/2, sqrt(g*h)<4".into(),
    })
}

/// Two cells joined by an interior face with wet inflow on the left and outflow on the right.
fn two_cells(cells: [CellState; 2]) -> Case {
    let face = |column| BoundaryFace {
        axis: Axis::X,
        column,
        row: 0,
    };
    let inflow_velocity = cells[0].qx / cells[0].h;
    create(
        2,
        1,
        vec![0.0; 2],
        cells.to_vec(),
        vec![interval(
            0.0,
            1.0,
            2,
            vec![
                wet(face(0), cells[0].h, -inflow_velocity, 0.0),
                BoundaryForcing::Transmissive(face(2)),
            ],
        )],
        config(),
    )
}

/// Two walled cells: one wet and one dry, so a front crosses the interior face.
fn dam_onto_dry_bed() -> Case {
    create(
        2,
        1,
        vec![0.0; 2],
        vec![cell(1.0, 0.0), cell(0.0, 0.0)],
        vec![interval(0.0, 1.0, 2, vec![])],
        config(),
    )
}

/// Minimal case for each admitted exterior-face boundary kind.
fn boundary_case(kind: BoundaryKind) -> Option<Case> {
    let rest = cell(1.0, 0.0);
    Some(match kind {
        BoundaryKind::Wall => one_cell(rest, None),
        BoundaryKind::FluxPort => one_cell(rest, Some(resting_port())),
        BoundaryKind::Stage => one_cell(rest, Some(stage(1.0))),
        BoundaryKind::Discharge => one_cell(rest, Some(discharge(0.0))),
        BoundaryKind::Exterior => one_cell(rest, Some(wet(RIGHT, 1.0, 0.0, 0.0))),
        BoundaryKind::Transmissive => through(Axis::X, 1.0),
        _ => return None,
    })
}

/// Minimal case in which `scope` holds flow of `regime`.
fn regime_case(scope: RegimeScope, regime: Regime) -> Option<Case> {
    let rest = cell(1.0, 0.0);
    // Unit depth with outward velocity 4 m/s exceeds sqrt(g) and is supercritical.
    let fast = cell(1.0, 4.0);
    let dry = cell(0.0, 0.0);
    Some(match (scope, regime) {
        (RegimeScope::Boundary(BoundaryKind::Stage), Regime::WetSubcritical) => {
            one_cell(rest, Some(stage(1.0)))
        }
        (RegimeScope::Boundary(BoundaryKind::Stage), Regime::Supercritical) => {
            one_cell(fast, Some(stage(1.0)))
        }
        (RegimeScope::Boundary(BoundaryKind::Stage), Regime::Dry) => {
            one_cell(dry, Some(stage(1.0)))
        }
        (RegimeScope::Boundary(BoundaryKind::Discharge), Regime::WetSubcritical) => {
            one_cell(rest, Some(discharge(0.0)))
        }
        (RegimeScope::Boundary(BoundaryKind::Discharge), Regime::Supercritical) => {
            one_cell(fast, Some(discharge(4.0)))
        }
        (RegimeScope::Boundary(BoundaryKind::Exterior), Regime::WetSubcritical) => {
            one_cell(rest, Some(wet(RIGHT, 1.0, 0.0, 0.0)))
        }
        // The fixture's upstream exterior supplies 5 m/s inflow at unit depth.
        (RegimeScope::Boundary(BoundaryKind::Exterior), Regime::Supercritical) => {
            through(Axis::X, 1.0)
        }
        (RegimeScope::Boundary(BoundaryKind::Exterior), Regime::Dry) => {
            one_cell(rest, Some(dry_exterior()))
        }
        (RegimeScope::Boundary(BoundaryKind::Transmissive), Regime::Dry) => {
            one_cell(dry, Some(BoundaryForcing::Transmissive(RIGHT)))
        }
        (RegimeScope::Boundary(BoundaryKind::Transmissive), Regime::Supercritical) => {
            through(Axis::X, 1.0)
        }
        (RegimeScope::Boundary(BoundaryKind::Transmissive), Regime::WetSubcritical) => {
            one_cell(rest, Some(BoundaryForcing::Transmissive(RIGHT)))
        }
        (RegimeScope::Interior, Regime::WetSubcritical) => one_cell(rest, None),
        (RegimeScope::Interior, Regime::Supercritical) => {
            two_cells([cell(1.0, 5.0), cell(1.0, 5.0)])
        }
        (RegimeScope::Interior, Regime::Dry) => one_cell(dry, None),
        (RegimeScope::Interior, Regime::WetDryFront) => dam_onto_dry_bed(),
        _ => return None,
    })
}

/// Cases and message fragments for each `RejectedByAdmission` entry.
fn rejection_cases(id: &str) -> Option<Vec<(Case, &'static str)>> {
    let rest = cell(1.0, 0.0);
    Some(match id {
        "stage-outside-wet-subcritical" => vec![
            (
                one_cell(cell(1.0, 4.0), Some(stage(1.0))),
                "Interior is dry, critical, or supercritical",
            ),
            (
                one_cell(cell(0.0, 0.0), Some(stage(1.0))),
                "Dry interior has no admitted wet characteristic",
            ),
            // Subcritical 2 m/s outflow at unit depth expands to a supercritical 0.1 m trace.
            (
                one_cell(cell(1.0, 2.0), Some(stage(0.1))),
                "Requested trace is critical or supercritical",
            ),
        ],
        "discharge-outside-wet-subcritical" => vec![(
            one_cell(rest, Some(discharge(1000.0))),
            "Discharge lies outside the connected subcritical branch",
        )],
        "transmissive-subcritical-or-incoming" => vec![(
            one_cell(rest, Some(BoundaryForcing::Transmissive(RIGHT))),
            "Transmissive boundary has incoming or unresolved critical characteristics",
        )],
        _ => return None,
    })
}

/// Runs one step and returns the rejection reason, or `None` when the step is accepted.
fn rejection(case: Case) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut engine = case?;
    match engine.advance_to(0.01, 1) {
        Ok(_) => Ok(None),
        Err(EngineError::Rejected { reason, .. }) => Ok(Some(reason)),
        Err(other) => Err(format!("expected EngineError::Rejected, found {other}").into()),
    }
}

fn admits(case: Case) -> TestResult {
    let mut engine = case?;
    let report = engine.advance_to(0.01, 1)?;
    assert_eq!(report.status, AdvanceStatus::ReachedTarget);
    assert_eq!(report.accepted_steps, 1);
    Ok(())
}

#[test]
fn solver_admits_declared_boundaries() -> TestResult {
    let endpoint = surface_solver_capabilities()
        .endpoint(EndpointId::ExteriorFace)
        .ok_or("solver declares no exterior face")?;
    for kind in endpoint.admitted {
        let case = boundary_case(*kind).ok_or(format!("no case for `{}`", kind.id()))?;
        admits(case).map_err(|error| format!("`{}`: {error}", kind.id()))?;
    }
    Ok(())
}

#[test]
fn solver_rejects_declared_rejections() -> TestResult {
    let rejected = surface_solver_capabilities()
        .unsupported
        .iter()
        .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission);
    let mut count = 0;
    for entry in rejected {
        let cases = rejection_cases(entry.id).ok_or(format!("no case for `{}`", entry.id))?;
        for (case, fragment) in cases {
            let reason = rejection(case)?.ok_or(format!("`{}` admitted", entry.id))?;
            assert!(
                reason.contains(fragment),
                "`{}`: `{reason}` lacks `{fragment}`",
                entry.id
            );
            count += 1;
        }
    }
    assert!(count > 0, "solver declares no admission rejection");
    Ok(())
}

#[test]
fn solver_regimes_match_admission() -> TestResult {
    let declared = surface_solver_capabilities().regimes;
    for claim in declared {
        let case = regime_case(claim.scope, claim.regime).ok_or(format!(
            "no case for `{}` at {:?}",
            claim.regime.id(),
            claim.scope
        ))?;
        admits(case).map_err(|error| format!("{claim:?}: {error}"))?;
    }
    for (kind, regime) in [
        (BoundaryKind::Stage, Regime::Supercritical),
        (BoundaryKind::Stage, Regime::Dry),
        (BoundaryKind::Discharge, Regime::Supercritical),
        (BoundaryKind::Transmissive, Regime::WetSubcritical),
    ] {
        let scope = RegimeScope::Boundary(kind);
        assert!(
            !declared.contains(&RegimeSupport { scope, regime }),
            "{kind:?} claims {regime:?}"
        );
        let case = regime_case(scope, regime).ok_or(format!("no case for {kind:?} {regime:?}"))?;
        assert!(
            rejection(case)?.is_some(),
            "{kind:?} admitted undeclared {regime:?}"
        );
    }
    Ok(())
}

#[test]
fn solver_capabilities_point_to_standalone_components() -> TestResult {
    for id in [
        "channel-reaches",
        "channel-pools",
        "level-pools",
        "coupled-level-pools",
        "runoff-pools",
        "reservoir",
        "subcatchment",
        "structures",
    ] {
        let entry = surface_solver_capabilities()
            .unsupported_by_id(id)
            .ok_or(format!("`{id}` is undeclared"))?;
        assert_eq!(entry.reason, UnsupportedReason::NotAttached, "`{id}`");
        assert!(entry.summary.contains("not attached"), "`{id}`");
        assert!(entry.summary.contains("standalone component"), "`{id}`");
    }
    for entry in surface_solver_capabilities().unsupported {
        let summary = entry.summary.to_lowercase();
        assert!(
            !summary.contains("network") || summary.contains("not attached"),
            "`{}` names networks without the standalone wording",
            entry.id
        );
    }
    Ok(())
}
