//! Public runoff variants mapped to declared capability ids.
//!
//! Each mapping is an exhaustive `match`, so a new public variant fails to compile until it
//! is classified.

use crate::land_fixture::domain;
use crate::runoff_fixture::{
    Kind, RunoffPools, TestResult, drainage_branch, drainage_samples, soil_parameters,
};
use crate::runoff_pools::fixture::{inactive_pools, pervious, setup};
use crate::subcatchment_fixture::parameters;
use hydr::capability::{BoundaryKind, EndpointId, SourceKind, UnsupportedReason};
use hydr::runoff::{ReservoirParameters, Subarea, SubcatchmentRouting};
use hydr::soil::{LandCell, LandModel, NativeSoilState, SoilCellState, SoilState};

/// One sample of each public routing choice.
pub const ROUTINGS: [SubcatchmentRouting; 3] = [
    SubcatchmentRouting::Direct,
    SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
    SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 },
];

/// Value of the `subcatchment-routing` option that each routing choice selects.
pub fn routing_id(routing: SubcatchmentRouting) -> &'static str {
    match routing {
        SubcatchmentRouting::Direct => "direct",
        SubcatchmentRouting::ImperviousToPervious { .. } => "impervious-to-pervious",
        SubcatchmentRouting::PerviousToImpervious { .. } => "pervious-to-impervious-storage",
    }
}

/// Source kind that each subarea moves water through.
fn subarea_source(subarea: Subarea) -> SourceKind {
    match subarea {
        Subarea::ImperviousZeroStorage | Subarea::ImperviousStorage => SourceKind::ImperviousLand,
        Subarea::Pervious => SourceKind::ModifiedGreenAmpt,
    }
}

/// Source kind of each land cell.
fn land_source(cell: LandCell) -> SourceKind {
    match cell {
        LandCell::Impervious => SourceKind::ImperviousLand,
        LandCell::ModifiedGreenAmpt(_) => SourceKind::ModifiedGreenAmpt,
    }
}

/// Whether antecedent memory fits its land cell, or the declared rejection id.
fn memory_fit(cell: LandCell, state: SoilCellState) -> Result<(), &'static str> {
    match (cell, state) {
        (LandCell::Impervious, SoilCellState::Impervious)
        | (LandCell::ModifiedGreenAmpt(_), SoilCellState::ModifiedGreenAmpt(_)) => Ok(()),
        (LandCell::Impervious, SoilCellState::ModifiedGreenAmpt(_))
        | (LandCell::ModifiedGreenAmpt(_), SoilCellState::Impervious) => {
            Err("mismatched-land-state")
        }
    }
}

/// Asserts that `id` names a declared `RejectedByAdmission` entry of `kind`.
fn assert_refused(kind: Kind, id: &str) -> TestResult {
    let entry = kind
        .capabilities()
        .unsupported_by_id(id)
        .ok_or(format!("{kind:?}: `{id}` is undeclared"))?;
    assert_eq!(entry.reason, UnsupportedReason::RejectedByAdmission, "{id}");
    Ok(())
}

#[test]
fn runoff_public_variants_are_classified() -> TestResult {
    let subcatchment = Kind::Subcatchment.capabilities();
    let routing = subcatchment
        .option("subcatchment-routing")
        .ok_or("subcatchment declares no routing option")?;
    assert_eq!(routing.values.len(), ROUTINGS.len(), "routing values");
    for sample in ROUTINGS {
        assert!(
            routing.values.contains(&routing_id(sample)),
            "`{}` is undeclared",
            routing_id(sample)
        );
        parameters(0.5, 0.5, sample)?;
    }

    // Every subarea is a potential outlet of a direct pervious subcatchment.
    let (pools, states) = inactive_pools([0.0, 0.0])?;
    let (coupled, initial) = setup(vec![pervious(ROUTINGS[0])?], pools, states)?;
    let outlets: Vec<_> = coupled.outlets.iter().map(|o| o.subarea).collect();
    assert_eq!(outlets, Subarea::ALL, "potential outlets");
    RunoffPools::new(crate::runtime::client(), coupled, initial)?;
    let runoff_pools = Kind::RunoffPools.capabilities();
    for subarea in Subarea::ALL {
        let source = subarea_source(subarea);
        for kind in [Kind::Subcatchment, Kind::RunoffPools] {
            assert!(kind.capabilities().sources.contains(&source), "{kind:?}");
        }
    }
    assert!(runoff_pools.admits(EndpointId::Outlet, BoundaryKind::RunoffOutletNoFeedback));

    let soil = soil_parameters()?;
    let cells = [LandCell::Impervious, LandCell::ModifiedGreenAmpt(soil)];
    let memories = [
        SoilCellState::Impervious,
        SoilCellState::ModifiedGreenAmpt(NativeSoilState {
            imd: soil.maximum_deficit(),
            f_ft: 0.0,
            fu_ft: 0.0,
            saturated: false,
            t_s: 0.0,
        }),
    ];
    let solver = hydr::solver::capabilities();
    let one = domain(1, 1, vec![true])?;
    for cell in cells {
        let source = land_source(cell);
        assert!(solver.sources.contains(&source), "solver `{}`", source.id());
        if source == SourceKind::ModifiedGreenAmpt {
            assert!(Kind::Soil.capabilities().sources.contains(&source));
        }
        let land = LandModel::new(&one, vec![cell])?;
        for memory in memories {
            let admitted = SoilState::new(&land, vec![memory]).is_ok();
            match memory_fit(cell, memory) {
                Ok(()) => assert!(admitted, "{cell:?} {memory:?}"),
                Err(id) => {
                    assert!(!admitted, "{cell:?} {memory:?}");
                    assert_refused(Kind::Soil, id)?;
                }
            }
        }
    }

    let mut branches = Vec::new();
    for parameters in drainage_samples()? {
        branches.push(drainage_branch(parameters)?);
    }
    branches.sort_unstable();
    branches.dedup();
    assert_eq!(branches.len(), 3, "drainage branches");
    let ambiguous = ReservoirParameters::from_native(1024.0, 1024.0, 0.0, 1.0 / 64.0, 0.0, 0.125);
    assert!(ambiguous.is_err(), "ambiguous closure admitted");
    assert_refused(Kind::Reservoir, "ambiguous-closure")
}
