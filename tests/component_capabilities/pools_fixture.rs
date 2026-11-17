//! Minimal level pools, coupled pools and channel/pool systems for capability checks.

use crate::coupled_pool_fixture::{DATUM, assemble, orifice_case};
use crate::reference::CASES;
use hydr::capability::ComponentCapabilities;
use hydr::channel::{
    ChannelGeometry, ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint,
    SurveyedChannelCell,
};
use hydr::coupling::{
    ChannelPoolBatch, ChannelPoolEndpoint, ChannelPoolError, ChannelPoolForcingInterval,
    ChannelPoolInitialState, ChannelPoolSetup, CoupledLevelPoolBatch, CoupledLevelPoolError,
    CoupledLevelPoolInitialState, CoupledLevelPoolSetup, channel_pool_capabilities,
    coupled_level_pool_capabilities,
};
use hydr::storage::{
    LevelPoolBatch, LevelPoolError, LevelPoolInput, LevelPoolSourcePolicy, level_pool_capabilities,
};
use hydr::structures::{
    DirectionPolicy, GateApplicability, GateParameters, PumpApplicability, PumpCurve,
    PumpCurveKnot, PumpParameters, PumpSpeedPolicy, StructureElement, StructureError, StructureLaw,
    StructureOperatingPoint, StructureOperation, SuppressedRectangularWeirParameters,
    WeirApplicability,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;
pub type Pool = LevelPoolBatch<crate::runtime::Runtime>;
pub type Coupled = CoupledLevelPoolBatch<crate::runtime::Runtime>;
pub type Connected = ChannelPoolBatch<crate::runtime::Runtime>;

/// Requested coupled interval in seconds, within every fixture's step limits.
pub const STEP: f64 = 0.25;

/// Requested channel/pool interval in seconds.
pub const CHANNEL_STEP: f64 = 0.125;

/// Pool component, one per declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    LevelPool,
    Coupled,
    ChannelPools,
}

pub const KINDS: [Kind; 3] = [Kind::LevelPool, Kind::Coupled, Kind::ChannelPools];

impl Kind {
    pub fn capabilities(self) -> &'static ComponentCapabilities {
        match self {
            Self::LevelPool => level_pool_capabilities(),
            Self::Coupled => coupled_level_pool_capabilities(),
            Self::ChannelPools => channel_pool_capabilities(),
        }
    }
}

/// Error class a rejection reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Input,
    Numerical,
    Operation,
    Channel,
    Unresolved,
}

/// Rejection variant and message, or `None` when the case is accepted.
pub type Outcome = Result<Option<(Variant, String)>, Box<dyn std::error::Error>>;

fn pool_rejection(error: LevelPoolError) -> Outcome {
    match error {
        LevelPoolError::InvalidInput { context, .. } => Ok(Some((Variant::Input, context))),
        LevelPoolError::Numerical { context, .. } => Ok(Some((Variant::Numerical, context))),
        other => Err(format!("unexpected level-pool rejection: {other}").into()),
    }
}

/// Runs one exchange trial to `1 s` and checks that a rejection leaves accepted state unchanged.
pub fn exchange_outcome(pool: &Pool, input: LevelPoolInput) -> Outcome {
    let before = pool.snapshot_data()?;
    let outcome = match pool.trial_to(1.0, &[input]) {
        Ok(trial) => {
            trial.diagnostics()?;
            None
        }
        Err(error) => pool_rejection(error)?,
    };
    assert_eq!(pool.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Classifies a level-pool construction result.
pub fn construction_outcome(result: Result<Pool, LevelPoolError>) -> Outcome {
    match result {
        Ok(_) => Ok(None),
        Err(error) => pool_rejection(error),
    }
}

/// Source policy with explicit footprint and bottom-demand evidence.
pub fn policy() -> LevelPoolSourcePolicy {
    LevelPoolSourcePolicy {
        dry_basal_footprint_assertion:
            "Synthetic first segment is the rain-receiving basal footprint, including dry storage"
                .into(),
        bottom_loss_provenance: "Explicit synthetic potential bottom demand without soil state"
            .into(),
    }
}

/// One sample of each public structure law. A new law fails to compile in `laws_of`.
pub fn laws() -> Result<[StructureLaw; 4], Box<dyn std::error::Error>> {
    let (orifice, _) = orifice_case([4.0, 2.0])?;
    let orifice = orifice
        .structures
        .into_iter()
        .next()
        .ok_or("orifice fixture holds no structure")?
        .law;
    let weir = StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
        1.0,
        0.0,
        1.0,
        WeirApplicability {
            coefficient_provenance: "Explicit synthetic free-weir coefficient".into(),
            free_nappe_and_approach_assertion:
                "Suppressed sharp crest with quiescent synthetic approaches".into(),
            submergence_correction_assertion:
                "Declared existing Villemonte correction where applicable".into(),
            symmetric_reversal: true,
        },
    )?);
    let pump = StructureLaw::Pump(PumpParameters::new(
        PumpCurve::new(
            [(-16.0, 1.0), (16.0, 1.0)]
                .into_iter()
                .map(|(head_m, discharge_m3_per_s)| PumpCurveKnot {
                    head_m,
                    discharge_m3_per_s,
                })
                .collect(),
            "Explicit synthetic SI reference-speed curve".into(),
        )?,
        0.0,
        PumpSpeedPolicy::ReferenceOnly,
        PumpApplicability {
            endpoint_stage_head_assertion: "Quiescent common-datum synthetic endpoint stages"
                .into(),
            connection_losses_and_velocity_head_account:
                "Effective reference curve includes declared connection losses".into(),
            stopped_isolation_assertion:
                "Explicit isolation prevents siphoning and reverse flow when stopped".into(),
            operating_envelope: "Synthetic fixture with positive wet suction and receiving support"
                .into(),
        },
    )?);
    let gate = StructureLaw::HydrostaticRectangularGate(GateParameters::new(
        1.0,
        0.0,
        1.0,
        1.0,
        0.5,
        GateApplicability {
            coefficient_provenance: "Analytical numerical aperture fixture".into(),
            hydrostatic_quiescent_pool_assertion:
                "Hydrostatic pools with negligible approach velocity on one datum".into(),
            free_submerged_aperture_applicability:
                "One analytical coefficient covers free and submerged aperture limits".into(),
            unobstructed_aperture_assertion:
                "Each connection exposes the complete sill-to-crown aperture".into(),
            symmetric_reversal: true,
        },
    )?);
    Ok([gate, pump, orifice, weir])
}

/// One sample of each public operating point.
pub fn operating_points() -> [StructureOperatingPoint; 4] {
    [
        StructureOperatingPoint::Passive,
        StructureOperatingPoint::PumpStoppedIsolated,
        StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
        StructureOperatingPoint::GateOpening { opening_m: 0.5 },
    ]
}

/// Two pools on a zero bed joined by one structure from `a` to `b` under `operating_point`.
pub fn coupled_system(
    law: StructureLaw,
    operating_point: StructureOperatingPoint,
    volumes: [f64; 2],
) -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState), Box<dyn std::error::Error>> {
    let direction = if matches!(law, StructureLaw::Pump(_)) {
        DirectionPolicy::AtoBOnly
    } else {
        DirectionPolicy::Bidirectional
    };
    assemble(
        &[("a", 0.0), ("b", 0.0)],
        &volumes,
        vec![StructureElement {
            id: "structure".into(),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: DATUM.into(),
            direction,
            law,
        }],
        vec![StructureOperation {
            structure_id: "structure".into(),
            operating_point,
        }],
    )
}

/// Classifies a coupled-pool rejection.
pub fn coupled_rejection(error: CoupledLevelPoolError) -> Outcome {
    match error {
        CoupledLevelPoolError::Input(message) => Ok(Some((Variant::Input, message))),
        CoupledLevelPoolError::Numerical(message) => Ok(Some((Variant::Numerical, message))),
        CoupledLevelPoolError::Structure(StructureError::InvalidInput { context, .. }) => {
            Ok(Some((Variant::Operation, context)))
        }
        CoupledLevelPoolError::Unresolved { reason, .. } => Ok(Some((Variant::Unresolved, reason))),
        other => Err(format!("unexpected coupled-pool rejection: {other}").into()),
    }
}

/// Constructs `setup` and runs one exchange-only trial to `STEP`.
pub fn coupled_outcome(
    setup: CoupledLevelPoolSetup,
    initial: CoupledLevelPoolInitialState,
) -> Outcome {
    let origin = match Coupled::new(crate::runtime::client(), setup, initial) {
        Ok(batch) => batch,
        Err(error) => return coupled_rejection(error),
    };
    let before = origin.snapshot_data()?;
    let outcome = match origin.trial_to(STEP) {
        Ok(trial) => {
            assert_eq!(trial.into_next().time(), STEP, "trial shortened");
            None
        }
        Err(error) => coupled_rejection(error)?,
    };
    assert_eq!(origin.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// One resting reach of the reference geometry with `endpoint` connected to a finite pool.
pub fn channel_pool_case(
    triangle: bool,
    endpoint: ChannelPoolEndpoint,
) -> Result<(ChannelPoolSetup, ChannelPoolInitialState), Box<dyn std::error::Error>> {
    let case = CASES
        .iter()
        .find(|case| case.triangle == triangle)
        .ok_or("no reference case of the requested geometry")?;
    crate::channel_pool_fixture::single(*case, endpoint)
}

/// One-cell rectangular survey on `datum`, the geometry the coordinator refuses.
pub fn surveyed_geometry(datum: &str) -> ChannelGeometry {
    ChannelGeometry::PrismaticSurveyed {
        profile: CrossSectionDefinition {
            id: "surveyed-rectangle".into(),
            vertical_datum: datum.into(),
            points: [(0.0, 4.0), (0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: 4.0,
            profile_provenance: "Explicit analytical rectangular survey".into(),
        },
        cells: vec![SurveyedChannelCell {
            id: "cell-0".into(),
            length_m: 128.0,
            manning_n: 0.0,
        }],
        numerical: ChannelSurveyedConfig::default(),
    }
}

fn channel_pool_rejection(error: ChannelPoolError) -> Outcome {
    match error {
        ChannelPoolError::Input(message) => Ok(Some((Variant::Input, message))),
        ChannelPoolError::Numerical(message) => Ok(Some((Variant::Numerical, message))),
        ChannelPoolError::Channel { source, .. } => {
            Ok(Some((Variant::Channel, source.to_string())))
        }
        ChannelPoolError::Pool(error) => Ok(Some((Variant::Numerical, error.to_string()))),
        ChannelPoolError::Unresolved { reason, .. } => Ok(Some((Variant::Unresolved, reason))),
        other => Err(format!("unexpected channel/pool rejection: {other}").into()),
    }
}

/// Constructs `setup` and runs one trial to `CHANNEL_STEP` under `forcing`.
pub fn channel_pool_outcome(
    setup: ChannelPoolSetup,
    initial: ChannelPoolInitialState,
    forcing: &ChannelPoolForcingInterval,
) -> Outcome {
    let origin = match Connected::new(crate::runtime::client(), setup, initial) {
        Ok(batch) => batch,
        Err(error) => return channel_pool_rejection(error),
    };
    let before = origin.snapshot_data()?;
    let outcome = match origin.trial_to(CHANNEL_STEP, forcing) {
        Ok(trial) => {
            assert!(trial.time() > origin.time(), "trial accepted no time");
            None
        }
        Err(error) => channel_pool_rejection(error)?,
    };
    assert_eq!(origin.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Runs `channel_pool_outcome` under the fixture's held exterior forcing.
pub fn channel_pool_step(setup: ChannelPoolSetup, initial: ChannelPoolInitialState) -> Outcome {
    let forcing = crate::channel_pool_fixture::forcing(&setup, 0.0, CHANNEL_STEP);
    channel_pool_outcome(setup, initial, &forcing)
}
