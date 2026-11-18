//! Structure link tests: one 2 m square surface cell beside one 2 m by 2 m reach cell.
//!
//! Both cells hold `AREA` of plan area behind impermeable walls, so the two tanks exchange
//! water only through their structure links and the caller's lateral inputs on the reach cell.

use super::{
    StructureScenario, SurfaceChannelAttachment, SurfaceChannelNode, SurfaceChannelOperationOwner,
    SurfaceChannelStructure,
};
use crate::channel::{
    ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelLateralBinding,
    ChannelLateralDirection, ChannelLateralInput, ChannelReachDefinition, RectangularChannelCell,
};
use crate::coupling::surface_channels::{
    SurfaceChannelBatch, SurfaceChannelConfig, SurfaceChannelError, SurfaceChannelForcingInterval,
    SurfaceChannelInitialState, SurfaceChannelReach, SurfaceChannelReachForcing,
    SurfaceChannelSetup,
};
use crate::forcing::WaterAmount;
use crate::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use crate::storage::LevelPoolConfig;
use crate::structures::{
    DirectionPolicy, GateApplicability, GateParameters, OrificeApplicability, PumpApplicability,
    PumpCurve, PumpCurveKnot, PumpFreeOutlet, PumpParameters, PumpSpeedPolicy, StructureElement,
    StructureLaw, StructureOperatingPoint, SubmergedOrificeParameters,
    SuppressedRectangularWeirParameters, WeirApplicability,
};
use hydr_core::grid::CartesianGrid;
use hydr_core::model::{CellState, Domain, FlowState, SpatialReference};
use hydr_soil::soil::{LandModel, SoilState};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
pub type Batch = SurfaceChannelBatch<CpuRuntime>;

pub const DATUM: &str = "Synthetic structure datum";
pub const REACH: &str = "r";
/// Plan area of the surface cell and of the reach cell, in m².
pub const AREA: f64 = 4.0;
/// Caller inflow binding on the reach cell.
pub const INFLOW: &str = "caller:inflow";
/// Caller withdrawal binding on the reach cell.
pub const WITHDRAWAL: &str = "caller:withdrawal";
/// Absolute and relative head allowance of every case.
pub const HEAD_ALLOWANCE: f64 = 1e-12;
/// Requested duration of every interval, in s.
pub const DT: f64 = 0.5;

/// Structures regime codes.
pub const EQUAL_HEADS: f64 = 5.0;
pub const PUMP_CURVE: f64 = 6.0;
pub const FREE_GATE: f64 = 10.0;
pub const MIXED_GATE: f64 = 11.0;
pub const SUBMERGED_GATE: f64 = 12.0;

/// Fixture gravity, bit-identical to the surface default.
fn gravity() -> f64 {
    SolverConfig::default().gravity
}

/// Suppressed weir with its crest at `crest`, symmetric under reversal.
pub fn weir_law(crest: f64) -> TestResult<StructureLaw> {
    Ok(StructureLaw::SuppressedRectangularWeir(
        SuppressedRectangularWeirParameters::new(
            1.0,
            crest,
            1.7,
            WeirApplicability {
                coefficient_provenance: "Declared weir coefficient".into(),
                free_nappe_and_approach_assertion: "Suppressed crest, quiescent approach".into(),
                submergence_correction_assertion: "Villemonte correction adopted".into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// Fully submerged 0.04 m² orifice between `sill` and `crown`, symmetric under reversal.
pub fn orifice_law(sill: f64, crown: f64) -> TestResult<StructureLaw> {
    Ok(StructureLaw::SubmergedOrifice(
        SubmergedOrificeParameters::new(
            0.04,
            sill,
            crown,
            0.6,
            gravity(),
            OrificeApplicability {
                coefficient_provenance: "Declared culvert-approximating coefficient".into(),
                fully_submerged_applicability: "Both pools stand above the crown".into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// 0.5 m wide gate with its sill at `sill`, symmetric under reversal.
pub fn gate_law(sill: f64) -> TestResult<StructureLaw> {
    Ok(StructureLaw::HydrostaticRectangularGate(
        GateParameters::new(
            0.5,
            sill,
            1.0,
            0.6,
            gravity(),
            GateApplicability {
                coefficient_provenance: "Declared gate coefficient".into(),
                hydrostatic_quiescent_pool_assertion: "Quiescent tanks on both sides".into(),
                free_submerged_aperture_applicability: "One coefficient across regimes".into(),
                unobstructed_aperture_assertion: "Both tanks expose the aperture".into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// Pump on a falling linear curve from 0.2 m³/s at lift -1 m to shutoff at lift 3 m.
///
/// The donor must stay above `suction`. `outlet` adds a free delivery outlet.
pub fn pump_law(suction: f64, outlet: Option<f64>) -> TestResult<StructureLaw> {
    let curve = PumpCurve::new(
        vec![
            PumpCurveKnot {
                head_m: -1.0,
                discharge_m3_per_s: 0.2,
            },
            PumpCurveKnot {
                head_m: 3.0,
                discharge_m3_per_s: 0.0,
            },
        ],
        "Analytical linear curve".into(),
    )?;
    let pump = PumpParameters::new(
        curve,
        suction,
        PumpSpeedPolicy::ReferenceOnly,
        PumpApplicability {
            endpoint_stage_head_assertion: "Quiescent tank stages".into(),
            connection_losses_and_velocity_head_account: "Curve includes connections".into(),
            stopped_isolation_assertion: "Ideal isolation when stopped".into(),
            operating_envelope: "Wet donor within the curve".into(),
        },
    )?;
    Ok(StructureLaw::Pump(match outlet {
        None => pump,
        Some(elevation_m) => pump.with_free_outlet(PumpFreeOutlet {
            elevation_m,
            evidence: "Free discharge at the outlet".into(),
        })?,
    }))
}

/// Reference discharge of [`pump_law`] at `lift`, in m³/s.
pub fn pump_curve(lift: f64) -> f64 {
    0.2 * (3.0 - lift) / 4.0
}

/// Structure `id` from the surface cell (A) to the reach cell (B) on attachment `beds`.
pub fn structure(
    id: &str,
    law: StructureLaw,
    direction: DirectionPolicy,
    operation: StructureOperatingPoint,
    beds: [f64; 2],
) -> SurfaceChannelStructure {
    let a = SurfaceChannelNode::SurfaceCell(0);
    let b = SurfaceChannelNode::ChannelCell {
        reach_id: REACH.into(),
        cell: 0,
    };
    SurfaceChannelStructure {
        element: StructureElement {
            id: id.into(),
            endpoint_a: a.identity(),
            endpoint_b: b.identity(),
            datum: DATUM.into(),
            direction,
            law,
        },
        endpoint_a: SurfaceChannelAttachment {
            node: a,
            bed_elevation_m: beds[0],
            physical_connection_assertion: "The structure opens onto the cell".into(),
        },
        endpoint_b: SurfaceChannelAttachment {
            node: b,
            bed_elevation_m: beds[1],
            physical_connection_assertion: "The structure opens into the reach".into(),
        },
        scenario: StructureScenario {
            id: format!("{id}-as-built"),
            evidence: "As-built geometry, no blockage".into(),
        },
        operation: SurfaceChannelOperationOwner::Manual(operation),
    }
}

/// Beds and initial stages of the two tanks, in metres. A stage at its bed is dry.
#[derive(Clone, Copy, Debug)]
pub struct Tanks {
    pub surface_bed: f64,
    pub surface_stage: f64,
    pub reach_bed: f64,
    pub reach_stage: f64,
}

impl Tanks {
    /// Tanks on beds at `0.0` with stages `surface` and `reach`.
    pub fn at(surface: f64, reach: f64) -> Self {
        Self {
            surface_bed: 0.0,
            surface_stage: surface,
            reach_bed: 0.0,
            reach_stage: reach,
        }
    }

    /// Setup and initial state of these tanks, with no bank link and no port.
    pub fn parts(&self) -> TestResult<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
        let grid = CartesianGrid::new(1, 1, 2.0, 2.0)?;
        let domain = Domain::new(
            grid,
            vec![self.surface_bed],
            vec![0.0],
            vec![true],
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1000.0,
                rain: vec![0.0],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: Vec::new(),
            }],
        )?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let depth = (self.surface_stage - self.surface_bed).max(0.0);
        let flow = FlowState::new(
            &domain,
            vec![CellState {
                h: depth,
                qx: 0.0,
                qy: 0.0,
            }],
        )?;
        let binding = |boundary_id: &str, direction| ChannelLateralBinding {
            boundary_id: boundary_id.into(),
            cell_id: "cell-0".into(),
            direction,
            ownership_assertion: "Caller water at the reach cell".into(),
        };
        let reach = ChannelReachDefinition {
            vertical_datum: DATUM.into(),
            provenance: "One walled rectangular tank cell".into(),
            geometry: ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell-0".into(),
                    length_m: 2.0,
                    width_m: 2.0,
                    bed_elevation_m: self.reach_bed,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![
                binding(INFLOW, ChannelLateralDirection::Inflow),
                binding(WITHDRAWAL, ChannelLateralDirection::Withdrawal),
            ],
            config: ChannelConfig::default(),
        };
        let volume = (self.reach_stage - self.reach_bed).max(0.0) * AREA;
        let setup = SurfaceChannelSetup {
            surface: SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            reaches: vec![SurfaceChannelReach {
                id: REACH.into(),
                definition: reach,
            }],
            links: Vec::new(),
            ports: Vec::new(),
            pools: Vec::new(),
            pool_support: Vec::new(),
            pool_sources: None,
            pool_inflows: Vec::new(),
            storage: LevelPoolConfig::default(),
            structures: Vec::new(),
            pool_ends: Vec::new(),
            operating: None,
            storage_exclusion_evidence: "The reach tank lies outside the surface cell".into(),
            config: SurfaceChannelConfig {
                head_absolute_m: HEAD_ALLOWANCE,
                head_relative: HEAD_ALLOWANCE,
                transfer_absolute_m3: 1e-12,
                transfer_relative: 1e-12,
                mass_absolute_m3: 1e-9,
                mass_relative: 1e-12,
                minimum_step_s: 1e-10,
                maximum_step_s: 1.0,
                iteration_limit: 64,
                retry_limit: 16,
                contraction_target: 0.5,
            },
        };
        let initial = SurfaceChannelInitialState {
            time_s: 0.0,
            surface: InitialState {
                time: 0.0,
                flow,
                soil,
            },
            reaches: vec![(
                REACH.into(),
                ChannelInitialState {
                    time_s: 0.0,
                    volume_m3: vec![volume],
                    longitudinal_momentum_m4_per_s: vec![0.0],
                },
            )],
            pools: Vec::new(),
        };
        Ok((setup, initial))
    }

    /// The batch of these tanks joined by `structures`.
    pub fn build(&self, structures: Vec<SurfaceChannelStructure>) -> TestResult<Batch> {
        let (mut setup, initial) = self.parts()?;
        setup.structures = structures;
        Ok(SurfaceChannelBatch::new(client(), setup, initial)?)
    }
}

pub fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
}

/// One accepted interval toward `start + dt` with `lateral` caller inputs on the reach cell.
///
/// A caller binding absent from `lateral` takes a zero rate. Other inputs pass through.
pub fn step(
    batch: &Batch,
    dt: f64,
    lateral: &[ChannelLateralInput],
) -> Result<Batch, SurfaceChannelError> {
    let start = batch.time();
    let end = start + dt;
    let named = |input: &ChannelLateralInput, id: &str| match input {
        ChannelLateralInput::Inflow { boundary_id, .. }
        | ChannelLateralInput::Withdrawal { boundary_id, .. } => boundary_id == id,
    };
    let mut inputs = lateral.to_vec();
    for (id, zero) in [(INFLOW, inflow(0.0)), (WITHDRAWAL, withdrawal(0.0))] {
        if !lateral.iter().any(|input| named(input, id)) {
            inputs.push(zero);
        }
    }
    let forcing = SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: inputs,
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
    };
    Ok(batch.trial_to(end, &forcing)?.into_next())
}

/// A caller inflow into the reach cell at `rate`, in m³/s.
pub fn inflow(rate: f64) -> ChannelLateralInput {
    ChannelLateralInput::Inflow {
        boundary_id: INFLOW.into(),
        amount: WaterAmount::Rate {
            rate_m3_per_s: rate,
        },
        longitudinal_velocity_m_per_s: 0.0,
    }
}

/// A caller withdrawal from the reach cell at `rate`, in m³/s.
pub fn withdrawal(rate: f64) -> ChannelLateralInput {
    ChannelLateralInput::Withdrawal {
        boundary_id: WITHDRAWAL.into(),
        amount: WaterAmount::Rate {
            rate_m3_per_s: rate,
        },
    }
}

/// Receipt of one link over the last accepted interval.
#[derive(Clone, Copy, Debug)]
pub struct Receipt {
    /// Accepted volume, positive surface-to-channel, in m³.
    pub signed: f64,
    pub level: f64,
    pub regime: f64,
    /// Midpoint surface and channel stages, in m.
    pub midpoint: [f64; 2],
}

/// Receipt of link `id` over the last accepted interval.
pub fn receipt(batch: &Batch, id: &str) -> TestResult<Receipt> {
    let step = batch.last.as_ref().ok_or("an accepted interval")?;
    let index = batch
        .context
        .link_ids
        .iter()
        .position(|link| link == id)
        .ok_or("the link")?;
    let read = |values: &numr::tensor::Tensor<CpuRuntime>| -> TestResult<f64> {
        Ok(*values
            .try_to_vec::<f64>()?
            .get(index)
            .ok_or("a link value")?)
    };
    Ok(Receipt {
        signed: read(&step.signed_m3)?,
        level: read(&step.level)?,
        regime: read(&step.regime)?,
        midpoint: [
            read(&step.midpoint_surface_m)?,
            read(&step.midpoint_channel_m)?,
        ],
    })
}

/// Accepted surface and reach cell stages, in m.
pub fn heads(batch: &Batch) -> TestResult<[f64; 2]> {
    let context = &batch.context;
    let linked: Vec<_> = context
        .linked
        .iter()
        .map(|entry| &batch.reaches[entry.reach])
        .collect();
    let (stage, _) = context.stages(&batch.surface, &linked, &[REACH])?;
    let stage = stage.try_to_vec::<f64>()?;
    Ok([stage[0], stage[1]])
}

/// Checks the last interval: mass, pairing on both sides, and the lifetime composite balance.
pub fn check(batch: &Batch) -> TestResult {
    let interval = batch.last_step()?.ok_or("an accepted interval")?;
    assert!(
        interval.mass_residual_m3.abs() <= interval.mass_bound_m3,
        "{interval:?}"
    );
    let step = batch.last.as_ref().ok_or("an accepted interval")?;
    let net: f64 = step.signed_m3.try_to_vec::<f64>()?.iter().sum();
    let bound = interval.paired_bound_m3;
    let surface = interval.surface_exchange_out_m3 - interval.surface_exchange_in_m3;
    let channel = interval.channel_exchange_in_m3 - interval.channel_exchange_out_m3;
    assert!((surface - net).abs() <= bound, "{interval:?}");
    assert!((channel - net).abs() <= bound, "{interval:?}");
    let balance = batch.ledger()?;
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    Ok(())
}

/// Head allowance at heads `[a, b]`, in m.
pub fn allowance([a, b]: [f64; 2]) -> f64 {
    HEAD_ALLOWANCE + HEAD_ALLOWANCE * a.abs().max(b.abs())
}
