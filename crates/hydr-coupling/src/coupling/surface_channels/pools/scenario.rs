//! Pool test fixtures: constant-area ponds beside the structure tanks of
//! [`super::super::structures::scenario`].

use super::{SurfaceChannelPoolForcing, SurfaceChannelPools};
use crate::channel::{ChannelBoundary, ChannelLateralInput};
use crate::coupling::level_pools::{
    CoupledPoolInflow, CoupledPoolInflowBinding, CoupledPoolSupport,
};
use crate::coupling::surface_channels::context::NodeEnd;
use crate::coupling::surface_channels::structures::scenario::{
    Batch, DATUM, REACH, Tanks, TestResult, client, inflow, withdrawal,
};
use crate::coupling::surface_channels::structures::{
    StructureScenario, SurfaceChannelAttachment, SurfaceChannelNode, SurfaceChannelOperationOwner,
    SurfaceChannelStructure,
};
use crate::coupling::surface_channels::{
    SurfaceChannelBatch, SurfaceChannelDefinition, SurfaceChannelError,
    SurfaceChannelForcingInterval, SurfaceChannelReachForcing,
};
use crate::forcing::WaterAmount;
use crate::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolSourcePolicy,
    LevelPoolState,
};
use crate::structures::{
    DirectionPolicy, PumpApplicability, PumpCurve, PumpCurveKnot, PumpParameters, PumpSpeedPolicy,
    StructureElement, StructureLaw, StructureOperatingPoint, SuppressedRectangularWeirParameters,
    WeirApplicability,
};

/// Pump on a steep linear curve, `Q = 0.5 (1 - lift)` m³/s for lift in `[-1, 1]` m, with its
/// donor above `suction`.
pub fn steep_pump(suction: f64) -> TestResult<StructureLaw> {
    let knot = |head_m, discharge_m3_per_s| PumpCurveKnot {
        head_m,
        discharge_m3_per_s,
    };
    let curve = PumpCurve::new(
        vec![knot(-1.0, 1.0), knot(1.0, 0.0)],
        "Analytical steep curve".into(),
    )?;
    Ok(StructureLaw::Pump(PumpParameters::new(
        curve,
        suction,
        PumpSpeedPolicy::ReferenceOnly,
        PumpApplicability {
            endpoint_stage_head_assertion: "Quiescent pond stages".into(),
            connection_losses_and_velocity_head_account: "Curve includes connections".into(),
            stopped_isolation_assertion: "Ideal isolation when stopped".into(),
            operating_envelope: "Wet donor within the curve".into(),
        },
    )?))
}

/// Weir regime code of free flow.
pub const FREE_WEIR: f64 = 2.0;

/// A constant-area pond with its origin at `origin`, `area` m² of plan area, and `height` m of
/// curve above the origin.
pub fn pond(id: &str, origin: f64, area: f64, height: f64) -> TestResult<LevelPoolElement> {
    let knot = |volume_m3, stage_m| LevelPoolKnot { volume_m3, stage_m };
    Ok(LevelPoolElement {
        id: id.into(),
        curve: LevelPoolCurve::new(
            vec![knot(0.0, origin), knot(area * height, origin + height)],
            DATUM.into(),
            "Synthetic constant-area pond".into(),
        )?,
        connected_pool_assertion: "One level water body".into(),
        storage_support_identity: format!("{id}-basin"),
    })
}

/// Declaration of `pools` with support at each origin, optional source policies, and `inflows`.
pub fn declaration(
    pools: Vec<LevelPoolElement>,
    sources: bool,
    inflows: Vec<CoupledPoolInflowBinding>,
) -> SurfaceChannelPools {
    let support = pools
        .iter()
        .map(|pool| CoupledPoolSupport {
            pool_id: pool.id.clone(),
            zero_volume_stage_m: pool.curve.origin(),
            zero_volume_limit_assertion: "The pond empties at its curve origin".into(),
        })
        .collect();
    let policies = sources.then(|| {
        pools
            .iter()
            .map(|_| LevelPoolSourcePolicy {
                dry_basal_footprint_assertion: "The first segment is the rain footprint".into(),
                bottom_loss_provenance: "Supplied bottom demand".into(),
            })
            .collect()
    });
    SurfaceChannelPools {
        pools,
        support,
        sources: policies,
        inflows,
        storage: LevelPoolConfig::default(),
    }
}

/// External inflow binding `id` into pool `pool`.
pub fn inflow_binding(id: &str, pool: &str) -> CoupledPoolInflowBinding {
    CoupledPoolInflowBinding {
        boundary_id: id.into(),
        pool_id: pool.into(),
        input_id: format!("{id}-input"),
        physical_support_id: format!("{id}-pipe"),
        boundary_assertion: "The pipe discharges into the pond".into(),
        ownership_assertion: "No other model holds this water".into(),
    }
}

/// External inflow input of binding `id` at `rate` m³/s.
pub fn inflow_input(id: &str, rate: f64) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: id.into(),
        input_id: format!("{id}-input"),
        value: WaterAmount::Rate {
            rate_m3_per_s: rate,
        },
    }
}

/// Node of pool `id`.
pub fn pool(id: &str) -> SurfaceChannelNode {
    SurfaceChannelNode::Pool(id.into())
}

/// The surface cell of the tanks.
pub fn surface() -> SurfaceChannelNode {
    SurfaceChannelNode::SurfaceCell(0)
}

/// The reach cell of the tanks.
pub fn channel() -> SurfaceChannelNode {
    SurfaceChannelNode::ChannelCell {
        reach_id: REACH.into(),
        cell: 0,
    }
}

/// Structure `id` from node `a` to node `b`, each with its attachment bed.
pub fn link(
    id: &str,
    law: StructureLaw,
    direction: DirectionPolicy,
    operation: StructureOperatingPoint,
    (a, bed_a): (SurfaceChannelNode, f64),
    (b, bed_b): (SurfaceChannelNode, f64),
) -> SurfaceChannelStructure {
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
            bed_elevation_m: bed_a,
            physical_connection_assertion: "The structure opens onto endpoint A".into(),
        },
        endpoint_b: SurfaceChannelAttachment {
            node: b,
            bed_elevation_m: bed_b,
            physical_connection_assertion: "The structure opens onto endpoint B".into(),
        },
        scenario: StructureScenario {
            id: format!("{id}-as-built"),
            evidence: "As-built geometry, no blockage".into(),
        },
        operation: SurfaceChannelOperationOwner::Manual(operation),
    }
}

/// Suppressed weir of `length` m with its crest at `crest`, symmetric under reversal.
pub fn long_weir(crest: f64, length: f64) -> TestResult<StructureLaw> {
    Ok(StructureLaw::SuppressedRectangularWeir(
        SuppressedRectangularWeirParameters::new(
            length,
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

/// The batch of `tanks` with `pools` at `volumes`, in declaration order, joined by `structures`.
pub fn build(
    tanks: Tanks,
    pools: SurfaceChannelPools,
    volumes: &[f64],
    structures: Vec<SurfaceChannelStructure>,
) -> Result<Batch, SurfaceChannelError> {
    let (setup, initial) = tanks
        .parts()
        .map_err(|error| SurfaceChannelError::Input(error.to_string()))?;
    let definition = SurfaceChannelDefinition::with_nodes(setup, pools, structures)?;
    let states: Vec<LevelPoolState> = volumes
        .iter()
        .map(|&volume_m3| LevelPoolState { volume_m3 })
        .collect();
    SurfaceChannelBatch::with_pools(client(), definition, initial, &states)
}

/// One accepted interval toward `start + dt` with zero caller reach inputs and `pools` forcing.
pub fn advance(
    batch: &Batch,
    dt: f64,
    pools: &SurfaceChannelPoolForcing,
) -> Result<Batch, SurfaceChannelError> {
    let start = batch.time();
    let end = start + dt;
    let lateral: Vec<ChannelLateralInput> = vec![inflow(0.0), withdrawal(0.0)];
    let forcing = SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: lateral,
        }],
    };
    Ok(batch.trial_with_pools_to(end, &forcing, pools)?.into_next())
}

/// Accepted volume of every pool in ascending identity, in m³.
pub fn volumes(batch: &Batch) -> TestResult<Vec<f64>> {
    let pools = batch.pools.as_ref().ok_or("the pools")?;
    Ok(pools.batch.states()?.iter().map(|s| s.volume_m3).collect())
}

/// Accepted stage of the linked surface cell, in m.
pub fn surface_stage(batch: &Batch) -> TestResult<f64> {
    let context = &batch.context;
    let linked: Vec<_> = context
        .linked
        .iter()
        .map(|entry| &batch.reaches[entry.reach])
        .collect();
    let (stage, _) = context.stages(&batch.surface, &linked, &[REACH])?;
    Ok(*stage
        .try_to_vec::<f64>()?
        .first()
        .ok_or("a linked surface cell")?)
}

/// Accepted stage of every pool in ascending identity, in m.
pub fn stages(batch: &Batch) -> TestResult<Vec<f64>> {
    let pools = batch.pools.as_ref().ok_or("the pools")?;
    let (stage, _) = batch.context.pool_stages(&pools.batch)?;
    Ok(stage.try_to_vec::<f64>()?)
}

/// Checks the last interval: mass, pairing at the surface, the channel, and every pool, and the
/// lifetime composite balance.
///
/// `external` holds each pool's external inflow volume over the interval, in ascending identity.
pub fn check(batch: &Batch, external: &[f64]) -> TestResult {
    let interval = batch.last_step()?.ok_or("an accepted interval")?;
    assert!(
        interval.mass_residual_m3.abs() <= interval.mass_bound_m3,
        "{interval:?}"
    );
    let context = &batch.context;
    let step = batch.last.as_ref().ok_or("an accepted interval")?;
    let signed = step.signed_m3.try_to_vec::<f64>()?;
    let bound = interval.paired_bound_m3;
    let net = |sign: &numr::tensor::Tensor<_>| -> TestResult<f64> {
        let sign: Vec<f64> = sign.try_to_vec::<f64>()?;
        Ok(signed.iter().zip(&sign).map(|(v, s)| v * s).sum())
    };
    let surface = interval.surface_exchange_out_m3 - interval.surface_exchange_in_m3;
    let channel = interval.channel_exchange_in_m3 - interval.channel_exchange_out_m3;
    assert!(
        (surface - net(&context.surface_sign)?).abs() <= bound,
        "{interval:?}"
    );
    assert!(
        (channel - net(&context.channel_sign)?).abs() <= bound,
        "{interval:?}"
    );
    let pools = batch.pools.as_ref().ok_or("the pools")?;
    let native = pools
        .batch
        .coupling_native_interval()
        .ok_or("a pool interval")?;
    let incoming = native.incoming_volume_m3.try_to_vec::<f64>()?;
    let outgoing = native.outgoing_volume_m3.try_to_vec::<f64>()?;
    for (index, (into, out)) in incoming.iter().zip(&outgoing).enumerate() {
        let mut expected = external.get(index).copied().unwrap_or(0.0);
        for (volume, ends) in signed.iter().zip(&context.link_ends) {
            if ends[1] == NodeEnd::Pool(index) {
                expected += volume;
            }
            if ends[0] == NodeEnd::Pool(index) {
                expected -= volume;
            }
        }
        assert!(
            ((into - out) - expected).abs() <= bound,
            "pool {index}: native net={}, expected={expected}",
            into - out
        );
    }
    let balance = batch.ledger()?;
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    Ok(())
}
