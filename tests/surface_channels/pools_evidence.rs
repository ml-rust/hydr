//! Shared pieces of the pool and structure evidence tests: the frozen analytical controls, the
//! per-interval mass, pairing, and composite checks, and the observed order of a refinement.

use crate::pools_fixture::{AREA, Batch, Case, PoolForcing, REACH, Result, forcing};
use hydr::coupling::{SurfaceChannelNode, SurfaceChannelSetup, SurfaceChannelStepDiagnostics};

/// Frozen absolute head allowance of the analytical cases, in m.
pub const HEAD_ABSOLUTE_M: f64 = 1e-13;

/// Frozen absolute transfer allowance of the analytical cases, in m³.
pub const TRANSFER_ABSOLUTE_M3: f64 = 1e-12;

/// The batch of `case` under the frozen analytical controls: relative parts zero.
pub fn build(case: &Case) -> Result<Batch> {
    let (mut setup, initial) = case.parts()?;
    analytic_controls(&mut setup);
    Ok(Batch::new(crate::runtime::client(), setup, initial)?)
}

/// Sets the frozen analytical head and transfer allowances of `setup`, relative parts zero.
pub fn analytic_controls(setup: &mut SurfaceChannelSetup) {
    setup.config.head_absolute_m = HEAD_ABSOLUTE_M;
    setup.config.head_relative = 0.0;
    setup.config.transfer_absolute_m3 = TRANSFER_ABSOLUTE_M3;
    setup.config.transfer_relative = 0.0;
}

/// Accepted stage of the surface cell, in m.
pub fn surface_stage(batch: &Batch, case: &Case) -> Result<f64> {
    let state = batch.surface().state()?;
    Ok(case.surface_bed + state.cells()[0].h)
}

/// Accepted volume of the reach cell, in m³.
pub fn reach_volume(batch: &Batch) -> Result<f64> {
    Ok(batch.reach(REACH).ok_or("the reach")?.state()?[0].volume_m3)
}

/// Accepted stage of the reach cell, in m.
pub fn reach_stage(batch: &Batch, case: &Case) -> Result<f64> {
    Ok(case.reach_bed + reach_volume(batch)? / AREA)
}

/// Accepted stage of pool `id`, in m. The pool has taken at least one interval.
pub fn pool_stage(batch: &Batch, id: &str) -> Result<f64> {
    let step = batch.last_step()?.ok_or("an accepted interval")?;
    Ok(step
        .pools
        .iter()
        .find(|p| p.pool_id == id)
        .ok_or("the pool")?
        .stage_m)
}

/// Accepted volume of pool `id`, in m³. The pool has taken at least one interval.
pub fn pool_volume(batch: &Batch, id: &str) -> Result<f64> {
    let step = batch.last_step()?.ok_or("an accepted interval")?;
    Ok(step
        .pools
        .iter()
        .find(|p| p.pool_id == id)
        .ok_or("the pool")?
        .volume_m3)
}

/// Net volume every node of `batch` took through structures and pool ends over the last
/// interval, positive into the node.
struct Net {
    nodes: Vec<(SurfaceChannelNode, f64)>,
}

impl Net {
    fn add(&mut self, node: &SurfaceChannelNode, volume: f64) {
        match self.nodes.iter_mut().find(|(n, _)| n == node) {
            Some((_, total)) => *total += volume,
            None => self.nodes.push((node.clone(), volume)),
        }
    }

    fn of(&self, node: &SurfaceChannelNode) -> f64 {
        self.nodes
            .iter()
            .find(|(n, _)| n == node)
            .map_or(0.0, |(_, total)| *total)
    }
}

/// Fails unless the last interval of `batch` holds mass, pairing, and the composite balance.
///
/// Pairing: the surface and the channels receive exactly what the structures send them, within
/// the paired bound, and every pool's native incoming minus outgoing volume equals what its
/// structures and pool ends carry plus `external` (its external inflow volume, in ascending pool
/// identity, default zero). Pool-end volumes equal the reach end volumes in every bit.
pub fn check(batch: &Batch, external: &[f64]) -> Result {
    let step: SurfaceChannelStepDiagnostics = batch.last_step()?.ok_or("an accepted interval")?;
    assert!(
        step.mass_residual_m3.abs() <= step.mass_bound_m3,
        "mass: {step:?}"
    );
    let mut net = Net { nodes: Vec::new() };
    for structure in &batch.setup().structures {
        let transfer = step
            .structures
            .iter()
            .find(|t| t.structure_id == structure.element.id)
            .ok_or("the structure receipt")?;
        net.add(&structure.endpoint_a.node, -transfer.signed_volume_m3);
        net.add(&structure.endpoint_b.node, transfer.signed_volume_m3);
    }
    let bound = step.paired_bound_m3;
    let surface = step.surface_exchange_in_m3 - step.surface_exchange_out_m3;
    assert!(
        (surface - net.of(&SurfaceChannelNode::SurfaceCell(0))).abs() <= bound,
        "surface pairing: {step:?}"
    );
    let channel = step.channel_exchange_in_m3 - step.channel_exchange_out_m3;
    let channel_net = net.of(&SurfaceChannelNode::ChannelCell {
        reach_id: REACH.into(),
        cell: 0,
    });
    assert!(
        (channel - channel_net).abs() <= bound,
        "channel pairing: {step:?}"
    );
    for end in &batch.setup().pool_ends {
        let transfer = step
            .pool_ends
            .iter()
            .find(|t| t.pool_end_id == end.id)
            .ok_or("the pool-end receipt")?;
        net.add(
            &SurfaceChannelNode::Pool(end.pool_id.clone()),
            transfer.signed_volume_m3,
        );
    }
    for (index, pool) in step.pools.iter().enumerate() {
        let expected = net.of(&SurfaceChannelNode::Pool(pool.pool_id.clone()))
            + external.get(index).copied().unwrap_or(0.0);
        let native = pool.incoming_m3 - pool.outgoing_m3;
        assert!(
            (native - expected).abs() <= bound,
            "pool {}: native net={native}, expected={expected}",
            pool.pool_id
        );
    }
    let balance = batch.ledger()?;
    assert!(
        balance.residual_m3.abs() <= balance.bound_m3,
        "composite: {balance:?}"
    );
    Ok(())
}

/// One accepted interval of exactly `dt`, checked, with the receipts of the interval.
pub fn advance(
    batch: &Batch,
    dt: f64,
    pools: &PoolForcing,
    external: &[f64],
) -> Result<(Batch, SurfaceChannelStepDiagnostics)> {
    let start = batch.time();
    let end = start + dt;
    let held = forcing(batch, end, pools);
    let next = batch.trial_to(end, &held)?.into_next();
    let step = next.last_step()?.ok_or("an accepted interval")?;
    assert_eq!(
        step.end_s.to_bits(),
        end.to_bits(),
        "the interval must take the full duration: {step:?}"
    );
    assert_eq!(step.start_s.to_bits(), start.to_bits());
    check(&next, external)?;
    Ok((next, step))
}

/// Orifice area of the fixture law `orifice_law`, in m².
pub const ORIFICE_AREA_M2: f64 = 0.04;

/// Discharge coefficient of the fixture law `orifice_law`.
pub const ORIFICE_DISCHARGE: f64 = 0.6;

/// Two level bodies of constant plan area joined by the fixture orifice.
///
/// The head difference `Δ` follows `Δ(t) = (sqrt(Δ0) - k t / 2)^2` with
/// `k = C_d a sqrt(2g) (1/A_1 + 1/A_2)`. The heads meet at `t* = 2 sqrt(Δ0) / k`.
#[derive(Clone, Copy, Debug)]
pub struct OrificeFill {
    /// Plan area of the first body, in m².
    pub area_1_m2: f64,
    /// Plan area of the second body, in m².
    pub area_2_m2: f64,
    /// Initial head difference, in m.
    pub difference_m: f64,
}

impl OrificeFill {
    /// `k` of the exact solution, in `sqrt(m)/s`.
    pub fn rate(&self) -> f64 {
        let g = hydr::solver::SolverConfig::default().gravity;
        ORIFICE_DISCHARGE
            * ORIFICE_AREA_M2
            * (2.0 * g).sqrt()
            * (1.0 / self.area_1_m2 + 1.0 / self.area_2_m2)
    }

    /// Time the heads meet, in s.
    pub fn meeting_time(&self) -> f64 {
        2.0 * self.difference_m.sqrt() / self.rate()
    }

    /// Exact head difference at `t`, in m.
    pub fn exact(&self, t: f64) -> f64 {
        (self.difference_m.sqrt() - self.rate() * t / 2.0).powi(2)
    }
}

/// `batch` advanced to `end`, with every accepted interval checked and returned.
///
/// The coordinator may end an interval early, at an actuator arrival bound for example. Then the
/// next interval starts there. `rates` holds each pool's external inflow rate in m³/s, in
/// ascending pool identity, default zero.
pub fn advance_to(
    mut batch: Batch,
    end: f64,
    pools: &PoolForcing,
    rates: &[f64],
) -> Result<(Batch, Vec<SurfaceChannelStepDiagnostics>)> {
    let mut steps = Vec::new();
    while batch.time() < end {
        let held = forcing(&batch, end, pools);
        let next = batch.trial_to(end, &held)?.into_next();
        let step = next.last_step()?.ok_or("an accepted interval")?;
        assert!(step.end_s > step.start_s, "{step:?}");
        let external: Vec<f64> = rates
            .iter()
            .map(|r| r * (step.end_s - step.start_s))
            .collect();
        check(&next, &external)?;
        steps.push(step);
        batch = next;
    }
    assert_eq!(batch.time().to_bits(), end.to_bits());
    Ok((batch, steps))
}

/// Observed order between each consecutive pair of `errors`, each from half the duration of the
/// one before.
pub fn orders(errors: &[f64]) -> Vec<f64> {
    errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
}

/// Fails unless the orders of the two finest pairs of `errors` reach `floor`.
pub fn assert_order(errors: &[f64], floor: f64) {
    let orders = orders(errors);
    let finest = &orders[orders.len() - 2..];
    assert!(
        finest.iter().all(|order| *order >= floor),
        "errors={errors:?}, orders={orders:?}"
    );
}
