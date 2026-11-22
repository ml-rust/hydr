//! Paired exchange checks and the composite water balance.

use super::context::{Context, Ops};
use super::exchange::Exchange;
use super::pool_ends::{PoolEndCounters, PoolEndInterval};
use super::pools::{PoolChild, inflow_error, pool_error};
use super::ports::{PortCounters, PortInterval};
use super::runoff::RunoffTerms;
use super::state::{LinkCounters, StepTotals};
use super::{SurfaceChannelConfig, SurfaceChannelError, SurfaceChannelWaterBalance};
use crate::channel::ChannelReach;
use crate::runoff::SubcatchmentBatch;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ShapeOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Rounding allowance per lifetime counter magnitude, in units of `f64::EPSILON`.
const COUNTER_ULPS: f64 = 8.0;

/// External water terms of the surface, every reach, and the pools, exchange excluded.
#[derive(Clone, Copy, Debug)]
struct Terms {
    initial: f64,
    storage: f64,
    external_in: f64,
    external_out: f64,
    adjustment: f64,
    child_residual: f64,
    /// Pool residual and rounding bounds, also held in `child_residual`.
    pool_bound: f64,
    magnitude: f64,
}

/// Accepted children of the coordinator: the surface, every reach, and the pools.
pub(super) type Children<'a, R> = (
    &'a Solver<R>,
    &'a [ChannelReach<R>],
    Option<&'a PoolChild<R>>,
);

/// External pool inflow of one interval: its volume and its reduction bound, in m³.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PoolInflowStep {
    pub volume_m3: f64,
    pub bound_m3: f64,
}

/// True when `|value| <= bound`. A NaN on either side fails.
pub(super) fn within(value: f64, bound: f64) -> bool {
    value.abs() <= bound
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Per-channel-link `(inflow, withdrawal)` values from per-binding tensors of every linked
    /// reach, each `[Lc]`.
    fn link_bindings(
        &self,
        per_binding: &[&Tensor<R>],
    ) -> Result<(Tensor<R>, Tensor<R>), SurfaceChannelError> {
        let c = &self.client;
        if self.linked.is_empty() {
            let empty = Tensor::<R>::zeros(&[0], DType::F64, c.device())?;
            return Ok((empty.clone(), empty));
        }
        let mut parts = Vec::with_capacity(per_binding.len());
        for (entry, values) in self.linked.iter().zip(per_binding) {
            parts.push(values.narrow(0, entry.caller_bindings, 2 * entry.links.len())?);
        }
        let flat = c.cat(&parts.iter().collect::<Vec<_>>(), 0)?;
        Ok((
            c.index_select(&flat, 0, &self.inflow_binding_index)?,
            c.index_select(&flat, 0, &self.withdrawal_binding_index)?,
        ))
    }

    fn total(&self, values: &Tensor<R>) -> Result<f64, SurfaceChannelError> {
        if values.numel() == 0 {
            return Ok(0.0);
        }
        Ok(self.client.sum(values, &[0], false)?.item::<f64>()?)
    }

    /// Surface, channel, and pool water terms in reach identity order.
    ///
    /// `exchange` holds the lifetime channel link totals. `ports` and `ends` hold the lifetime
    /// port and pool-end volumes, which the reach boundary totals hold. Pools join with their
    /// storage, local sources, and external inflow ledgers. Their native counters hold link and
    /// pool-end water, so they stay out of the external terms. `runoff` joins the runoff child,
    /// and its delivered water leaves the receivers' external inflow.
    fn terms(
        &self,
        (surface, reaches, pools): Children<'_, R>,
        exchange: (f64, f64),
        (ports, ends): (&PortCounters, &PoolEndCounters),
        runoff: Option<&RunoffTerms>,
    ) -> Result<Terms, SurfaceChannelError> {
        let s = surface.ledger();
        let mut terms = Terms {
            initial: s.initial_storage,
            storage: s.storage,
            external_in: s.rain + s.boundary_in,
            external_out: s.withdrawal + s.infiltration + s.evaporation + s.boundary_out,
            adjustment: s.water_adjustment,
            child_residual: s.residual.abs(),
            pool_bound: 0.0,
            magnitude: 0.0,
        };
        for &index in &self.reach_order {
            let r = reaches[index].ledger();
            terms.initial += r.initial_storage_m3;
            terms.storage += r.storage_m3;
            terms.external_in += r.boundary_in_m3 + r.lateral_in_m3;
            terms.external_out += r.boundary_out_m3 + r.lateral_out_m3;
            terms.child_residual += r.residual_m3.abs();
        }
        let mut pool_magnitude = 0.0;
        if let (Some(pools), Some(topology)) = (pools, &self.pools) {
            let totals = (topology.fns.totals)(&pools.batch).map_err(pool_error)?;
            let external = topology
                .inflows
                .state_summary(&self.client, &pools.external)
                .map_err(inflow_error)?;
            let (rain, losses, source_bound) =
                totals.sources.as_ref().map_or((0.0, 0.0, 0.0), |s| {
                    (
                        s.cumulative_rain_volume_m3,
                        s.cumulative_bottom_loss_volume_m3 + s.cumulative_evaporation_volume_m3,
                        s.rain_roundoff_bound_m3
                            + s.bottom_loss_roundoff_bound_m3
                            + s.evaporation_roundoff_bound_m3,
                    )
                });
            terms.initial += totals.initial_volume_m3;
            terms.storage += totals.volume_m3;
            terms.external_in += rain + external.volume_m3;
            terms.external_out += losses;
            terms.pool_bound = totals.residual_m3.abs()
                + totals.reduction_bound_m3
                + source_bound
                + external.roundoff_bound_m3
                + external.reduction_bound_m3;
            terms.child_residual += terms.pool_bound;
            pool_magnitude = totals.cumulative_incoming_volume_m3
                + totals.cumulative_outgoing_volume_m3
                + rain
                + losses
                + external.volume_m3;
        }
        // Channel lateral totals hold the exchange bindings; the composite keeps that water inside.
        terms.external_in -= exchange.0;
        terms.external_out -= exchange.1;
        terms.magnitude = terms.initial
            + terms.storage
            + terms.external_in.abs()
            + terms.external_out.abs()
            + terms.adjustment.abs()
            + exchange.0
            + exchange.1
            + s.exchange_in_m3
            + s.exchange_out_m3;
        if pools.is_some() {
            terms.magnitude += pool_magnitude;
        }
        // Port water also stays inside: the reach boundary totals hold it.
        if !ports.is_empty() {
            let (into_reaches, out_of_reaches, _) = ports.reach_totals();
            terms.external_in -= into_reaches;
            terms.external_out -= out_of_reaches;
            terms.magnitude += 2.0 * (into_reaches + out_of_reaches);
        }
        // Pool-end water stays inside: the reach boundary totals and the pool natives hold it.
        if !ends.is_empty() {
            let (into_reaches, out_of_reaches, bound) = ends.reach_totals();
            terms.external_in -= into_reaches;
            terms.external_out -= out_of_reaches;
            terms.magnitude += 2.0 * (into_reaches + out_of_reaches);
            terms.child_residual += bound;
        }
        if let Some(r) = runoff {
            terms.initial += r.initial;
            terms.storage += r.storage;
            terms.external_in += r.rain - r.delivered;
            terms.external_out += r.losses;
            terms.adjustment += r.adjustment;
            terms.child_residual += r.bound;
            terms.magnitude += r.magnitude;
        }
        Ok(terms)
    }

    /// Lifetime channel exchange `(in, out, rounding bound)` from the reach counters.
    fn channel_lifetime(
        &self,
        reaches: &[ChannelReach<R>],
    ) -> Result<(f64, f64, f64), SurfaceChannelError> {
        let accounting: Vec<_> = self
            .linked
            .iter()
            .map(|entry| reaches[entry.reach].coupling_accounting())
            .collect();
        let volumes: Vec<&Tensor<R>> = accounting.iter().map(|a| a.lateral_volume_m3).collect();
        let bounds: Vec<&Tensor<R>> = accounting
            .iter()
            .map(|a| a.lateral_counter_roundoff_bound_m3)
            .collect();
        let (inflow, withdrawal) = self.link_bindings(&volumes)?;
        let (inflow_bound, withdrawal_bound) = self.link_bindings(&bounds)?;
        Ok((
            self.total(&inflow)?,
            self.total(&withdrawal)?,
            self.total(&inflow_bound)? + self.total(&withdrawal_bound)?,
        ))
    }

    /// Composite lifetime balance. Exchange stays inside and appears only in its totals.
    ///
    /// `runoff` holds the runoff child. Delivered runoff stays inside.
    pub fn balance(
        &self,
        config: &SurfaceChannelConfig,
        children: Children<'_, R>,
        counters: &LinkCounters<R>,
        runoff: Option<&SubcatchmentBatch<R>>,
    ) -> Result<SurfaceChannelWaterBalance, SurfaceChannelError> {
        let (surface, reaches, pools) = children;
        let (channel_in, channel_out, channel_bound) = self.channel_lifetime(reaches)?;
        let ports = &counters.ports;
        let runoff = self.runoff_terms(runoff.zip(counters.runoff.as_ref()), reaches, pools)?;
        let terms = self.terms(
            children,
            (channel_in, channel_out),
            (ports, &counters.pool_ends),
            runoff.as_ref(),
        )?;
        let s = surface.ledger();
        let residual = (terms.storage - terms.initial)
            - (terms.external_in - terms.external_out + terms.adjustment);
        let scale = terms
            .storage
            .max(terms.initial)
            .max(terms.external_in.abs())
            .max(terms.external_out.abs());
        let link_bound = self.total(&counters.roundoff_bound_m3)?;
        let mut bound = config.mass_absolute_m3
            + config.mass_relative * scale
            + terms.child_residual
            + channel_bound
            + link_bound
            + COUNTER_ULPS * f64::EPSILON * terms.magnitude;
        // The surface exchange totals hold port water. The reported totals keep link water only.
        let (mut surface_in, mut surface_out) = (s.exchange_in_m3, s.exchange_out_m3);
        if !ports.is_empty() {
            let (_, _, port_bound) = ports.reach_totals();
            bound += port_bound + ports.paired_bound_m3;
            surface_in -= ports.surface_m3[0];
            surface_out -= ports.surface_m3[1];
        }
        Ok(SurfaceChannelWaterBalance {
            initial_storage_m3: terms.initial,
            storage_m3: terms.storage,
            external_in_m3: terms.external_in,
            external_out_m3: terms.external_out,
            water_adjustment_m3: terms.adjustment,
            surface_exchange_in_m3: surface_in,
            surface_exchange_out_m3: surface_out,
            channel_exchange_in_m3: channel_in,
            channel_exchange_out_m3: channel_out,
            runoff_storage_m3: runoff.map_or(0.0, |r| r.storage),
            runoff_delivered_m3: runoff.map_or(0.0, |r| r.delivered),
            residual_m3: residual,
            bound_m3: bound,
        })
    }

    /// Checks one accepted interval: every node applies the link volumes, and water balances.
    ///
    /// Each channel binding must hold its link volume bit for bit.
    /// The surface's, the channel's, and the pools' net exchange must each match the summed link
    /// volumes within the paired bound. The surface side removes the port water its exchange
    /// totals hold first. The pools' native incoming adds the external inflow `pool_inflow`.
    /// Storage change minus external water must stay within the mass bound. Port water stays
    /// inside: it leaves the reach boundary totals, and each port's pairing bound joins the bound.
    ///
    /// `counters` holds the lifetime counters before and after the interval, then its port
    /// receipt. The pools' native net exchange adds every pool-end volume of `ends`. Pool-end
    /// water stays inside: it leaves the reach boundary totals.
    pub fn step_totals(
        &self,
        config: &SurfaceChannelConfig,
        (before, after): (Children<'_, R>, Children<'_, R>),
        exchange: &Exchange<R>,
        counters: (&LinkCounters<R>, &LinkCounters<R>, &PortInterval),
        (pool_inflow, ends): (PoolInflowStep, &PoolEndInterval),
        runoff: [Option<&SubcatchmentBatch<R>>; 2],
    ) -> Result<StepTotals, SurfaceChannelError> {
        let (counters_before, counters_after, port_step) = counters;
        let (ports_before, ports_after) = (&counters_before.ports, &counters_after.ports);
        let ported = !ports_after.is_empty();
        let (end_net, end_magnitude) = ends.totals();
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let signed = &exchange.signed;
        let to_b = c.where_cond(&c.gt(signed, &zero)?, signed, &zero)?;
        let to_a = c.where_cond(&c.lt(signed, &zero)?, &c.neg(signed)?, &zero)?;
        let at_channel = |into_b: &Tensor<R>, into_a: &Tensor<R>| {
            c.index_select(
                &c.where_cond(&self.channel_b, into_b, into_a)?,
                0,
                &self.channel_links,
            )
        };
        let to_channel = at_channel(&to_b, &to_a)?;
        let to_surface = at_channel(&to_a, &to_b)?;
        let accounting: Vec<_> = self
            .linked
            .iter()
            .map(|entry| after.1[entry.reach].coupling_accounting())
            .collect();
        let mut interval = Vec::with_capacity(accounting.len());
        for item in &accounting {
            interval.push(item.lateral_interval_volume_m3.ok_or_else(|| {
                SurfaceChannelError::Numerical("advanced reach lacks its lateral receipt".into())
            })?);
        }
        let (channel_inflow, channel_withdrawal) = self.link_bindings(&interval)?;
        let differs = |a: &Tensor<R>, b: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.gt(&c.abs(&c.sub(a, b)?)?, &zero)?)
        };
        self.reject(
            &c.maximum(
                &differs(&channel_inflow, &to_channel)?,
                &differs(&channel_withdrawal, &to_surface)?,
            )?,
            "a channel accepted a link volume that differs from the coordinator volume",
        )?;
        let channel_in = self.total(&channel_inflow)?;
        let channel_out = self.total(&channel_withdrawal)?;
        let net = |sign: &Tensor<R>| self.total(&c.mul(signed, sign)?);
        let surface_net = net(&self.surface_sign)?;
        let channel_net = net(&self.channel_sign)?;
        let magnitude = self.total(&c.abs(signed)?)?;
        let (s0, s1) = (before.0.ledger(), after.0.ledger());
        let mut surface_in = s1.exchange_in_m3 - s0.exchange_in_m3;
        let mut surface_out = s1.exchange_out_m3 - s0.exchange_out_m3;
        if ported {
            surface_in -= port_step.surface_m3[0];
            surface_out -= port_step.surface_m3[1];
        }
        let pool_native = match after.2.and_then(|p| p.batch.coupling_native_interval()) {
            Some(native) => Some((
                self.total(native.incoming_volume_m3)?,
                self.total(native.outgoing_volume_m3)?,
            )),
            None => None,
        };
        let width = (2 * (self.transfer_count() + self.group_count()) + 8) as f64;
        let pool_magnitude = pool_native.map_or(0.0, |(into, out)| into + out);
        let paired_bound = exchange.aggregation_bound
            + width
                * f64::EPSILON
                * (magnitude
                    + end_magnitude
                    + s1.exchange_in_m3
                    + s1.exchange_out_m3
                    + channel_in
                    + channel_out
                    + pool_magnitude);
        let surface_gap = (surface_out - surface_in) - surface_net;
        let channel_gap = (channel_in - channel_out) - channel_net;
        if !within(surface_gap, paired_bound) || !within(channel_gap, paired_bound) {
            return Err(SurfaceChannelError::Numerical(format!(
                "paired exchange: summed surface links={surface_net}, surface net out={}, summed channel links={channel_net}, channel net in={}, bound={paired_bound}: the two sides disagree; shorten the common interval",
                surface_out - surface_in,
                channel_in - channel_out
            )));
        }
        if let Some((into, out)) = pool_native {
            let pool_net = net(&self.pool_sign)?;
            let gap = if ends.volume_m3.is_empty() {
                (into - out) - (pool_net + pool_inflow.volume_m3)
            } else {
                (into - out) - (pool_net + end_net + pool_inflow.volume_m3)
            };
            let bound = paired_bound + pool_inflow.bound_m3;
            if !within(gap, bound) {
                return Err(SurfaceChannelError::Numerical(format!(
                    "paired pool exchange: summed pool links={pool_net}, pool ends={end_net}, external inflow={}, pool net in={}, bound={bound}: the pools disagree with their links; shorten the common interval",
                    pool_inflow.volume_m3,
                    into - out
                )));
            }
        }
        let (old_in, old_out, _) = self.channel_lifetime(before.1)?;
        let (new_in, new_out, _) = self.channel_lifetime(after.1)?;
        let [runoff_before, runoff_after] = runoff;
        let old_runoff = self.runoff_terms(
            runoff_before.zip(counters_before.runoff.as_ref()),
            before.1,
            before.2,
        )?;
        let new_runoff = self.runoff_terms(
            runoff_after.zip(counters_after.runoff.as_ref()),
            after.1,
            after.2,
        )?;
        let old = self.terms(
            before,
            (old_in, old_out),
            (ports_before, &counters_before.pool_ends),
            old_runoff.as_ref(),
        )?;
        let new = self.terms(
            after,
            (new_in, new_out),
            (ports_after, &counters_after.pool_ends),
            new_runoff.as_ref(),
        )?;
        let external = (new.external_in - old.external_in) - (new.external_out - old.external_out)
            + (new.adjustment - old.adjustment);
        let mass_residual = (new.storage - old.storage) - external;
        let scale = new
            .storage
            .max(old.storage)
            .max((new.external_in - old.external_in).abs())
            .max((new.external_out - old.external_out).abs());
        let mut child = (s1.residual - s0.residual).abs()
            + self
                .reach_order
                .iter()
                .map(|&r| {
                    (after.1[r].ledger().residual_m3 - before.1[r].ledger().residual_m3).abs()
                })
                .sum::<f64>();
        if after.2.is_some() {
            child += old.pool_bound + new.pool_bound;
        }
        if let (Some(old_runoff), Some(new_runoff)) = (old_runoff, new_runoff) {
            child += old_runoff.bound + new_runoff.bound;
        }
        let mut mass_bound = config.mass_absolute_m3
            + config.mass_relative * scale
            + child
            + 2.0 * paired_bound
            + COUNTER_ULPS * f64::EPSILON * (old.magnitude + new.magnitude);
        if ported {
            mass_bound += port_step.paired_bound_m3;
        }
        if !ends.volume_m3.is_empty() {
            mass_bound += counters_after.pool_ends.reach_totals().2;
        }
        if !within(mass_residual, mass_bound) {
            return Err(SurfaceChannelError::Numerical(format!(
                "composite water residual={mass_residual}, bound={mass_bound}: storage change departs from external water; shorten the common interval"
            )));
        }
        Ok(StepTotals {
            storage_before_m3: old.storage,
            storage_after_m3: new.storage,
            surface_in_m3: surface_in,
            surface_out_m3: surface_out,
            channel_in_m3: channel_in,
            channel_out_m3: channel_out,
            paired_bound_m3: paired_bound,
            mass_residual_m3: mass_residual,
            mass_bound_m3: mass_bound,
        })
    }
}
