//! Paired exchange checks, composite water balance, and lifetime link and port counters.

use super::context::{Context, Ops, transfer_error};
use super::exchange::Exchange;
use super::ports::{PortCounters, PortInterval};
use super::state::{LinkCounters, StepTotals};
use super::{SurfaceChannelConfig, SurfaceChannelError, SurfaceChannelWaterBalance};
use crate::channel::ChannelReach;
use crate::numerics::water_transfers::upward_add;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Rounding allowance per lifetime counter magnitude, in units of `f64::EPSILON`.
const COUNTER_ULPS: f64 = 8.0;

/// External water terms of the surface and every reach, exchange excluded.
#[derive(Clone, Copy, Debug)]
struct Terms {
    initial: f64,
    storage: f64,
    external_in: f64,
    external_out: f64,
    adjustment: f64,
    child_residual: f64,
    magnitude: f64,
}

/// True when `|value| <= bound`. A NaN on either side fails.
pub(super) fn within(value: f64, bound: f64) -> bool {
    value.abs() <= bound
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Per-link `(inflow, withdrawal)` values from per-binding tensors of every linked reach.
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

    /// Surface and channel water terms in reach identity order.
    ///
    /// `exchange` holds the lifetime channel link totals. `ports` holds the lifetime port
    /// volumes, which the reach boundary totals hold.
    fn terms(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        exchange: (f64, f64),
        ports: &PortCounters,
    ) -> Terms {
        let s = surface.ledger();
        let mut terms = Terms {
            initial: s.initial_storage,
            storage: s.storage,
            external_in: s.rain + s.boundary_in,
            external_out: s.withdrawal + s.infiltration + s.evaporation + s.boundary_out,
            adjustment: s.water_adjustment,
            child_residual: s.residual.abs(),
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
        // Port water also stays inside: the reach boundary totals hold it.
        if !ports.is_empty() {
            let (into_reaches, out_of_reaches, _) = ports.reach_totals();
            terms.external_in -= into_reaches;
            terms.external_out -= out_of_reaches;
            terms.magnitude += 2.0 * (into_reaches + out_of_reaches);
        }
        terms
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
    pub fn balance(
        &self,
        config: &SurfaceChannelConfig,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        counters: &LinkCounters<R>,
    ) -> Result<SurfaceChannelWaterBalance, SurfaceChannelError> {
        let (channel_in, channel_out, channel_bound) = self.channel_lifetime(reaches)?;
        let ports = &counters.ports;
        let terms = self.terms(surface, reaches, (channel_in, channel_out), ports);
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
            residual_m3: residual,
            bound_m3: bound,
        })
    }

    /// Checks one accepted interval: both sides apply the link volumes, and water balances.
    ///
    /// Each channel binding must hold its link volume bit for bit.
    /// Each side's net exchange must match the summed link volumes within the paired bound.
    /// The surface side removes the port water its exchange totals hold first.
    /// Storage change minus external water must stay within the mass bound. Port water stays
    /// inside: it leaves the reach boundary totals, and each port's pairing bound joins the bound.
    ///
    /// `ports` holds the lifetime port counters before and after the interval, then its receipt.
    pub fn step_totals(
        &self,
        config: &SurfaceChannelConfig,
        before: (&Solver<R>, &[ChannelReach<R>]),
        after: (&Solver<R>, &[ChannelReach<R>]),
        exchange: &Exchange<R>,
        ports: (&PortCounters, &PortCounters, &PortInterval),
    ) -> Result<StepTotals, SurfaceChannelError> {
        let (ports_before, ports_after, port_step) = ports;
        let ported = !ports_after.is_empty();
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let signed = &exchange.signed;
        let to_channel = c.where_cond(&c.gt(signed, &zero)?, signed, &zero)?;
        let to_surface = c.where_cond(&c.lt(signed, &zero)?, &c.neg(signed)?, &zero)?;
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
        let net = self.total(signed)?;
        let magnitude = self.total(&c.abs(signed)?)?;
        let (s0, s1) = (before.0.ledger(), after.0.ledger());
        let mut surface_in = s1.exchange_in_m3 - s0.exchange_in_m3;
        let mut surface_out = s1.exchange_out_m3 - s0.exchange_out_m3;
        if ported {
            surface_in -= port_step.surface_m3[0];
            surface_out -= port_step.surface_m3[1];
        }
        let width = (2 * (self.link_count + self.surface_count + self.slot_count) + 8) as f64;
        let paired_bound = exchange.aggregation_bound
            + width
                * f64::EPSILON
                * (magnitude + s1.exchange_in_m3 + s1.exchange_out_m3 + channel_in + channel_out);
        let surface_gap = (surface_out - surface_in) - net;
        let channel_gap = (channel_in - channel_out) - net;
        if !within(surface_gap, paired_bound) || !within(channel_gap, paired_bound) {
            return Err(SurfaceChannelError::Numerical(format!(
                "paired exchange: summed links={net}, surface net out={}, channel net in={}, bound={paired_bound}: the two sides disagree; shorten the common interval",
                surface_out - surface_in,
                channel_in - channel_out
            )));
        }
        let (old_in, old_out, _) = self.channel_lifetime(before.1)?;
        let (new_in, new_out, _) = self.channel_lifetime(after.1)?;
        let old = self.terms(before.0, before.1, (old_in, old_out), ports_before);
        let new = self.terms(after.0, after.1, (new_in, new_out), ports_after);
        let external = (new.external_in - old.external_in) - (new.external_out - old.external_out)
            + (new.adjustment - old.adjustment);
        let mass_residual = (new.storage - old.storage) - external;
        let scale = new
            .storage
            .max(old.storage)
            .max((new.external_in - old.external_in).abs())
            .max((new.external_out - old.external_out).abs());
        let child = (s1.residual - s0.residual).abs()
            + self
                .reach_order
                .iter()
                .map(|&r| {
                    (after.1[r].ledger().residual_m3 - before.1[r].ledger().residual_m3).abs()
                })
                .sum::<f64>();
        let mut mass_bound = config.mass_absolute_m3
            + config.mass_relative * scale
            + child
            + 2.0 * paired_bound
            + COUNTER_ULPS * f64::EPSILON * (old.magnitude + new.magnitude);
        if ported {
            mass_bound += port_step.paired_bound_m3;
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

    /// Adds one interval's directional link and port volumes with error-free addition bounds.
    pub fn advance_counters(
        &self,
        counters: &LinkCounters<R>,
        signed: &Tensor<R>,
        ports: &PortInterval,
    ) -> Result<LinkCounters<R>, SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let to_channel = c.where_cond(&c.gt(signed, &zero)?, signed, &zero)?;
        let to_surface = c.where_cond(&c.lt(signed, &zero)?, &c.neg(signed)?, &zero)?;
        let (forward, forward_low) = c.two_sum(&counters.to_channel_m3, &to_channel)?;
        let (backward, backward_low) = c.two_sum(&counters.to_surface_m3, &to_surface)?;
        let low =
            upward_add(c, &c.abs(&forward_low)?, &c.abs(&backward_low)?).map_err(transfer_error)?;
        let roundoff_bound_m3 =
            upward_add(c, &counters.roundoff_bound_m3, &low).map_err(transfer_error)?;
        Ok(LinkCounters {
            to_channel_m3: forward,
            to_surface_m3: backward,
            roundoff_bound_m3,
            ports: counters.ports.advance(ports)?,
        })
    }
}
