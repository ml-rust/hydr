//! Lifetime port counters: interval accumulation, host projection, and checked import.

use super::super::{SurfaceChannelError, SurfaceChannelPortCounterData, SurfaceChannelPortLedger};
use super::PortInterval;
use crate::numerics::outward::upward_add;

/// Lifetime port volumes, host resident, in port identity order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in super::super) struct PortCounters {
    /// Water each reach took from the surface, in m³.
    pub to_channel_m3: Vec<f64>,
    /// Water each reach gave to the surface, in m³.
    pub to_surface_m3: Vec<f64>,
    /// Upward-rounded sum of every addition error of both counters, in m³.
    pub roundoff_bound_m3: Vec<f64>,
    /// Signed volume of each port over the last accepted interval, `+0.0` before the first.
    pub last_signed_m3: Vec<f64>,
    /// Port water the surface counted into it and out of it, in m³.
    pub surface_m3: [f64; 2],
    /// Upward-rounded sum of every interval's pairing bound, in m³.
    pub paired_bound_m3: f64,
}

/// `a + b` with its exact rounding error.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_part = sum - a;
    let a_part = sum - b_part;
    (sum, (a - a_part) + (b - b_part))
}

impl PortCounters {
    /// Zero counters for `ports` ports.
    pub fn zero(ports: usize) -> Self {
        Self {
            to_channel_m3: vec![0.0; ports],
            to_surface_m3: vec![0.0; ports],
            roundoff_bound_m3: vec![0.0; ports],
            last_signed_m3: vec![0.0; ports],
            surface_m3: [0.0; 2],
            paired_bound_m3: 0.0,
        }
    }

    /// True without ports.
    pub fn is_empty(&self) -> bool {
        self.to_channel_m3.is_empty()
    }

    /// Lifetime totals over every port: into the reaches, out of them, and the rounding bound.
    pub fn reach_totals(&self) -> (f64, f64, f64) {
        (
            self.to_channel_m3.iter().sum(),
            self.to_surface_m3.iter().sum(),
            self.roundoff_bound_m3.iter().sum(),
        )
    }

    /// Adds one interval's port volumes with error-free addition bounds.
    ///
    /// Counters without ports stay `+0.0`: an outward rounding of `0 + 0` would not.
    pub fn advance(&self, interval: &PortInterval) -> Result<Self, SurfaceChannelError> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let numerical = |error: crate::numerics::outward::OutwardError| {
            SurfaceChannelError::Numerical(format!("port counters: {error}"))
        };
        let mut next = self.clone();
        for (index, transfer) in interval.transfers.iter().enumerate() {
            let (Some(to_channel), Some(to_surface), Some(bound), Some(last)) = (
                next.to_channel_m3.get_mut(index),
                next.to_surface_m3.get_mut(index),
                next.roundoff_bound_m3.get_mut(index),
                next.last_signed_m3.get_mut(index),
            ) else {
                return Err(SurfaceChannelError::Numerical(
                    "port counters lack a port of the interval".into(),
                ));
            };
            let signed = transfer.signed_volume_m3;
            *last = signed;
            let (forward, forward_low) = two_sum(*to_channel, signed.max(0.0));
            let (backward, backward_low) = two_sum(*to_surface, (-signed).max(0.0));
            *to_channel = forward;
            *to_surface = backward;
            if forward_low != 0.0 || backward_low != 0.0 {
                let low = upward_add(forward_low.abs(), backward_low.abs()).map_err(numerical)?;
                *bound = upward_add(*bound, low).map_err(numerical)?;
            }
        }
        next.surface_m3 = [
            self.surface_m3[0] + interval.surface_m3[0],
            self.surface_m3[1] + interval.surface_m3[1],
        ];
        next.paired_bound_m3 =
            upward_add(self.paired_bound_m3, interval.paired_bound_m3).map_err(numerical)?;
        Ok(next)
    }

    /// Host ledgers with ports in `ids` order.
    pub fn ledgers(&self, ids: &[String]) -> Vec<SurfaceChannelPortLedger> {
        ids.iter()
            .zip(&self.to_channel_m3)
            .zip(&self.to_surface_m3)
            .zip(&self.roundoff_bound_m3)
            .map(
                |(((id, &to_channel), &to_surface), &bound)| SurfaceChannelPortLedger {
                    port_id: id.clone(),
                    surface_to_channel_m3: to_channel,
                    channel_to_surface_m3: to_surface,
                    roundoff_bound_m3: bound,
                },
            )
            .collect()
    }

    /// Checkpoint data with ports in `ids` order.
    pub fn data(&self, ids: &[String]) -> SurfaceChannelPortCounterData {
        SurfaceChannelPortCounterData {
            ledgers: self.ledgers(ids),
            last_signed_volume_m3: self.last_signed_m3.clone(),
            surface_in_m3: self.surface_m3[0],
            surface_out_m3: self.surface_m3[1],
            paired_bound_m3: self.paired_bound_m3,
        }
    }

    /// Counters from checkpoint `data` for ports `ids` in ascending identity.
    ///
    /// Every volume and bound is finite and nonnegative. Each last signed volume is finite.
    /// Before the first interval (`started` false) every value is `+0.0`.
    ///
    /// # Errors
    /// Returns the name of the first field that breaks a rule.
    pub fn import(
        data: &SurfaceChannelPortCounterData,
        ids: &[String],
        started: bool,
    ) -> Result<Self, String> {
        let ledgers = &data.ledgers;
        if ledgers.len() != ids.len()
            || data.last_signed_volume_m3.len() != ids.len()
            || ledgers
                .iter()
                .zip(ids)
                .any(|(ledger, id)| ledger.port_id != *id)
        {
            return Err("ledgers".into());
        }
        let fresh = |value: f64| started || value.to_bits() == 0;
        let volume = |value: f64| value.is_finite() && value.is_sign_positive() && fresh(value);
        for (index, ledger) in ledgers.iter().enumerate() {
            let values = [
                ledger.surface_to_channel_m3,
                ledger.channel_to_surface_m3,
                ledger.roundoff_bound_m3,
            ];
            if !values.into_iter().all(volume) {
                return Err(format!("ledgers[{index}] ({:?})", ledger.port_id));
            }
        }
        for (index, &signed) in data.last_signed_volume_m3.iter().enumerate() {
            if !signed.is_finite() || !fresh(signed) {
                return Err(format!("last_signed_volume_m3[{index}]"));
            }
        }
        for (name, value) in [
            ("surface_in_m3", data.surface_in_m3),
            ("surface_out_m3", data.surface_out_m3),
            ("paired_bound_m3", data.paired_bound_m3),
        ] {
            if !volume(value) || (ids.is_empty() && value.to_bits() != 0) {
                return Err(name.into());
            }
        }
        Ok(Self {
            to_channel_m3: ledgers.iter().map(|l| l.surface_to_channel_m3).collect(),
            to_surface_m3: ledgers.iter().map(|l| l.channel_to_surface_m3).collect(),
            roundoff_bound_m3: ledgers.iter().map(|l| l.roundoff_bound_m3).collect(),
            last_signed_m3: data.last_signed_volume_m3.clone(),
            surface_m3: [data.surface_in_m3, data.surface_out_m3],
            paired_bound_m3: data.paired_bound_m3,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> Vec<String> {
        vec!["a".into(), "b".into()]
    }

    fn sample() -> PortCounters {
        PortCounters {
            to_channel_m3: vec![1.0, 0.0],
            to_surface_m3: vec![0.5, 2.0],
            roundoff_bound_m3: vec![0.0, 1e-16],
            last_signed_m3: vec![-0.25, 0.75],
            surface_m3: [2.5, 1.0],
            paired_bound_m3: 1e-15,
        }
    }

    #[test]
    fn checkpoint_data_round_trips_every_bit() -> Result<(), String> {
        let counters = sample();
        let data = counters.data(&ids());
        assert_eq!(data.ledgers[1].port_id, "b");
        assert_eq!(PortCounters::import(&data, &ids(), true)?, counters);
        let zero = PortCounters::zero(2);
        assert_eq!(
            PortCounters::import(&zero.data(&ids()), &ids(), false)?,
            zero
        );
        let none = PortCounters::zero(0);
        assert_eq!(PortCounters::import(&none.data(&[]), &[], false)?, none);
        Ok(())
    }

    #[test]
    fn import_names_the_first_forged_field() {
        let reject = |edit: fn(&mut SurfaceChannelPortCounterData), started: bool| {
            let mut data = sample().data(&ids());
            edit(&mut data);
            PortCounters::import(&data, &ids(), started).err()
        };
        assert_eq!(
            reject(|d| d.ledgers[0].port_id = "z".into(), true),
            Some("ledgers".into())
        );
        assert_eq!(
            reject(|d| d.ledgers[1].channel_to_surface_m3 = -1.0, true),
            Some("ledgers[1] (\"b\")".into())
        );
        assert_eq!(
            reject(|d| d.last_signed_volume_m3[0] = f64::NAN, true),
            Some("last_signed_volume_m3[0]".into())
        );
        assert_eq!(
            reject(|d| d.paired_bound_m3 = f64::INFINITY, true),
            Some("paired_bound_m3".into())
        );
        // Before the first interval every value is zero.
        assert_eq!(reject(|_| {}, false), Some("ledgers[0] (\"a\")".into()));
        // Without ports the surface totals hold no port water.
        let mut data = PortCounters::zero(0).data(&[]);
        data.surface_in_m3 = 1.0;
        assert_eq!(
            PortCounters::import(&data, &[], true).err(),
            Some("surface_in_m3".into())
        );
    }
}
