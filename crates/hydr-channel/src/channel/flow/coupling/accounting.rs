use super::super::{
    ChannelError, ChannelReach,
    conserved_update::{exact, expanded, magnitude, transfer},
};
use super::ChannelCouplingAccounting;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> ChannelReach<R> {
    #[doc(hidden)]
    pub fn coupling_accounting(&self) -> ChannelCouplingAccounting<'_, R> {
        ChannelCouplingAccounting {
            cell_volume_m3: &self.accepted.volume.high,
            cell_volume_low_m3: &self.accepted.volume.low,
            lateral_volume_m3: &self.accepted.lateral.cumulative_volume,
            lateral_interval_volume_m3: self
                .accepted
                .last
                .as_ref()
                .map(|last| &last.lateral_receipt.volume),
            lateral_inflow_mask: &self.context.lateral.inflow_mask,
            lateral_counter_roundoff_bound_m3: &self.accepted.lateral.counter_roundoff_bound,
            face_transfer_underflow_bound_m3: self
                .accepted
                .last
                .as_ref()
                .map(|last| &last.transfer_underflow_bound),
        }
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + numr::ops::TensorOps<R>
        + numr::ops::NextRepresentableOps<R>,
{
    /// Bound on the water the stored interval update differs from its exact expansion update.
    #[doc(hidden)]
    pub fn coupling_water_assembly_bound_m3(&self) -> Result<Option<Tensor<R>>, ChannelError> {
        self.accepted
            .last
            .as_ref()
            .map(|last| {
                let c = &self.context.client;
                let count = self.context.count();
                let left = last.face_volume.narrow(0, 0, count)?;
                let right = last.face_volume.narrow(0, 1, count)?;
                // The stored update used these rounded values.
                let delta = c.sub(&left, &right)?;
                self.context
                    .finite(&delta, "channel transport water delta")?;
                let stored = (!last.forcing.lateral.has_nonzero_values)
                    .then(|| c.add(&last.previous_volume.high, &delta))
                    .transpose()?;
                if let Some(stored) = &stored {
                    self.context
                        .finite(stored, "channel transported water volume")?;
                }
                let (delta_error, stored_error) =
                    expanded(c, "channel transport water assembly", |a| {
                        let faces = exact(a, &last.face_volume, &last.face_volume_low)?;
                        let transfer = transfer(a, &faces, count)?;
                        let rounded = a.expansion_point(&delta)?;
                        let difference = a.expansion_sub(&transfer, &rounded)?;
                        let delta_error = magnitude(a, &difference)?;
                        let stored_error = match &stored {
                            Some(stored) => {
                                let previous = exact(
                                    a,
                                    &last.previous_volume.high,
                                    &last.previous_volume.low,
                                )?;
                                let update = a.expansion_add(&previous, &transfer)?;
                                let stored = a.expansion_point(stored)?;
                                let difference = a.expansion_sub(&update, &stored)?;
                                Some(magnitude(a, &difference)?)
                            }
                            None => None,
                        };
                        Ok((delta_error, stored_error))
                    })?;
                self.context
                    .finite(&delta_error, "channel transport subtraction bound")?;
                match stored_error {
                    Some(bound) => {
                        self.context
                            .finite(&bound, "channel water addition bound")?;
                        Ok(bound)
                    }
                    None => {
                        self.context.finite(
                            &last.lateral_water_assembly_bound,
                            "channel lateral water assembly bound",
                        )?;
                        self.context
                            .bound_add(&delta_error, &last.lateral_water_assembly_bound)
                    }
                }
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{interval_reach, zero_source};
    use super::*;
    use crate::channel::ChannelBoundary;
    use numr::runtime::cpu::CpuRuntime;
    use std::sync::Arc;

    fn accounting_receipt(
        previous: f64,
        left: f64,
        right: f64,
        active_lateral: bool,
    ) -> Result<ChannelReach<CpuRuntime>, ChannelError> {
        let reach = interval_reach()?;
        let mut accepted = reach
            .step(
                2.125,
                &super::super::super::ChannelForcingInterval {
                    start_s: 2.0,
                    end_s: 2.125,
                    left: ChannelBoundary::Wall,
                    right: ChannelBoundary::Wall,
                    lateral_inputs: vec![zero_source(crate::forcing::WaterAmount::Rate {
                        rate_m3_per_s: 0.0,
                    })],
                },
            )?
            .next;
        // Isolate retained arithmetic operands without changing the stepping calculation.
        let last = Arc::make_mut(accepted.accepted.last.as_mut().ok_or_else(|| {
            ChannelError::input("missing retained interval: calculate the accounting fixture")
        })?);
        last.previous_volume.high = accepted.context.tensor(&[previous])?;
        last.face_volume = accepted.context.tensor(&[left, right])?;
        last.forcing.lateral.has_nonzero_values = active_lateral;
        last.lateral_water_assembly_bound = accepted.context.tensor(&[0.125])?;
        Ok(accepted)
    }

    #[test]
    fn transport_accounting_includes_unforced_state_addition_rounding()
    -> Result<(), Box<dyn std::error::Error>> {
        let increment = 2.0_f64.powi(-54);
        let accepted = accounting_receipt(1.0, increment, 0.0, false)?;
        let snapshot = accepted.snapshot_data()?;
        assert_eq!(
            accepted
                .coupling_water_assembly_bound_m3()?
                .ok_or("missing water assembly bound")?
                .item::<f64>()?,
            increment
        );
        assert_eq!(accepted.snapshot_data()?, snapshot);
        Ok(())
    }

    #[test]
    fn transport_accounting_includes_face_difference_rounding()
    -> Result<(), Box<dyn std::error::Error>> {
        let increment = 2.0_f64.powi(-54);
        for active_lateral in [false, true] {
            let accepted = accounting_receipt(1.0, 1.0, -increment, active_lateral)?;
            let bound = accepted
                .coupling_water_assembly_bound_m3()?
                .ok_or("missing water assembly bound")?
                .item::<f64>()?;
            if active_lateral {
                assert!(bound >= 0.125 + increment);
                assert!(bound <= 0.125 + 4.0 * increment);
            } else {
                assert_eq!(bound, increment);
            }
        }
        Ok(())
    }

    #[test]
    fn active_lateral_accounting_counts_complete_assembly_charge_once()
    -> Result<(), Box<dyn std::error::Error>> {
        let increment = 2.0_f64.powi(-54);
        let accepted = accounting_receipt(1.0, increment, 0.0, true)?;
        assert_eq!(
            accepted
                .coupling_water_assembly_bound_m3()?
                .ok_or("missing water assembly bound")?
                .item::<f64>()?,
            0.125
        );
        assert!(
            interval_reach()?
                .coupling_water_assembly_bound_m3()?
                .is_none()
        );
        Ok(())
    }
}
