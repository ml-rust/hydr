//! Scalar endpoint policies read into trace requests on the end cell's section.

use super::{client::TraceClient, interior::Interior};
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    boundary::{Bounds, Reservoir},
    context::Context,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, ConditionalOps, NextRepresentableOps, TwoSumOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

/// One scalar endpoint policy in the outward frame.
pub(super) enum TraceRequest<R: Runtime<DType = DType>> {
    /// `d_b = η - z_min` by two-sum, with its directed one-ulp enclosure.
    Stage { depth: Tensor<R>, bounds: Bounds<R> },
    /// Outward target `σ Q` and the original longitudinal `Q` bits.
    Discharge {
        outward: Tensor<R>,
        longitudinal: Tensor<R>,
    },
    /// Reservoir head `η - z_min` above the bed, its enclosure and entrance loss.
    ///
    /// `cap` holds when the head equals an exact cap depth: outgoing flow then takes `d_b = H`.
    Reservoir { reservoir: Reservoir<R>, cap: bool },
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// Reads `policy` at `interior`'s end. A dry `Stage` or reservoir head rejects, and so
    /// does a `Stage` depth above the cap.
    pub(super) fn surveyed_trace_request(
        &self,
        interior: &Interior<R>,
        policy: &ChannelTensorBoundary<R>,
    ) -> Result<TraceRequest<R>, ChannelError> {
        let c = &self.client;
        let cell = if interior.endpoint == 0 {
            0
        } else {
            self.count() - 1
        };
        let bed = self.bed.narrow(0, cell, 1)?;
        let zero = self.constant(&bed, 0.0)?;
        // The exact difference lies within one ulp of its rounded head, on the tail's side.
        let local = |elevation: &Tensor<R>| -> Result<(Tensor<R>, Bounds<R>, bool), ChannelError> {
            let (head, tail) = c.two_sum(elevation, &c.neg(&bed)?)?;
            self.finite(&head, "surveyed scalar boundary local head")?;
            let bounds = Bounds {
                lo: c.where_cond(&c.ge(&tail, &zero)?, &head, &c.next_down(&head)?)?,
                hi: c.where_cond(&c.le(&tail, &zero)?, &head, &c.next_up(&head)?)?,
            };
            let exact = tail.item::<f64>()? == 0.0;
            Ok((head, bounds, exact))
        };
        match policy {
            ChannelTensorBoundary::Stage { elevation_m } => {
                let (depth, bounds, _) = local(elevation_m)?;
                self.boundary_require(&c.gt(&bounds.lo, &zero)?, "trace depth is dry")?;
                self.boundary_require(
                    &c.le(&bounds.hi, interior.cap.lower())?,
                    "trace depth exceeds the geometric cap",
                )?;
                Ok(TraceRequest::Stage { depth, bounds })
            }
            ChannelTensorBoundary::Discharge { discharge_m3_per_s } => {
                let outward = if interior.endpoint == 0 {
                    c.neg(discharge_m3_per_s)?
                } else {
                    discharge_m3_per_s.clone()
                };
                Ok(TraceRequest::Discharge {
                    outward,
                    longitudinal: discharge_m3_per_s.clone(),
                })
            }
            ChannelTensorBoundary::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => {
                let (head, bounds, exact) = local(elevation_m)?;
                self.boundary_require(&c.gt(&bounds.lo, &zero)?, "trace depth is dry")?;
                let cap = &interior.cap;
                let cap = exact
                    && self.all(&c.eq(&head, cap.lower())?)?
                    && self.all(&c.eq(&head, cap.upper())?)?;
                Ok(TraceRequest::Reservoir {
                    reservoir: Reservoir {
                        head,
                        bounds,
                        coefficient: entrance_loss_coefficient.clone(),
                    },
                    cap,
                })
            }
            ChannelTensorBoundary::Wall
            | ChannelTensorBoundary::Exterior { .. }
            | ChannelTensorBoundary::FluxPort(_) => Err(ChannelError::numerical(
                "surveyed scalar trace requires Stage, Discharge or QuiescentReservoir: inspect direct-frame dispatch",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{interior, policy, reach};
    use super::*;
    use crate::channel::flow::{
        ChannelBoundary,
        surveyed_update::tests::{Client, TestResult, cpu},
    };
    use numr::runtime::cpu::CpuRuntime;

    /// The bed at 0.1 m makes `η - z_min` inexact, so the enclosure spans its two neighbours.
    fn stage_depth_encloses_the_exact_difference<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let bed = 0.1;
        let reach = reach::<R>(client, 5.125, 0.0, bed)?;
        let context = &reach.context;
        let interior = interior(&reach, 0)?;
        let elevation = 1.3;
        let stage = policy(
            &reach,
            &ChannelBoundary::Stage {
                elevation_m: elevation,
            },
        )?;
        let TraceRequest::Stage { depth, bounds } =
            context.surveyed_trace_request(&interior, &stage)?
        else {
            return Err("a stage request is required".into());
        };
        let depth = depth.item::<f64>()?;
        let (lower, upper) = (bounds.lo.item::<f64>()?, bounds.hi.item::<f64>()?);
        assert_eq!(depth.to_bits(), (elevation - bed).to_bits());
        // Knuth's two-sum tail is nonzero: exactly one side widens by one ulp.
        let virtual_bed = depth - elevation;
        let tail = (elevation - (depth - virtual_bed)) + (-bed - virtual_bed);
        assert_ne!(tail, 0.0);
        if tail > 0.0 {
            assert_eq!((lower, upper), (depth, depth.next_up()));
        } else {
            assert_eq!((lower, upper), (depth.next_down(), depth));
        }
        for (elevation, fragment) in [
            (bed - 0.5, "trace depth is dry"),
            (bed + 3.5, "geometric cap"),
        ] {
            let stage = policy(
                &reach,
                &ChannelBoundary::Stage {
                    elevation_m: elevation,
                },
            )?;
            let error = match context.surveyed_trace_request(&interior, &stage) {
                Ok(_) => return Err("the stage was admitted".into()),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(fragment), "{error}");
        }
        Ok(())
    }

    /// A reservoir head equal to the exact cap depth marks the outgoing cap equality.
    fn exact_cap_head_marks_the_cap_equality<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client, 5.125, 0.0, 0.0)?;
        let context = &reach.context;
        let interior = interior(&reach, 1)?;
        for (head, expected) in [(3.0, true), (2.5, false)] {
            let reservoir = policy(
                &reach,
                &ChannelBoundary::QuiescentReservoir {
                    elevation_m: head,
                    entrance_loss_coefficient: 0.0,
                },
            )?;
            let TraceRequest::Reservoir { cap, .. } =
                context.surveyed_trace_request(&interior, &reservoir)?
            else {
                return Err("a reservoir request is required".into());
            };
            assert_eq!(cap, expected, "head={head}");
        }
        Ok(())
    }

    #[test]
    fn cpu_stage_depth_encloses_the_exact_difference() -> TestResult {
        stage_depth_encloses_the_exact_difference::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_exact_cap_head_marks_the_cap_equality() -> TestResult {
        exact_cap_head_marks_the_cap_equality::<CpuRuntime>(cpu())
    }
}
