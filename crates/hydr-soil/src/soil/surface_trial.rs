//! Immutable surface source trial with separate native and SI arithmetic receipts.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::SoilError;
use super::batch::ModifiedGreenAmptBatch;
use super::kernel::roundoff::Bounds;
use super::parameters::METRES_PER_FOOT;

#[derive(Debug)]
#[doc(hidden)]
pub struct SoilSurfaceTrial<R: Runtime<DType = DType>> {
    pub next: ModifiedGreenAmptBatch<R>,
    pub infiltration_depth: Tensor<R>,
    pub adjustment_depth: Tensor<R>,
    pub native_rate: Tensor<R>,
    pub native_available_depth: Tensor<R>,
    pub native_infiltration_depth: Tensor<R>,
    pub native_excess_depth: Tensor<R>,
    pub native_roundoff_bound: Tensor<R>,
    pub bridge_roundoff_bound: Tensor<R>,
    pub total_roundoff_bound: Tensor<R>,
}

impl<R> ModifiedGreenAmptBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Require round-to-nearest and gradual underflow, checked when the context is created.
    #[doc(hidden)]
    pub fn trial_surface_tensors(
        &self,
        dt: f64,
        rain_si: &Tensor<R>,
        ponding_si: &Tensor<R>,
        rain_depth_si: &Tensor<R>,
        available_si: &Tensor<R>,
    ) -> Result<SoilSurfaceTrial<R>, SoilError> {
        self.check_inputs(dt, rain_si.numel())?;
        let context = &self.context;
        let c = &context.client;
        for (name, value) in [
            ("SI incoming rate (m/s)", rain_si),
            ("SI ponding (m)", ponding_si),
            ("SI rain interval depth (m)", rain_depth_si),
            ("SI available depth (m)", available_si),
        ] {
            if value.shape() != [self.parameters().len()] {
                return Err(SoilError::input(format!(
                    "{name}: require a flat tensor of {} soil cells",
                    self.parameters().len()
                )));
            }
            context.finite(value, name)?;
            context.reject_mask(
                &c.lt(value, &context.zero)?,
                name,
                &[("value", value)],
                &[],
                "Require nonnegative source inputs",
            )?;
        }
        let native_rain = c.div_scalar(rain_si, METRES_PER_FOOT)?;
        let native_ponding = c.div_scalar(ponding_si, METRES_PER_FOOT)?;
        let ponding_rate = c.div_scalar(&native_ponding, dt)?;
        let (trial, trace) =
            self.compute_traced(dt, native_rain.clone(), native_ponding.clone())?;
        let native_available_depth = c.mul_scalar(&trace.raw_availability_rate, dt)?;
        let bounds = Bounds { context };
        let native_roundoff_bound =
            bounds.native(&trace, &trial.native_rate, &trial.native_depth, dt)?;
        let positive = c.gt(&trial.native_rate, &context.zero)?;
        let select = |x: &Tensor<R>| bounds.select(&positive, x);
        let input = bounds.sum(&[
            bounds.scale(&bounds.bound(&select(&native_rain)?)?, dt)?,
            bounds.bound(&select(&native_ponding)?)?,
            bounds.scale(&bounds.bound(&select(&ponding_rate)?)?, dt)?,
            bounds.scale(&bounds.bound(&select(&trace.raw_availability_rate)?)?, dt)?,
            bounds.bound(&select(&native_available_depth)?)?,
        ])?;
        let input = bounds.sum(&[
            bounds.scale(&input, METRES_PER_FOOT)?,
            bounds.bound(&select(rain_depth_si)?)?,
            bounds.bound(&select(available_si)?)?,
        ])?;
        let output = bounds.sum(&[
            bounds.scale(&bounds.bound(&select(&trial.si_rate)?)?, dt)?,
            bounds.bound(&select(&trial.si_depth)?)?,
            bounds.scale(
                &bounds.bound(&select(&trial.native_depth)?)?,
                METRES_PER_FOOT,
            )?,
        ])?;
        let bridge_roundoff_bound = select(&bounds.add(&input, &output)?)?;
        let total_roundoff_bound = select(&bounds.add(
            &bounds.scale(&native_roundoff_bound, METRES_PER_FOOT)?,
            &bridge_roundoff_bound,
        )?)?;
        context.finite(
            &native_available_depth,
            "native available interval depth (ft)",
        )?;
        let native_excess_depth = c.maximum(
            &c.sub(&trial.native_depth, &native_available_depth)?,
            &context.zero,
        )?;
        let native_excess = c.gt(&native_excess_depth, &context.zero)?;
        context.reject_mask(
            &c.mul(&native_excess, &c.gt(&bounds.upper(&native_excess_depth)?, &native_roundoff_bound)?)?,
            "native infiltration excess (ft)",
            &[("excess", &native_excess_depth), ("bound", &native_roundoff_bound)],
            &[("dt (s)", dt)],
            "Native excess lacks an arithmetic envelope. Shorten the interval or rescale event memory",
        )?;
        let adjustment_depth = c.maximum(&c.sub(&trial.si_depth, available_si)?, &context.zero)?;
        let adjusted = c.gt(&adjustment_depth, &context.zero)?;
        context.reject_mask(
            &c.mul(
                &adjusted,
                &c.gt(&bounds.upper(&adjustment_depth)?, &total_roundoff_bound)?,
            )?,
            "SI infiltration excess (m)",
            &[
                ("excess", &adjustment_depth),
                ("bound", &total_roundoff_bound),
            ],
            &[("dt (s)", dt)],
            "SI excess lacks an arithmetic envelope. Shorten the interval or rescale source inputs",
        )?;
        context.reject_mask(
            &c.mul(&adjusted, &c.ne(&c.add(available_si, &adjustment_depth)?, &trial.si_depth)?)?,
            "SI source reconciliation (m)",
            &[("available", available_si), ("adjustment", &adjustment_depth), ("infiltration", &trial.si_depth)],
            &[("dt (s)", dt)],
            "The adjustment cannot exactly reconcile source water. Shorten the interval or rescale source inputs",
        )?;
        Ok(SoilSurfaceTrial {
            next: trial.next,
            infiltration_depth: trial.si_depth,
            adjustment_depth,
            native_rate: trial.native_rate,
            native_available_depth,
            native_infiltration_depth: trial.native_depth,
            native_excess_depth,
            native_roundoff_bound,
            bridge_roundoff_bound,
            total_roundoff_bound,
        })
    }
}

#[cfg(test)]
mod tests;
