//! Immutable surface source trial with separate native and SI arithmetic receipts.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
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
        + CorrectlyRoundedOps<R>
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
mod tests {
    use super::*;
    use crate::soil::{ModifiedGreenAmptParameters, NativeSoilState, SoilInput};
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::CpuRuntime;

    fn crossing_batch(
        suction: f64,
        memory: f64,
        conductivity: f64,
    ) -> Result<ModifiedGreenAmptBatch<CpuRuntime>, SoilError> {
        ModifiedGreenAmptBatch::<CpuRuntime>::from_native_states(
            cpu_client(),
            &[ModifiedGreenAmptParameters::from_native(
                suction,
                conductivity,
                1.0,
            )?],
            &[NativeSoilState {
                imd: 1.0,
                f_ft: memory,
                fu_ft: 0.0,
                saturated: false,
                t_s: 0.0,
            }],
        )
    }

    fn surface(
        batch: &ModifiedGreenAmptBatch<CpuRuntime>,
        dt: f64,
        rain: f64,
        ponding: f64,
    ) -> Result<SoilSurfaceTrial<CpuRuntime>, SoilError> {
        let tensor = |value| batch.context.tensor(&[value]);
        let c = &batch.context.client;
        let rain = tensor(rain)?;
        let ponding = tensor(ponding)?;
        let rain_depth = c.mul_scalar(&rain, dt)?;
        let available = c.add(&ponding, &rain_depth)?;
        batch.trial_surface_tensors(dt, &rain, &ponding, &rain_depth, &available)
    }

    #[test]
    fn native_crossing_excess_retains_source_result_and_state() -> Result<(), SoilError> {
        let ks = 1.0 / 1024.0;
        for (suction, memory, dt) in [
            (0.125, 0.25, 1.02e-12),
            (2f64.powi(52), 2f64.powi(53), 1126.4),
        ] {
            let batch = crossing_batch(suction, memory, ks)?;
            let rain = 1.5 * ks * METRES_PER_FOOT;
            let raw = batch.trial(
                dt,
                &[SoilInput {
                    incoming_rate_m_per_s: rain,
                    ponded_depth_m: 0.0,
                }],
            )?;
            let outputs = raw.outputs()?;
            let admitted = surface(&batch, dt, rain, 0.0)?;
            assert!(admitted.native_excess_depth.item::<f64>()? > 0.0);
            assert!(admitted.adjustment_depth.item::<f64>()? > 0.0);
            assert_eq!(
                admitted.native_infiltration_depth.item::<f64>()?.to_bits(),
                outputs[0].native_depth_ft.to_bits()
            );
            assert_eq!(
                admitted.infiltration_depth.item::<f64>()?.to_bits(),
                outputs[0].depth_m.to_bits()
            );
            assert_eq!(
                admitted.next.native_states()?,
                raw.into_next().native_states()?
            );
            assert_eq!(batch.native_states()?[0].f_ft, memory);
        }
        Ok(())
    }

    #[test]
    fn simple_supply_separates_si_bridge_from_native_bound() -> Result<(), SoilError> {
        let batch = ModifiedGreenAmptBatch::<CpuRuntime>::initialized(
            cpu_client(),
            &[ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?],
        )?;
        let rain = f64::from_bits(0x3e86d127d05394fe);
        let admitted = surface(&batch, 1.0, rain, 0.0)?;
        assert_eq!(admitted.native_roundoff_bound.item::<f64>()?, 0.0);
        assert_eq!(admitted.native_excess_depth.item::<f64>()?, 0.0);
        assert_eq!(
            admitted.adjustment_depth.item::<f64>()?,
            2.6469779601696886e-23
        );
        assert!(
            admitted.bridge_roundoff_bound.item::<f64>()?
                > admitted.adjustment_depth.item::<f64>()?
        );
        Ok(())
    }

    #[test]
    fn negative_crossing_time_uses_predicate_envelope() -> Result<(), SoilError> {
        let ks = 2f64.powi(-33);
        let batch = crossing_batch(f64::from_bits(1.0f64.to_bits() + 1) / 2.0, 1.0, ks)?;
        let dt = 2f64.powi(-20);
        let (raw, trace) = batch.compute_traced(
            dt,
            batch.context.tensor(&[1.5 * ks])?,
            batch.context.tensor(&[0.0])?,
        )?;
        assert_eq!(trace.crossing.item::<f64>()?, 1.0);
        assert!(trace.remaining_time_raw.item::<f64>()? < 0.0);
        let bound = Bounds {
            context: &batch.context,
        }
        .native(&trace, &raw.native_rate, &raw.native_depth, dt)?;
        let available = trace.raw_availability_rate.item::<f64>()? * dt;
        let excess = raw.native_depth.item::<f64>()? - available;
        assert!(excess > 0.0);
        assert!(
            crate::numerics::outward::upward_add(excess, 0.0).map_err(SoilError::from_outward)?
                <= bound.item::<f64>()?
        );
        let admitted = surface(&batch, dt, 1.5 * ks * METRES_PER_FOOT, 0.0)?;
        assert!(admitted.adjustment_depth.item::<f64>()? > 0.0);
        Ok(())
    }

    #[test]
    fn zero_infiltration_has_zero_adjustment_and_bounds() -> Result<(), SoilError> {
        let batch = ModifiedGreenAmptBatch::<CpuRuntime>::initialized(
            cpu_client(),
            &[ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?],
        )?;
        let trial = surface(&batch, 1.0, 1e-15, 0.0)?;
        for value in [
            &trial.infiltration_depth,
            &trial.adjustment_depth,
            &trial.native_roundoff_bound,
            &trial.bridge_roundoff_bound,
            &trial.total_roundoff_bound,
        ] {
            assert_eq!(value.item::<f64>()?, 0.0);
        }
        assert!(trial.native_available_depth.item::<f64>()? > 0.0);
        Ok(())
    }

    #[test]
    fn unproved_supply_excess_rejects_without_advancing() -> Result<(), SoilError> {
        let batch = ModifiedGreenAmptBatch::<CpuRuntime>::initialized(
            cpu_client(),
            &[ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?],
        )?;
        let before = batch.native_states()?;
        let rain = batch.context.tensor(&[1e-7])?;
        let zero = batch.context.tensor(&[0.0])?;
        let error = batch
            .trial_surface_tensors(1.0, &rain, &zero, &rain, &zero)
            .err()
            .ok_or_else(|| SoilError::numerical("unsupported SI discrepancy requires rejection"))?;
        assert_eq!(error.cell(), Some(0));
        assert!(error.is_retryable());
        assert_eq!(batch.native_states()?, before);
        Ok(())
    }

    fn cpu_client() -> numr::runtime::cpu::CpuClient {
        numr::runtime::cpu::CpuClient::new(numr::runtime::cpu::CpuDevice::new())
            .with_parallelism(numr::runtime::cpu::ParallelismConfig::new(Some(1), None))
    }

    #[test]
    fn source_tensor_dtype_requires_f64() -> Result<(), SoilError> {
        let batch = ModifiedGreenAmptBatch::<CpuRuntime>::initialized(
            cpu_client(),
            &[ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?],
        )?;
        let wrong = numr::tensor::Tensor::<CpuRuntime>::from_slice(
            &[0.0f32],
            &[1],
            batch.context.client.device(),
        )?;
        let zero = batch.context.tensor(&[0.0])?;
        let result = batch.trial_surface_tensors(1.0, &wrong, &zero, &zero, &zero);
        assert!(result.is_err());
        if let Err(error) = result {
            assert!(!error.is_retryable());
            assert!(error.to_string().contains("require F64 tensors"));
        }
        assert_eq!(batch.native_states()?[0].f_ft.to_bits(), 0.0f64.to_bits());
        Ok(())
    }
}
