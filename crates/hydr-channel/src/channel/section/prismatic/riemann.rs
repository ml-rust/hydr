use super::{
    batch::PrismaticSectionBatch, hydraulic_geometry::HydraulicGeometry, query_input,
    riemann_capture::input_error,
};
use crate::channel::flow::surveyed::{
    PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannResult, controller,
    error::{HydraulicResult, checked, into_public},
    work::WorkLedger,
};
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

impl<R> PrismaticSectionBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    /// Enclose horizontal surveyed-channel Godunov fluxes using the configured native client.
    ///
    /// Owners use I64. Conserved areas and discharges use F64 with matching rank-one shapes.
    /// Inputs receive independent native snapshots. Keep result storage and endpoint aliases immutable.
    pub fn riemann_flux_tensor(
        &self,
        owners: &Tensor<R>,
        left_area_m2: &Tensor<R>,
        left_discharge_m3_per_s: &Tensor<R>,
        right_area_m2: &Tensor<R>,
        right_discharge_m3_per_s: &Tensor<R>,
        config: &PrismaticRiemannConfig,
    ) -> Result<PrismaticRiemannResult<R>, PrismaticRiemannError> {
        let work = WorkLedger::new(config);
        let run = || -> HydraulicResult<PrismaticRiemannResult<R>> {
            config.check()?;
            for value in [
                left_area_m2,
                left_discharge_m3_per_s,
                right_area_m2,
                right_discharge_m3_per_s,
            ] {
                query_input::metadata(&self.context, owners, value)?;
            }
            u32::try_from(owners.numel()).map_err(|_| {
                input_error(format!(
                    "input count={} exceeds U32 indexing. Split the batch",
                    owners.numel()
                ))
            })?;
            let geometry = HydraulicGeometry::from_batch(self);
            let input =
                checked::<R, _>(geometry.client(), "surveyed Riemann input capture", |_| {
                    self.capture_riemann(
                        owners,
                        left_area_m2,
                        left_discharge_m3_per_s,
                        right_area_m2,
                        right_discharge_m3_per_s,
                    )
                })?;
            let solution = controller::solve(&geometry, &input, config, &work)?;
            Ok(PrismaticRiemannResult::from_solution(
                input,
                solution,
                work.snapshot(),
            ))
        };
        run().map_err(|error| into_public(error, work.snapshot()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::PrismaticRiemannRejection;
    use crate::channel::flow::surveyed::potential_difference_test_data::{
        CASES, GRAVITY, MOMENTUM, PROFILES, WATER,
    };
    use crate::channel::flow::surveyed::potential_difference_test_support::{batch, values};
    use crate::channel::section::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint};
    use numr::ops::{HypotOps, MidpointOps};
    use solvr::common::arithmetic::Interval;

    fn endpoints<R: Runtime<DType = DType>>(
        value: &Interval<R>,
        row: usize,
    ) -> Result<[f64; 2], numr::error::Error> {
        Ok([
            value.lower().narrow(0, row, 1)?.item::<f64>()?,
            value.upper().narrow(0, row, 1)?.item::<f64>()?,
        ])
    }

    fn contains(actual: [f64; 2], bits: [u64; 2]) {
        let truth = bits.map(f64::from_bits);
        assert!(
            actual[0] <= truth[0] && actual[1] >= truth[1],
            "actual={actual:?}, frozen={truth:?}"
        );
    }

    fn mixed<R>(client: R::Client) -> Result<(), Box<dyn std::error::Error>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        let definition = CrossSectionDefinition {
            id: "trapezoid".into(),
            vertical_datum: "Frozen local survey datum".into(),
            points: [(0.0, 4.0), (4.0, 0.0), (5.0, 0.0), (9.0, 4.0)]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: 4.0,
            profile_provenance: "Original frozen trapezoid Riemann references".into(),
        };
        let definitions = (0..3)
            .map(|index| {
                let mut value = definition.clone();
                value.id = format!("trapezoid-{index}");
                value
            })
            .collect();
        let batch =
            CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?;
        let mut config = PrismaticRiemannConfig {
            gravity_m_per_s2: 1.0,
            ..Default::default()
        };
        config.root.atol = 1e-15;
        config.root.rtol = 1e-15;
        let owners = [2_i64, 1, 2, 1];
        let left = [6.0, 2.0, 6.0, 2.0];
        let right = [0.5, 2.0, 0.5, 2.0];
        let tensor =
            |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], client.device());
        let q = tensor(&[0.0; 4])?;
        let result = batch.riemann_flux_tensor(
            &Tensor::from_slice(&owners, &[4], client.device())?,
            &tensor(&left)?,
            &q,
            &tensor(&right)?,
            &q,
            &config,
        )?;
        assert_eq!(result.owners().try_to_vec::<i64>()?, owners);
        assert_eq!(result.left_area_m2().try_to_vec::<f64>()?, left);
        assert_eq!(result.right_area_m2().try_to_vec::<f64>()?, right);
        assert_eq!(
            result.left_discharge_m3_per_s().try_to_vec::<f64>()?,
            [0.0; 4]
        );
        assert_eq!(
            result.right_discharge_m3_per_s().try_to_vec::<f64>()?,
            [0.0; 4]
        );
        let mut independent_values = 0;
        let mut independent_derivatives = 0;
        let mut easy_panels = 0;
        let mut strong_panels = 0;
        let mut easy_values = 0;
        let mut easy_derivatives = 0;
        for row in 0..4 {
            let zero = tensor(&[0.0])?;
            let singleton = batch.riemann_flux_tensor(
                &Tensor::from_slice(&[owners[row]], &[1], client.device())?,
                &tensor(&[left[row]])?,
                &zero,
                &tensor(&[right[row]])?,
                &zero,
                &config,
            )?;
            for (mixed, single) in [
                (
                    result.water_flux_m3_per_s(),
                    singleton.water_flux_m3_per_s(),
                ),
                (
                    result.momentum_flux_m4_per_s2(),
                    singleton.momentum_flux_m4_per_s2(),
                ),
                (result.trace_area_m2(), singleton.trace_area_m2()),
                (
                    result.trace_discharge_m3_per_s(),
                    singleton.trace_discharge_m3_per_s(),
                ),
                (result.trace_depth_m(), singleton.trace_depth_m()),
                (result.wave_speed_m_per_s(), singleton.wave_speed_m_per_s()),
            ] {
                assert_eq!(
                    endpoints(mixed, row)?,
                    endpoints(single, 0)?,
                    "scattered row={row}"
                );
            }
            for (mixed, single) in [
                (result.trace_dry_possible(), singleton.trace_dry_possible()),
                (result.trace_wet_possible(), singleton.trace_wet_possible()),
            ] {
                assert_eq!(
                    mixed.narrow(0, row, 1)?.item::<f64>()?,
                    single.item::<f64>()?
                );
            }
            let work = singleton.work();
            independent_values += work.simpson_value_points;
            independent_derivatives += work.simpson_derivative_points;
            if row % 2 == 0 {
                strong_panels = work.maximum_simpson_panels;
                contains(
                    endpoints(result.water_flux_m3_per_s(), row)?,
                    [0x4001218d797712e1, 0x4001218d797712e2],
                );
                contains(
                    endpoints(result.momentum_flux_m4_per_s2(), row)?,
                    [0x40082ebacda719ac, 0x40082ebacda719ad],
                );
                contains(
                    endpoints(result.trace_area_m2(), row)?,
                    [0x4003c9f122462c1a, 0x4003c9f122462c1b],
                );
                contains(
                    endpoints(result.trace_depth_m(), row)?,
                    [0x3ff267c54e60a775, 0x3ff267c54e60a776],
                );
            } else {
                easy_panels = work.maximum_simpson_panels;
                easy_values += work.simpson_value_points;
                easy_derivatives += work.simpson_derivative_points;
                assert_eq!(endpoints(result.water_flux_m3_per_s(), row)?, [0.0; 2]);
                contains(
                    endpoints(result.momentum_flux_m4_per_s2(), row)?,
                    [0x3feaaaaaaaaaaaaa, 0x3feaaaaaaaaaaaab],
                );
            }
        }
        assert!(
            strong_panels > easy_panels,
            "strong panels={strong_panels}, easy={easy_panels}"
        );
        assert!(result.work().simpson_value_points <= independent_values);
        assert!(result.work().simpson_derivative_points <= independent_derivatives);
        assert!(easy_values > 0 && easy_derivatives > 0);
        // Global refinement repeats each nonanalytic easy row at least once beyond its singleton acceptance.
        assert!(result.work().simpson_value_points < independent_values + easy_values);
        assert!(
            result.work().simpson_derivative_points < independent_derivatives + easy_derivatives
        );
        let mut limited = config;
        limited.max_refinements = 0;
        limited.max_simpson_panels = limited.initial_simpson_panels;
        let failed = batch.riemann_flux_tensor(
            &Tensor::from_slice(&[1_i64, 2, 1], &[3], client.device())?,
            &tensor(&[2.0, 6.0, 2.0])?,
            &tensor(&[0.0; 3])?,
            &tensor(&[2.0, 0.5, 2.0])?,
            &tensor(&[0.0; 3])?,
            &limited,
        );
        assert!(matches!(
            failed,
            Err(PrismaticRiemannError::Rejected {
                reason: PrismaticRiemannRejection::FluxBudgetExceeded,
                query: Some(1),
                section_owner: Some(2),
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn mixed_faces_retain_only_unresolved_native_work_cpu() -> Result<(), Box<dyn std::error::Error>>
    {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        mixed::<CpuRuntime>(client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mixed_faces_retain_only_unresolved_native_work_cuda()
    -> Result<(), Box<dyn std::error::Error>> {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        mixed::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }

    fn flux<R>(
        actual: &Interval<R>,
        reference: [u64; 2],
        reverse: bool,
        label: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        R: Runtime<DType = DType>,
    {
        let reference = reference.map(f64::from_bits);
        let reference = if reverse {
            [-reference[1], -reference[0]]
        } else {
            reference
        };
        for (&lower, &upper) in actual
            .lower()
            .try_to_vec::<f64>()?
            .iter()
            .zip(&actual.upper().try_to_vec::<f64>()?)
        {
            assert!(
                lower <= reference[0] && upper >= reference[1],
                "{label}: actual=[{lower},{upper}], reference={reference:?}"
            );
            let allowance = 1e-12 + 1e-12 * lower.abs().max(upper.abs());
            assert!(
                upper - lower <= allowance,
                "{label}: width={}, allowance={allowance}",
                upper - lower
            );
        }
        Ok(())
    }

    fn references<R>(client: R::Client) -> Result<(), Box<dyn std::error::Error>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        let case = &CASES[0];
        assert_eq!(case.id, "captured_weak_face");
        let batch = batch::<R>(&client, &PROFILES[case.profile])
            .map_err(|error| into_public(error, Default::default()))?;
        let mut config = PrismaticRiemannConfig {
            gravity_m_per_s2: f64::from_bits(GRAVITY),
            water_flux_atol_m3_per_s: 1e-12,
            momentum_flux_atol_m4_per_s2: 1e-12,
            flux_rtol: 1e-12,
            initial_simpson_panels: 4,
            max_simpson_panels: 4096,
            ..Default::default()
        };
        config.root.atol = 1e-15;
        config.root.rtol = 1e-15;
        config.root.max_iterations = 256;
        config.root.max_points = 1 << 20;
        for strided in [false, true] {
            for reverse in [false, true] {
                let left = f64::from_bits(0x4024000000000000);
                let right = f64::from_bits(0x4023f45a904bb43e);
                let discharge = f64::from_bits(0x3fac91cb1810b76a);
                let (la, lq, ra, rq) = if reverse {
                    (right, -discharge, left, 0.0)
                } else {
                    (left, 0.0, right, discharge)
                };
                let input = |value| {
                    values::<R, _>(&client, &[value; 3], strided)
                        .map_err(|error| into_public(error, Default::default()))
                };
                let owners = values::<R, _>(&client, &[0_i64; 3], strided)
                    .map_err(|error| into_public(error, Default::default()))?;
                let result = batch.riemann_flux_tensor(
                    &owners,
                    &input(la)?,
                    &input(lq)?,
                    &input(ra)?,
                    &input(rq)?,
                    &config,
                )?;
                flux(
                    result.water_flux_m3_per_s(),
                    WATER,
                    reverse,
                    "captured water flux",
                )?;
                flux(
                    result.momentum_flux_m4_per_s2(),
                    MOMENTUM,
                    false,
                    "captured momentum flux",
                )?;
                assert_eq!(result.left_area_m2().try_to_vec::<f64>()?, [la; 3]);
                assert_eq!(result.right_area_m2().try_to_vec::<f64>()?, [ra; 3]);
            }
        }
        Ok(())
    }

    #[test]
    fn captured_weak_face_encloses_independent_flux_cpu() -> Result<(), Box<dyn std::error::Error>>
    {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        references::<CpuRuntime>(client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn captured_weak_face_encloses_independent_flux_cuda() -> Result<(), Box<dyn std::error::Error>>
    {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
