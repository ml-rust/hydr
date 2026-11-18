use crate::channel::flow::{
    ChannelError,
    conserved_update::{ExpansionClient, expanded},
    context::Context,
    geometry::TensorGeometry,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, MidpointOps, NextRepresentableOps,
    ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + CorrectlyRoundedOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + ExpansionClient<R>,
{
    pub(in crate::channel::flow) fn surveyed_radius_selection_bound(
        &self,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        selected: &Tensor<R>,
        magnitude: &Tensor<R>,
        active: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let TensorGeometry::Surveyed(geometry) = &self.geometry else {
            return Err(ChannelError::input(
                "surveyed radius selection requires surveyed geometry: select the matching source method",
            ));
        };
        let c = &self.client;
        let zero = self.constant(magnitude, 0.0)?;
        let one = self.constant(magnitude, 1.0)?;
        let lower = c.where_cond(active, lower, &one)?;
        let upper = c.where_cond(active, upper, &one)?;
        let radius = c.where_cond(active, selected, &one)?;
        let magnitude = c.where_cond(active, magnitude, &zero)?;
        self.finite(&lower, "surveyed Manning radius lower bound")?;
        self.finite(&upper, "surveyed Manning radius upper bound")?;
        if !self.all(&c.gt(&lower, &zero)?)? {
            return Err(ChannelError::numerical(
                "active surveyed Manning radius lower bound is nonpositive: refine recovery or rescale the profile",
            ));
        }
        let contained = c.mul(&c.ge(&radius, &lower)?, &c.le(&radius, &upper)?)?;
        if !self.all(&contained)? {
            return Err(ChannelError::numerical(
                "surveyed Manning selected radius leaves its enclosure: refine recovery or rescale the profile",
            ));
        }
        self.finite(&magnitude, "surveyed Manning original momentum magnitude")?;
        if !self.all(&c.ge(&magnitude, &zero)?)? {
            return Err(ChannelError::input(
                "surveyed Manning momentum magnitude is negative: supply the original absolute momentum",
            ));
        }
        let three = self.constant(&magnitude, 3.0)?;
        let absolute = self.constant(
            &magnitude,
            geometry.numerical.friction_geometry_atol_m4_per_s,
        )?;
        let relative = self.constant(&magnitude, geometry.numerical.friction_geometry_rtol)?;
        let (bound, allowance) = expanded::<R, _>(c, "surveyed Manning radius selection", |a| {
            let selected = a.point(&radius)?;
            let lo = a.point(&lower)?;
            let hi = a.point(&upper)?;
            let left = a.sub(&selected, &lo)?;
            let right = a.sub(&hi, &selected)?;
            let displacement = c.maximum(left.upper(), right.upper())?;
            let displacement = c.maximum(&displacement, &zero)?;
            let displacement = a.point(&displacement)?;
            let relative_displacement = a.div_positive(&displacement, &lower)?;
            let sensitivity = a.div_positive(&relative_displacement, &three)?;
            let original_magnitude = a.point(&magnitude)?;
            let bound = a.mul(&original_magnitude, &sensitivity)?;
            let relative = a.point(&relative)?;
            let relative_budget = a.mul(&relative, &original_magnitude)?;
            let absolute = a.point(&absolute)?;
            let allowance = a.add(&absolute, &relative_budget)?;
            Ok((bound.upper().clone(), allowance.lower().clone()))
        })?;
        let bound = c.where_cond(active, &bound, &zero)?;
        if !self.all(&c.le(&bound, &allowance)?)? {
            return Err(ChannelError::numerical(
                "surveyed Manning radius-selection bound exceeds its geometry allowance: refine recovery or review friction geometry tolerances",
            ));
        }
        Ok(bound)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::friction_test_data::{ODE, SELECTION};
    use super::super::fixture::{TestResult, reach, reach_with_client};
    use super::*;
    use numr::ops::HypotOps;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn six_frozen_radius_selections_check_the_production_geometry_gate() -> TestResult {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        check_radius_selections::<CpuRuntime>(client)
    }

    fn check_radius_selections<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + CompareOps<R>
            + CorrectlyRoundedOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>,
    {
        assert_eq!(SELECTION.len(), 6);
        let ordinary = ODE.first().ok_or("Missing ordinary Manning fixture")?;
        let reach = reach_with_client::<R>(client, ordinary, f64::from_bits(ordinary.operands[5]))?;
        let context = &reach.context;
        let active = context.tensor(&[1.0])?;
        for row in SELECTION {
            let [lower, upper, selected, momentum] = row.radii_momentum.map(f64::from_bits);
            let result = context.surveyed_radius_selection_bound(
                &context.tensor(&[lower])?,
                &context.tensor(&[upper])?,
                &context.tensor(&[selected])?,
                &context.tensor(&[momentum.abs()])?,
                &active,
            );
            if row.expected == "accept" {
                let actual = result?.item::<f64>()?;
                let exact = row.bound.map(f64::from_bits);
                assert!(
                    actual >= exact[1],
                    "{}: bound={actual:e}, exact={exact:?}",
                    row.id
                );
                assert!(actual <= 1e-9 + 1e-12 * momentum.abs(), "{}", row.id);
                if exact[1] == 0.0 {
                    assert_eq!(actual, 0.0, "{}", row.id);
                }
            } else {
                let error = result.err().ok_or("Expected radius-selection rejection")?;
                let text = error.to_string();
                let expected = if row.expected == "reject_selection" {
                    "enclosure"
                } else {
                    "geometry allowance"
                };
                assert!(text.contains(expected), "{}: {text}", row.id);
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn configured_cuda_checks_six_frozen_radius_selections() -> TestResult {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        check_radius_selections::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }

    #[test]
    fn inactive_radius_selection_masks_invalid_operands_before_arithmetic() -> TestResult {
        let ordinary = ODE.first().ok_or("Missing ordinary Manning fixture")?;
        let reach = reach(ordinary, 0.03)?;
        let context = &reach.context;
        let result = context.surveyed_radius_selection_bound(
            &context.tensor(&[0.0])?,
            &context.tensor(&[f64::INFINITY])?,
            &context.tensor(&[f64::NAN])?,
            &context.tensor(&[f64::MAX])?,
            &context.tensor(&[0.0])?,
        )?;
        assert_eq!(result.item::<f64>()?.to_bits(), 0.0_f64.to_bits());
        Ok(())
    }
}
