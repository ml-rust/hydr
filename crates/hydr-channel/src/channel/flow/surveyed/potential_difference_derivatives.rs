use super::equation_ops::{scalar, scale, select_interval};
use super::error::{Arithmetic, HydraulicResult};
use super::potential_difference_lanes::DifferenceLanes;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn integrand<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &DifferenceLanes<R>,
    owner: &Tensor<R>,
    s: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (area, sound, delta) = panel_state(a, lanes, owner, s)?;
    let relative = a.div_positive_interval(&delta, &area)?;
    Ok(a.mul(&relative, &sound)?)
}

pub(crate) fn panel_fourth_bound<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &DifferenceLanes<R>,
    owner: &Tensor<R>,
    s: &Interval<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (area, sound, delta) = panel_state(a, lanes, owner, s)?;
    let c = a.client();
    let absolute = c.maximum(&c.abs(delta.lower())?, &c.abs(delta.upper())?)?;
    let absolute = a.point(&absolute)?;
    normalized_fourth_bound(a, &sound, &absolute, &area)
}

pub(crate) fn normalized_fourth_bound<R>(
    a: &mut Arithmetic<'_, R>,
    sound: &Interval<R>,
    absolute_delta: &Interval<R>,
    area: &Interval<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    // Certified convex pressure gives C >= m*A and m >= 0, hence 2*m*D/C <= 2*D/A.
    // Leibniz then bounds |G''''| by (255/4)*c*(D/A)^5 throughout this panel.
    let relative = a.div_positive_interval(absolute_delta, area)?;
    let mut bound = a.mul(sound, &relative)?;
    for _ in 0..4 {
        bound = a.mul(&bound, &relative)?;
    }
    let bound = scale(a, &bound, 255.0 / 4.0)?;
    Ok(bound.upper().clone())
}

fn panel_state<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &DifferenceLanes<R>,
    owner: &Tensor<R>,
    s: &Interval<R>,
) -> HydraulicResult<(Interval<R>, Interval<R>, Interval<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let original = select_interval(a, &lanes.original, owner)?;
    let delta = select_interval(a, &lanes.delta, owner)?;
    let hull = select_interval(a, &lanes.hull, owner)?;
    let offset = a.mul(&delta, s)?;
    let affine = a.add(&original, &offset)?;
    // Every complete physical segment lies inside the endpoint hull, for either delta sign.
    let area = a.bounds(
        &c.maximum(affine.lower(), hull.lower())?,
        &c.minimum(affine.upper(), hull.upper())?,
    )?;
    let start = select_interval(a, &lanes.start_area, owner)?;
    let width = select_interval(a, &lanes.width, owner)?;
    let slope = select_interval(a, &lanes.slope, owner)?;
    let zero = scalar(a, 0.0)?;
    let local = a.sub(&area, &start)?;
    let local = a.bounds(&c.maximum(local.lower(), zero.lower())?, local.upper())?;
    let width_squared = a.mul(&width, &width)?;
    let variable = a.mul(&slope, &local)?;
    let variable = scale(a, &variable, 2.0)?;
    let squared_width = a.add(&width_squared, &variable)?;
    let top_width = a.sqrt_nonnegative(&squared_width)?;
    let depth = a.div_positive_interval(&area, &top_width)?;
    let squared_sound = a.mul(&lanes.gravity, &depth)?;
    let sound = a.sqrt_nonnegative(&squared_sound)?;
    Ok((area, sound, delta))
}

#[cfg(test)]
mod tests {
    use super::super::error::{PrismaticRiemannError, checked, into_public};
    use super::super::potential_difference_test_data::{GRAVITY, PANELS};
    use super::super::potential_difference_test_support::operand;
    use super::*;

    fn bound<R>(
        a: &mut Arithmetic<'_, R>,
        area: [u64; 2],
        squared_width: [u64; 2],
        absolute_delta: [u64; 2],
    ) -> HydraulicResult<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let area = operand(a, area)?;
        let squared_width = operand(a, squared_width)?;
        let width = a.sqrt_nonnegative(&squared_width)?;
        let gravity = operand(a, [GRAVITY; 2])?;
        let numerator = a.mul(&gravity, &area)?;
        let squared_sound = a.div_positive_interval(&numerator, &width)?;
        let sound = a.sqrt_nonnegative(&squared_sound)?;
        let absolute_delta = operand(a, absolute_delta)?;
        normalized_fourth_bound(a, &sound, &absolute_delta, &area)
    }

    fn references<R>(client: R::Client) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        assert_eq!(PANELS.len(), 7);
        let mut points = 0;
        for panel in PANELS {
            let panel_bound =
                checked::<R, _>(&client, "normalized whole-panel fourth bound", |a| {
                    bound(a, panel.area, panel.squared_width, panel.absolute_delta)
                })?
                .item::<f64>()?;
            assert!(
                panel_bound.is_finite() && panel_bound >= 0.0,
                "{}",
                panel.id
            );
            for point in panel.points {
                points += 1;
                assert!(
                    f64::from_bits(point.slope[0]) >= 0.0,
                    "{} convex slope",
                    panel.id
                );
                assert!(
                    f64::from_bits(point.convex_margin[0]) >= 0.0,
                    "{} C-m*A",
                    panel.id
                );
                let delta = point.delta.map(f64::from_bits);
                let absolute = delta[0].abs().max(delta[1].abs()).to_bits();
                let point_bound =
                    checked::<R, _>(&client, "normalized narrow-point fourth bound", |a| {
                        bound(a, point.area, point.squared_width, [absolute; 2])
                    })?
                    .item::<f64>()?;
                let reference = point.reference.map(f64::from_bits);
                let absolute_reference = reference[0].abs().max(reference[1].abs());
                assert!(
                    point_bound.is_finite() && point_bound >= absolute_reference,
                    "{} point: M4={point_bound}, independent={reference:?}",
                    panel.id
                );
                // Narrow corner truths check the proven bound without comparing interval-series overestimation.
                assert!(
                    panel_bound >= absolute_reference,
                    "{} panel: M4={panel_bound}, independent={reference:?}",
                    panel.id
                );
            }
        }
        assert_eq!(points, 87);
        Ok(())
    }

    #[test]
    fn normalized_fourth_bounds_enclose_independent_series_points_cpu()
    -> Result<(), PrismaticRiemannError> {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        references::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn normalized_fourth_bounds_enclose_independent_series_points_cuda()
    -> Result<(), Box<dyn std::error::Error>> {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
            .map_err(|error| into_public(error, Default::default()))?;
        Ok(())
    }
}
