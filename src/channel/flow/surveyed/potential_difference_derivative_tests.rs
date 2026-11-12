use super::error::{Arithmetic, HydraulicResult, PrismaticRiemannError, checked, into_public};
use super::potential_difference_derivatives::normalized_fourth_bound;
use super::potential_difference_test_data::{GRAVITY, PANELS};
use super::potential_difference_test_support::operand;
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
        let panel_bound = checked::<R, _>(&client, "normalized whole-panel fourth bound", |a| {
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
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
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
