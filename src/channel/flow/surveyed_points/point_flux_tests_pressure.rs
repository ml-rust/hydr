use super::super::super::point_flux_pressure;
use super::{chart_test_support, frozen, hydraulic, support};
use crate::channel::flow::surveyed::error::checked;
use frozen::Result;
use numr::{dtype::DType, runtime::Runtime};

fn tiny_complete_pressure<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 1., 0.]],
        &config,
        |pair, work| {
            let area = 2.0_f64.powi(-164);
            let truth = 2.0_f64.powi(-329);
            let query = chart_test_support::chart(
                pair,
                chart_test_support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[area, area]],
            )?;
            let parent = support::rows::<R>(client, &[0])?;
            let (pressure, numerical) = hydraulic(checked::<R, _>(
                client,
                "independent tiny rectangle pressure control",
                |a| {
                    let pressure = point_flux_pressure::selected(a, pair, &parent, &query, 0)?;
                    let numerical = a.expansion_interval(&pressure.chosen)?;
                    Ok((pressure, numerical))
                },
            ))?;
            let physical = frozen::endpoints(&pressure.evaluation)?[0];
            let numerical = frozen::endpoints(&numerical)?[0];
            eprintln!(
                "POINT_FLUX_TINY_PRESSURE\ttruth={truth:e}\tphysical={physical:?}\tnumerical={numerical:?}\tanchor={:?}\tincrement={:?}\tselected_increment={:?}\tsecant={:?}",
                frozen::components(&pressure.source_anchor)?,
                frozen::components(&pressure.increment.retained_increment)?,
                frozen::components(&pressure.selected_increment)?,
                frozen::endpoints(&pressure.increment.secant)?
            );
            assert!(physical[0] <= truth && truth <= physical[1]);
            // The trace takes the model integral at its own area, not the source anchor and secant.
            assert_eq!(pressure.proposal_contained.try_to_vec::<f64>()?, [1.]);
            assert!(physical[0] <= numerical[0] && numerical[1] <= physical[1]);
            let error = (numerical[0] - truth)
                .abs()
                .max((numerical[1] - truth).abs());
            assert!(
                error / truth < 1e-10,
                "Selected pressure loses exact rectangle P=A^2/2: {numerical:?}, truth={truth:e}"
            );
            assert_eq!(work.snapshot().root_solves, 0);
            assert_eq!(work.snapshot().residual_points, 0);
            let identity_query = chart_test_support::chart(
                pair,
                chart_test_support::Anchor::Source,
                [0, 0],
                &[[0., 0.]],
            )?;
            let delta = 2.0_f64.powi(-80);
            let weak_query = chart_test_support::chart(
                pair,
                chart_test_support::Anchor::Source,
                [delta.to_bits(), 0],
                &[[1., 1.]],
            )?;
            let (identity, weak) = hydraulic(checked::<R, _>(
                client,
                "independent source and weak rectangle pressure controls",
                |a| {
                    Ok((
                        point_flux_pressure::selected(a, pair, &parent, &identity_query, 0)?,
                        point_flux_pressure::selected(a, pair, &parent, &weak_query, 0)?,
                    ))
                },
            ))?;
            assert_eq!(
                frozen::components(&identity.increment.retained_increment)?,
                [[0, 0, 0]]
            );
            let identity_bounds = frozen::endpoints(&identity.evaluation)?[0];
            assert!(identity_bounds[0] <= 0.5 && 0.5 <= identity_bounds[1]);
            // P(1+delta)-P(1)=delta+delta^2/2. These adjacent F64 values enclose that exact dyadic value.
            let expected = [delta, f64::from_bits(delta.to_bits() + 1)];
            let increment = frozen::endpoints(&weak.increment.physical_increment)?[0];
            assert!(increment[0] <= expected[0] && expected[1] <= increment[1]);
            assert!((increment[1] - increment[0]) / delta < 1e-10);
            let retained =
                frozen::components(&weak.increment.retained_increment)?[0].map(f64::from_bits);
            assert!(retained[0] > 0.);
            assert!((retained[0] + retained[1] - delta).abs() / delta < 1e-10);
            Ok(())
        },
    )
}

#[test]
fn configured_cpu_point_flux_pressure_preserves_tiny_complete_pressure() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    tiny_complete_pressure::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_point_flux_pressure_preserves_tiny_complete_pressure() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    tiny_complete_pressure::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
