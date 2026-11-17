use super::super::super::{
    point_flux_admission, point_flux_arithmetic, point_flux_types::PointFluxStatus,
    point_root_session::PointRootSession,
};
use super::{chart_test_support, diagnostics, frozen, hydraulic, support};
use crate::channel::flow::surveyed::error::{self, HydraulicError, PrismaticRiemannRejection};
use frozen::Result;
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    arithmetic::<R>(client)?;
    demand_and_isolation::<R>(client)
}

fn arithmetic<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let tiny = 2.0_f64.powi(-80);
    let (velocity, momentum) = hydraulic(error::checked::<R, _>(
        client,
        "independent exact dyadic momentum controls",
        |a| {
            let area =
                a.expansion_point(&Tensor::from_slice(&[2., 0.5, 1.], &[3], client.device())?)?;
            let discharge = a.expansion_point(&Tensor::from_slice(
                &[3., -0.25, 0.],
                &[3],
                client.device(),
            )?)?;
            let pressure = a.expansion_bounds(
                &Tensor::from_slice(&[0.625, 0.125, 1.], &[3], client.device())?,
                &Tensor::from_slice(&[0., 0., tiny], &[3], client.device())?,
                &Tensor::zeros(&[3], a.dtype(), client.device())?,
            )?;
            point_flux_arithmetic::momentum(a, &area, &discharge, &pressure)
        },
    ))?;
    let velocity = frozen::components(&velocity)?;
    let momentum = frozen::components(&momentum)?;
    for (row, expected_u, expected_j, expected_low) in
        [(0, 1.5, 5.125, 0.), (1, -0.5, 0.25, 0.), (2, 0., 1., tiny)]
    {
        let u = velocity[row].map(f64::from_bits);
        let j = momentum[row].map(f64::from_bits);
        assert_eq!([u[0], u[1]], [expected_u, 0.]);
        assert_eq!([j[0], j[1]], [expected_j, expected_low]);
        assert!(u[2] >= 0. && u[2] <= 2.0_f64.powi(-100));
        assert!(j[2] >= 0. && j[2] <= 2.0_f64.powi(-100));
    }
    Ok(())
}

fn demand_and_isolation<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 1., 1., 1.], [0., 0., 0., 0.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 1, 0, 1])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let before = work.snapshot();
            let empty = Tensor::empty(&[0], DType::I64, client.device())?;
            let empty = hydraulic(session.select_point_fluxes(&empty))?;
            assert!(empty.selected.is_empty());
            assert_eq!(empty.status.numel(), 0);
            for interval in [
                &empty.physical_water,
                &empty.physical_momentum,
                &empty.joint_water,
                &empty.joint_momentum,
            ] {
                assert_eq!(interval.lower().numel(), 0);
                assert_eq!(interval.upper().numel(), 0);
            }
            assert_eq!(work.snapshot(), before);
            let storage = Tensor::from_slice(&[777_i64, 2, 888, 0], &[2, 2], client.device())?;
            let demand = storage.narrow(1, 1, 1)?.squeeze(Some(1));
            assert!(!demand.is_contiguous());
            let result = hydraulic(session.select_point_fluxes(&demand))?;
            assert_eq!(
                result.candidates.invocation_owner.try_to_vec::<i64>()?,
                [2, 0]
            );
            assert_eq!(result.candidates.parent_rows.try_to_vec::<i64>()?, [0, 0]);
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [PointFluxStatus::Accepted as i64; 2]
            );
            assert_eq!(result.physical_water.lower().numel(), 2);
            assert_eq!(result.joint_momentum.upper().numel(), 2);
            diagnostics::check(&session, &result)?;
            let counts = work.snapshot();
            let callbacks = session.callback_points();
            for point in &result.selected {
                let zeros = Tensor::zeros(point.area.high().shape(), DType::F64, client.device())?;
                client.copy_into(point.area.high(), &zeros)?;
                client.copy_into(point.discharge.high(), &zeros)?;
                client.copy_into(point.query.query.parameter.lower(), &zeros)?;
            }
            let repeated = hydraulic(session.select_point_fluxes(&demand))?;
            diagnostics::check(&session, &repeated)?;
            assert_eq!(session.callback_points(), callbacks);
            assert_eq!(work.snapshot().residual_points, counts.residual_points);
            assert_eq!(work.snapshot().root_solves, counts.root_solves);
            assert_eq!(work.snapshot().root_iterations, counts.root_iterations);
            for point in &repeated.selected {
                for state in [
                    frozen::components(&point.area)?,
                    frozen::components(&point.discharge)?,
                ] {
                    assert!(
                        state
                            .iter()
                            .all(|value| *value == [1.0_f64.to_bits(), 0, 0])
                    );
                }
            }
            let query = &repeated
                .selected
                .first()
                .ok_or("No repeated selected point")?
                .query;
            let original = query.query.original_rows.try_to_vec::<i64>()?[0];
            let profile = query.query.owners.try_to_vec::<i64>()?[0];
            let rejected =
                error::checked::<R, _>(client, "typed selected positive geometry rejection", |a| {
                    point_flux_admission::positive(
                        a,
                        query,
                        &Tensor::ones(query.query.owners.shape(), a.dtype(), client.device())?,
                        "control depth",
                    )
                });
            assert!(matches!(rejected, Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::GeometryUnresolved,
            query: Some(row), section_owner: Some(owner), ..
        }) if row == usize::try_from(original)? && owner == usize::try_from(profile)?));
            drop(session);
            let foreign = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            assert!(
                foreign
                    .invocation
                    .check(&repeated.candidates.invocation_witness)
                    .is_err()
            );
            Ok(())
        },
    )
}
