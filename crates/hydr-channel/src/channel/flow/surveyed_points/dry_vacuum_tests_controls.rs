use super::super::super::source_sonic_types::{SonicSide, SourceSonicStatus};
use super::{
    DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus, PointRootSession, Result,
    chart_test_support, frozen, hydraulic, support,
};
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
    moving_contacts::<R>(client)?;
    retained_exterior::<R>(client)?;
    uncertain_front::<R>(client)?;
    uncertain_regime::<R>(client)?;
    exhausted::<R>(client)
}

fn retained_exterior<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let section = chart_test_support::rectangle::<R>(client, 1., 4.)?;
    let owners = support::rows::<R>(client, &[0])?;
    let low = 2.0_f64.powi(-80);
    let area = [1.0_f64.to_bits(), low.to_bits(), 0];
    let discharge = [3.0_f64.to_bits(), (3. * low).to_bits(), 0];
    let inputs = [area, discharge, [0; 3], [0; 3]]
        .iter()
        .map(|v| frozen::expansion::<R>(client, &[*v], true))
        .collect::<Result<Vec<_>>>()?;
    chart_test_support::with_inputs(client, section, owners, inputs, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let result = hydraulic(session.certify_dry_vacuum(&owners))?;
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [DryVacuumStatus::Complete as i64]
        );
        assert_eq!(result.candidates.len(), 1);
        let DryVacuumCandidate::Exterior(exterior) = &result.candidates[0] else {
            return Err("Moving retained source lacks its exterior certificate".into());
        };
        assert_eq!(frozen::components(&exterior.area)?, [area]);
        assert_eq!(frozen::components(&exterior.discharge)?, [discharge]);
        assert_eq!(
            frozen::components(exterior.recovery.requested_areas_m2())?,
            [area]
        );
        assert_eq!(
            exterior.query.query.original_rows.try_to_vec::<i64>()?,
            [101]
        );
        let selected = hydraulic(
            session
                .sources
                .select_rows(&exterior.parent_rows, &exterior.query),
        )?;
        let retained = hydraulic(crate::channel::flow::surveyed::error::checked::<R, _>(
            client,
            "retained exterior source expression",
            |a| {
                exterior
                    .query
                    .retained_endpoint_area(a, &selected.bands, &selected.boundaries)
            },
        ))?;
        assert_eq!(frozen::components(&retained)?, [area]);
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[[1., 3., 0., 0.]],
            &config,
            |other, _| {
                assert!(other.select_rows(&owners, &exterior.query).is_err());
                Ok(())
            },
        )?;
        let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
        client.copy_into(exterior.area.high(), &zero)?;
        client.copy_into(exterior.discharge.low(), &zero)?;
        client.copy_into(exterior.recovery.requested_areas_m2().low(), &zero)?;
        client.copy_into(
            &exterior.query.query.original_rows,
            &support::rows::<R>(client, &[99])?,
        )?;
        let before = work.snapshot();
        let repeated = hydraulic(session.certify_dry_vacuum(&owners))?;
        assert_eq!(work.snapshot(), before);
        let DryVacuumCandidate::Exterior(exterior) = &repeated.candidates[0] else {
            return Err("Repeated retained source loses its exterior certificate".into());
        };
        assert_eq!(frozen::components(&exterior.area)?, [area]);
        assert_eq!(frozen::components(&exterior.discharge)?, [discharge]);
        assert_eq!(
            frozen::components(exterior.recovery.requested_areas_m2())?,
            [area]
        );
        assert_eq!(
            exterior.query.query.original_rows.try_to_vec::<i64>()?,
            [101]
        );
        Ok(())
    })
}
fn moving_contacts<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let states = [
        [1., 2., 0., 0.],
        [0., 0., 1., -2.],
        [1., 1., 0., 0.],
        [0., 0., 1., -1.],
        [1., -2., 0., 0.],
        [0., 0., 1., 2.],
        [1., -1., 1., 3.],
        [1., -3., 1., 1.],
    ];
    support::with_rectangle::<R, _>(client, 1., 4., &states, &config, |pair, _| {
        let parents = support::rows::<R>(client, &[0, 1, 2, 3, 4, 5, 6, 7])?;
        let mut session = hydraulic(PointRootSession::new(
            pair,
            &parents,
            config.root.max_points,
        ))?;
        for owner in 0..8 {
            let owners = support::rows::<R>(client, &[owner])?;
            let result = hydraulic(session.certify_dry_vacuum(&owners))?;
            hydraulic(session.invocation.check(&result.invocation_witness))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [DryVacuumStatus::Complete as i64],
                "owner {owner}"
            );
            assert_eq!(result.callback_points, session.callback_points());
            assert_eq!(result.point_limit, config.root.max_points);
            let p = &result.possibilities;
            let masks = [
                &p.left_exterior,
                &p.left_fan,
                &p.dry,
                &p.right_fan,
                &p.right_exterior,
            ]
            .map(|m| m.try_to_vec::<f64>())
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
            let expected = match owner {
                0 => [1., 0., 0., 0., 0.],
                1 => [0., 0., 0., 0., 1.],
                2 => [1., 1., 0., 0., 0.],
                3 => [0., 0., 0., 1., 1.],
                4 => [0., 1., 1., 0., 0.],
                5 => [0., 0., 1., 1., 0.],
                6 => [0., 1., 0., 0., 0.],
                7 => [0., 0., 0., 1., 0.],
                _ => return Err("Unexpected moving control owner".into()),
            };
            for (actual, expected) in masks.iter().zip(expected) {
                assert_eq!(actual, &[expected], "owner {owner}");
            }
            for candidate in &result.candidates {
                match candidate {
                    DryVacuumCandidate::Exterior(exterior) => {
                        let source = states[usize::try_from(owner)?];
                        let i = exterior.side.index() * 2;
                        assert_eq!(exterior.invocation_owner.try_to_vec::<i64>()?, [owner]);
                        assert_eq!(exterior.parent_rows.try_to_vec::<i64>()?, [owner]);
                        assert_eq!(
                            frozen::components(&exterior.area)?,
                            [[source[i].to_bits(), 0, 0]]
                        );
                        assert_eq!(
                            frozen::components(&exterior.discharge)?,
                            [[source[i + 1].to_bits(), 0, 0]]
                        );
                        assert_eq!(
                            frozen::components(&exterior.velocity)?,
                            [[source[i + 1].to_bits(), 0, 0]]
                        );
                        assert_eq!(frozen::endpoints(&exterior.query.area)?, [[1., 1.]]);
                        let depth = frozen::endpoints(exterior.recovery.depth_m())?[0];
                        assert!(depth[0] <= 1. && 1. <= depth[1]);
                    }
                    DryVacuumCandidate::Dry(dry) => {
                        assert_eq!(dry.invocation_owner.try_to_vec::<i64>()?, [owner]);
                        assert_eq!(dry.parent_rows.try_to_vec::<i64>()?, [owner]);
                        assert_eq!(frozen::endpoints(&dry.query.area)?, [[0., 0.]]);
                    }
                    DryVacuumCandidate::Fan(fan) => {
                        assert!(owner == 6 || owner == 7);
                        let expected = [0x3fbc71c71c71c71c, 0x3fbc71c71c71c71d].map(f64::from_bits);
                        let area = frozen::endpoints(&fan.query.area)?[0];
                        assert!(area[0] <= expected[0] && expected[1] <= area[1]);
                    }
                    DryVacuumCandidate::PendingFan(_) => {
                        return Err("Exact moving control leaves pending fan".into());
                    }
                }
            }
            if owner >= 6 {
                assert_eq!(
                    result.regime.try_to_vec::<i64>()?,
                    [DryVacuumRegime::Touching as i64]
                );
                assert_eq!(frozen::components(&result.invariant.retained)?, [[0; 3]]);
                assert_eq!(result.invariant.available.try_to_vec::<f64>()?, [1.]);
                let speed = if owner == 6 { 1. } else { -1. };
                for edge in &result.edges {
                    let front = frozen::endpoints(&edge.front.physical)?[0];
                    assert!(front[0] <= speed && speed <= front[1]);
                    assert_eq!(edge.front_exact_zero.try_to_vec::<f64>()?, [0.]);
                }
            }
            if owner == 2 || owner == 3 {
                let side = if owner == 2 { 0 } else { 1 };
                assert_eq!(
                    result.edges[side].head_exact_zero.try_to_vec::<f64>()?,
                    [1.]
                );
                assert_eq!(
                    result.sonic_status[side].try_to_vec::<i64>()?,
                    [SourceSonicStatus::SourceContact as i64]
                );
                assert!(
                    !result
                        .candidates
                        .iter()
                        .any(|v| matches!(v, DryVacuumCandidate::Fan(_)))
                );
            }
            if owner == 4 || owner == 5 {
                let side = if owner == 4 { 0 } else { 1 };
                assert_eq!(
                    result.edges[side].front_exact_zero.try_to_vec::<f64>()?,
                    [1.]
                );
                assert_eq!(
                    result.sonic_status[side].try_to_vec::<i64>()?,
                    [SourceSonicStatus::DryContact as i64]
                );
            }
        }
        Ok(())
    })
}
fn uncertain_front<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let section = chart_test_support::rectangle::<R>(client, 3., 4.)?;
    let owners = support::rows::<R>(client, &[0])?;
    // Exact Q=-6+2^-1074 gives a positive, unrepresentable epsilon=2^-1074/3.
    let input = [
        [3.0_f64.to_bits(), 0, 0],
        [(-6.0_f64).to_bits(), 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    .iter()
    .map(|v| frozen::expansion::<R>(client, &[*v], true))
    .collect::<Result<Vec<_>>>()?;
    chart_test_support::with_inputs(client, section, owners, input, &config, |pair, _| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let result = hydraulic(session.certify_dry_vacuum(&owners))?;
        assert_eq!(
            result.regime.try_to_vec::<i64>()?,
            [DryVacuumRegime::LeftWet as i64]
        );
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [DryVacuumStatus::Pending as i64]
        );
        let front = frozen::endpoints(&result.edges[0].front.physical)?[0];
        assert!(front[0] <= 0. && front[1] >= 0.);
        assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [0.]);
        assert_eq!(result.possibilities.left_fan.try_to_vec::<f64>()?, [1.]);
        assert_eq!(result.possibilities.dry.try_to_vec::<f64>()?, [1.]);
        let pending = result
            .candidates
            .iter()
            .find_map(|v| {
                if let DryVacuumCandidate::PendingFan(p) = v {
                    Some(p)
                } else {
                    None
                }
            })
            .ok_or("Uncertain front loses pending fan")?;
        assert_eq!(pending.side, SonicSide::Left);
        assert_eq!(pending.invocation_owner.try_to_vec::<i64>()?, [0]);
        assert_eq!(pending.parent_rows.try_to_vec::<i64>()?, [0]);
        assert_eq!(
            pending.source_status.try_to_vec::<i64>()?,
            [SourceSonicStatus::Unresolved as i64]
        );
        Ok(())
    })
}
fn exhausted<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 0., 0.], [0., 0., 0., 0.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 1])?;
            let mut session = hydraulic(PointRootSession::new(pair, &parents, 1))?;
            let owner = support::rows::<R>(client, &[0])?;
            let result = hydraulic(session.certify_dry_vacuum(&owner))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [DryVacuumStatus::Pending as i64]
            );
            assert_eq!(
                result.sonic_status[0].try_to_vec::<i64>()?,
                [SourceSonicStatus::WorkLimit as i64]
            );
            assert_eq!(result.edges[0].head.available.try_to_vec::<f64>()?, [0.]);
            assert_eq!(result.edges[0].head_exact_zero.try_to_vec::<f64>()?, [0.]);
            assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [0.]);
            let before = work.snapshot();
            hydraulic(session.certify_dry_vacuum(&owner))?;
            assert_eq!(work.snapshot(), before);
            assert_eq!(session.callback_points(), 0);
            let dry = hydraulic(session.certify_dry_vacuum(&support::rows::<R>(client, &[1])?))?;
            assert_eq!(
                dry.status.try_to_vec::<i64>()?,
                [DryVacuumStatus::Complete as i64]
            );
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

fn uncertain_regime<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let section = chart_test_support::rectangle::<R>(client, 3., 4.)?;
    let owners = support::rows::<R>(client, &[0])?;
    // Equal low discharges cancel physically, while separate quotient radii do not prove equality.
    let inputs = [
        [3.0_f64.to_bits(), 0, 0],
        [(-6.0_f64).to_bits(), 1, 0],
        [3.0_f64.to_bits(), 0, 0],
        [6.0_f64.to_bits(), 1, 0],
    ]
    .iter()
    .map(|v| frozen::expansion::<R>(client, &[*v], true))
    .collect::<Result<Vec<_>>>()?;
    chart_test_support::with_inputs(client, section, owners, inputs, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let result = hydraulic(session.certify_dry_vacuum(&owners))?;
        assert_eq!(
            result.regime.try_to_vec::<i64>()?,
            [DryVacuumRegime::Unresolved as i64]
        );
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [DryVacuumStatus::RegimeUnresolved as i64]
        );
        let invariant = frozen::endpoints(&result.invariant.physical)?[0];
        assert!(invariant[0] <= 0. && invariant[1] >= 0.);
        assert!(
            frozen::components(&result.invariant.retained)?[0]
                .iter()
                .any(|&v| f64::from_bits(v) != 0.)
        );
        assert!(result.candidates.is_empty());
        assert_eq!(session.callback_points(), 0);
        assert_eq!(work.snapshot().residual_points, 0);
        for status in &result.sonic_status {
            assert_eq!(
                status.try_to_vec::<i64>()?,
                [SourceSonicStatus::NotRequested as i64]
            );
        }
        Ok(())
    })
}
