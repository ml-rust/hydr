use super::{
    PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen, hydraulic,
    support,
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    exact_contacts::<R>(client)?;
    uncertain_contact::<R>(client)?;
    vacuum_limits::<R>(client)
}

fn exact_contacts<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let states = [
        [1.0, -2.0, 1.0, 2.0],
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 2.0, 1.0, -2.0],
        [1.0, -3.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    support::with_rectangle::<R, _>(client, 1.0, 4.0, &states, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[0, 1, 2, 3, 4])?;
        let mut session = hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
        let owners = support::rows::<R>(client, &[0, 1, 2, 3, 4])?;
        for side in [SonicSide::Left, SonicSide::Right] {
            hydraulic(session.start_source_sonic(&owners, side))?;
        }
        let before = work.snapshot();
        let result = hydraulic(session.certify_source_sonic())?;
        assert_eq!(work.snapshot().residual_points, before.residual_points);
        let expected = [
            SourceSonicStatus::DryContact,
            SourceSonicStatus::SourceContact,
            SourceSonicStatus::NoRootSource,
            SourceSonicStatus::NoRootDry,
            SourceSonicStatus::DrySourceUnavailable,
        ]
        .map(|s| s as i64);
        assert_eq!(result.status[0].try_to_vec::<i64>()?, expected);
        assert_eq!(result.status[1].try_to_vec::<i64>()?, expected);
        assert_eq!(result.domains.len(), 4);
        for domain in &result.domains {
            super::diagnostics::check(domain)?;
            assert_eq!(domain.status.try_to_vec::<i64>()?, [0]);
            match domain.classification {
                SourceSonicStatus::DryContact => {
                    for value in [
                        &domain.query.area,
                        &domain.depth,
                        &domain.normalized_depth,
                        &domain.sound,
                        &domain.velocity,
                        &domain.discharge,
                    ] {
                        assert_eq!(frozen::endpoints(value)?, [[0.0, 0.0]]);
                    }
                }
                SourceSonicStatus::SourceContact => {
                    let sign = if domain.side == SonicSide::Left {
                        1.0
                    } else {
                        -1.0
                    };
                    assert_eq!(frozen::endpoints(&domain.query.area)?, [[1.0, 1.0]]);
                    assert_eq!(frozen::endpoints(&domain.discharge)?, [[sign, sign]]);
                    assert_eq!(
                        frozen::components(&domain.anchored_discharge)?,
                        frozen::components(&domain.source_discharge)?
                    );
                }
                _ => return Err("Noncontact outcome manufactured a contact query".into()),
            }
        }
        for evidence in &result.evidence {
            assert_eq!(evidence.status.try_to_vec::<i64>()?, expected);
            assert_eq!(evidence.dry_residual.available.try_to_vec::<f64>()?[4], 0.0);
            assert_eq!(
                evidence.source_residual.available.try_to_vec::<f64>()?[4],
                0.0
            );
        }
        Ok(())
    })
}

fn uncertain_contact<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let section = frozen::batch::<R>(client)?;
    let profile = frozen::cases()
        .iter()
        .find(|case| case.id == "uniform_cap_retained_deficit_dyadic_velocity")
        .ok_or("Missing independent cap profile")?
        .owner;
    let owners = support::rows::<R>(client, &[i64::try_from(profile)?])?;
    // Exact rational Q=14*binary64(0x3ff52a7fa9d2f8ea), frozen as two exact components.
    // The represented velocity is near sqrt(14/8), while the complete sound certificate overlaps it.
    let input = [
        [14.0_f64.to_bits(), 0, 0],
        [0x4032852fb49899cd, 0xbcd0000000000000, 0],
        [14.0_f64.to_bits(), 0, 0],
        [0xc032852fb49899cd, 0x3cd0000000000000, 0],
    ]
    .iter()
    .map(|point| frozen::expansion::<R>(client, &[*point], true))
    .collect::<Result<Vec<_>>>()?;
    chart_test_support::with_inputs(client, section, owners, input, &config, |pair, _| {
        let rows = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
        for side in [SonicSide::Left, SonicSide::Right] {
            hydraulic(session.start_source_sonic(&rows, side))?;
        }
        let result = hydraulic(session.certify_source_sonic())?;
        assert!(result.domains.is_empty());
        for status in &result.status {
            assert_eq!(
                status.try_to_vec::<i64>()?,
                [SourceSonicStatus::Unresolved as i64]
            );
        }
        for evidence in &result.evidence {
            let residual = frozen::endpoints(&evidence.source_residual.physical)?[0];
            assert!(residual[0] <= 0.0 && residual[1] >= 0.0);
            assert!(
                frozen::components(&evidence.source_residual.retained)?[0]
                    .map(f64::from_bits)
                    .iter()
                    .any(|&x| x != 0.0)
            );
        }
        Ok(())
    })
}

fn vacuum_limits<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    for (id, expected) in [
        (
            "scalar_rectangle_vacuum_threshold",
            SourceSonicStatus::DryContact,
        ),
        ("scalar_rectangle_vacuum", SourceSonicStatus::NoRootDry),
    ] {
        chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, _| {
            let rows = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            for side in [SonicSide::Left, SonicSide::Right] {
                hydraulic(session.start_source_sonic(&rows, side))?;
            }
            let result = hydraulic(session.certify_source_sonic())?;
            for status in &result.status {
                assert_eq!(status.try_to_vec::<i64>()?, [expected as i64]);
            }
            if expected == SourceSonicStatus::DryContact {
                assert_eq!(result.domains.len(), 2);
                for domain in &result.domains {
                    // Frozen sonic-dry-limit label and zero limiting front speed establish these values without Q/A.
                    for value in [
                        &domain.query.area,
                        &domain.depth,
                        &domain.velocity,
                        &domain.discharge,
                    ] {
                        assert_eq!(frozen::endpoints(value)?, [[0.0, 0.0]]);
                    }
                }
            } else {
                assert!(result.domains.is_empty());
                for evidence in &result.evidence {
                    assert!(frozen::endpoints(&evidence.dry_residual.physical)?[0][0] > 0.0);
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}
