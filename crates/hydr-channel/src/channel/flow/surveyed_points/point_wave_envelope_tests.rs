use super::super::corpus_coverage::Coverage;
use super::super::{
    chart_test_support,
    point_candidates::tests::{fixtures, wave_references},
    point_flux_types::PointFluxStatus,
    point_root_session::PointRootSession,
    source_tests::support as frozen,
    wet_depth::tests::support,
};
use crate::channel::flow::surveyed::error::checked;
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

/// Every frozen edge of every evaluated corpus row lies inside its face envelope.
fn corpus<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let executions = coverage.select(fixtures::executions(), |row| (row.outcome, row.physical));
    let mut evaluated = 0;
    for (config_index, declared) in fixtures::CONFIGS.iter().enumerate() {
        let rows = executions
            .iter()
            .filter(|e| e.config == config_index)
            .collect::<Vec<_>>();
        if rows.is_empty() || rows.iter().all(|row| row.outcome == "WorkLimit") {
            continue;
        }
        let ids = rows.iter().map(|row| row.id).collect::<Vec<_>>();
        let config = declared.build();
        chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, _| {
            let parents = (0..rows.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let parents = support::rows::<R>(client, &parents)?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let result = hydraulic(session.resolve_point_fluxes(&parents))?;
            let bounds = frozen::endpoints(&result.wave_speed_m_per_s)?;
            let valid = result.wave_valid.try_to_vec::<f64>()?;
            let status = result.status.try_to_vec::<i64>()?;
            for (i, row) in rows.iter().enumerate() {
                if status[i] == PointFluxStatus::PendingCandidates as i64 {
                    assert!(
                        row.outcome == "CapExceeded" || row.outcome == "physical_cap_rejection",
                        "{} loses its candidates",
                        row.id
                    );
                    continue;
                }
                assert_eq!(status[i], PointFluxStatus::Accepted as i64, "{}", row.id);
                assert_eq!(valid[i], 1., "{} lacks a certified envelope", row.id);
                if row.physical == "both_dry" {
                    assert_eq!(bounds[i], [0., 0.], "{}", row.id);
                }
                for (edge, label) in wave_references::expected(row) {
                    assert!(
                        bounds[i][0] <= edge[0] && edge[1] <= bounds[i][1],
                        "{} {label} {edge:?} escapes the envelope {:?}",
                        row.id,
                        bounds[i]
                    );
                }
                evaluated += 1;
            }
            Ok(())
        })?;
    }
    if coverage.is_full() {
        assert_eq!(evaluated, 66);
    }
    Ok(())
}

/// Still water on a rectangle reaches ±√(gh). A both-dry face is exactly [0, 0].
/// Removing a possible branch speed rejects only its face.
fn controls<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    // Width 1 makes depth equal to area.
    let depth = 2.;
    let gh = config.gravity_m_per_s2 * depth;
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[depth, 0., depth, 0.], [0., 0., 0., 0.]],
        &config,
        |pair, _| {
            let owners = support::rows::<R>(client, &[0, 1])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let result = hydraulic(session.resolve_point_fluxes(&owners))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [PointFluxStatus::Accepted as i64; 2]
            );
            assert_eq!(result.wave_valid.try_to_vec::<f64>()?, [1., 1.]);
            let bounds = frozen::endpoints(&result.wave_speed_m_per_s)?;
            // The sign of x² - gh is exact under one fused rounding.
            let reaches = |x: f64| x > 0. && x.mul_add(x, -gh) >= 0.;
            assert!(reaches(-bounds[0][0]), "{:?} misses -√(gh)", bounds[0]);
            assert!(reaches(bounds[0][1]), "{:?} misses +√(gh)", bounds[0]);
            // Tightness: still water carries no wave faster than the gravity wave.
            let sound = gh.sqrt();
            assert!(-bounds[0][0] - sound < 1e-12 && bounds[0][1] - sound < 1e-12);
            assert_eq!(bounds[1], [0., 0.]);

            let mut selection = hydraulic(session.select_point_fluxes(&owners))?;
            let waves = &mut selection.candidates.waves;
            assert!(!waves.is_empty());
            for wave in waves.iter_mut() {
                let side = &mut wave.sides[0];
                assert_eq!(side.equality_possible.try_to_vec::<f64>()?, [1.]);
                let available = &side.head.available;
                side.head.available =
                    Tensor::zeros(available.shape(), available.dtype(), available.device())?;
            }
            let status = selection.status.clone();
            let (envelope, status) = hydraulic(checked::<R, _>(client, "wave envelope", |a| {
                super::resolve(a, 2, &owners, std::slice::from_ref(&selection), &status)
            }))?;
            assert_eq!(envelope.valid.try_to_vec::<f64>()?, [0., 1.]);
            assert_eq!(
                status.try_to_vec::<i64>()?,
                [
                    PointFluxStatus::WaveEnvelopeUnresolved as i64,
                    PointFluxStatus::Accepted as i64
                ]
            );
            assert_eq!(
                frozen::endpoints(&envelope.wave_speed_m_per_s)?,
                [[0., 0.], [0., 0.]]
            );
            let direct = hydraulic(checked::<R, _>(client, "wave envelope rows", |a| {
                super::bounds(a, 2, &selection.candidates)
            }))?;
            assert_eq!(direct.2.try_to_vec::<f64>()?, [0., 1.]);
            Ok(())
        },
    )
}

/// A star owner repeated inside one wave domain rejects its face and no other.
fn duplicate<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let wet = [2., 0., 2., 0.];
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[wet, wet, wet, [0., 0., 0., 0.]],
        &config,
        |pair, _| {
            let owners = support::rows::<R>(client, &[0, 1, 2, 3])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let mut selection = hydraulic(session.select_point_fluxes(&owners))?;
            assert_eq!(
                selection.status.try_to_vec::<i64>()?,
                [PointFluxStatus::Accepted as i64; 4]
            );
            let waves = &mut selection.candidates.waves;
            assert_eq!(waves.len(), 1);
            let star = &mut waves[0].star.invocation_owner;
            assert_eq!(star.try_to_vec::<i64>()?, [0, 1, 2]);
            // Owner 0 appears twice. Owner 1 loses its only wave domain.
            *star = support::rows::<R>(client, &[0, 0, 2])?;
            let status = selection.status.clone();
            let (envelope, status) = hydraulic(checked::<R, _>(client, "wave envelope", |a| {
                super::resolve(a, 4, &owners, std::slice::from_ref(&selection), &status)
            }))?;
            assert_eq!(envelope.valid.try_to_vec::<f64>()?, [0., 0., 1., 1.]);
            let unresolved = PointFluxStatus::WaveEnvelopeUnresolved as i64;
            let accepted = PointFluxStatus::Accepted as i64;
            assert_eq!(
                status.try_to_vec::<i64>()?,
                [unresolved, unresolved, accepted, accepted]
            );
            Ok(())
        },
    )
}

#[test]
fn configured_cpu_point_wave_envelopes_enclose_every_signal() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    let client = CpuClient::new(CpuDevice::new());
    controls::<CpuRuntime>(&client)?;
    duplicate::<CpuRuntime>(&client)?;
    corpus::<CpuRuntime>(&client, Coverage::Full)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_point_wave_envelopes_enclose_every_signal() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    controls::<CudaRuntime>(&client)?;
    duplicate::<CudaRuntime>(&client)?;
    corpus::<CudaRuntime>(&client, Coverage::Sample)
}
