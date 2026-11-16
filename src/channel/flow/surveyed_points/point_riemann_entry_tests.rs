use super::super::{
    corpus_coverage::Coverage,
    point_candidates::tests::{fixtures, flux_fixtures},
    point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
    source_tests::support::{self as frozen, Result, hydraulic},
    wet_depth::tests::support,
};
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannRejection, error::checked,
    },
    section::PrismaticSectionBatch,
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use std::collections::BTreeMap;

#[path = "point_riemann_entry_tests_exterior.rs"]
mod exterior;
#[path = "point_riemann_entry_tests_star.rs"]
mod star;

/// Runs one batch of frozen source cases through the public point entry.
fn call<R>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    ids: &[&str],
    config: &PrismaticRiemannConfig,
) -> Result<std::result::Result<PrismaticPointRiemannResult<R>, PrismaticRiemannError>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let cases = frozen::cases();
    let selected = ids
        .iter()
        .map(|id| {
            cases
                .iter()
                .find(|case| case.id == *id)
                .ok_or("Missing frozen point source")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let owners = Tensor::from_slice(
        &selected
            .iter()
            .map(|case| i64::try_from(case.owner))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        &[ids.len()],
        client.device(),
    )?;
    let inputs = (0..4)
        .map(|field| {
            frozen::expansion::<R>(
                client,
                &selected
                    .iter()
                    .map(|case| case.inputs[field])
                    .collect::<Vec<_>>(),
                true,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(section.riemann_flux_points_tensor(
        &owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3], config,
    ))
}

fn point<R>(client: &R::Client, value: &CertifiedExpansion<R>) -> Result<Vec<[f64; 2]>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let interval = hydraulic(checked::<R, _>(client, "selected point bounds", |a| {
        Ok(a.expansion_interval(value)?)
    }))?;
    frozen::endpoints(&interval)
}

fn contains(actual: [f64; 2], truth: [f64; 2], label: &str) {
    assert!(
        actual[0] <= truth[0] && truth[1] <= actual[1],
        "{label}: {actual:?} excludes {truth:?}"
    );
}

pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let executions = coverage.select(fixtures::executions(), |row| (row.outcome, row.physical));
    let references = flux_fixtures::REFERENCES
        .iter()
        .map(|r| (r.id, r))
        .collect::<BTreeMap<_, _>>();
    let section = frozen::batch::<R>(client)?;
    let mut counts = [0_usize; 3];
    for (config_index, original) in fixtures::CONFIGS.iter().enumerate() {
        let config = original.build();
        let rows = executions
            .iter()
            .filter(|row| row.config == config_index)
            .collect::<Vec<_>>();
        let mut accepted = Vec::new();
        for row in rows {
            let reference = references
                .get(row.id)
                .ok_or("Execution lacks independent full flux truth")?;
            let expected = if row.outcome == "WorkLimit" {
                Some(PrismaticRiemannRejection::WorkLimit)
            } else if reference.water.is_none() || reference.momentum.is_none() {
                Some(PrismaticRiemannRejection::CapExceeded)
            } else {
                None
            };
            let Some(expected) = expected else {
                accepted.push((row.id, *reference));
                continue;
            };
            match call::<R>(client, &section, &[row.id], &config)? {
                Err(PrismaticRiemannError::Rejected {
                    reason,
                    query,
                    section_owner,
                    ..
                }) => {
                    assert_eq!(reason, expected, "{}", row.id);
                    if reason == PrismaticRiemannRejection::CapExceeded {
                        let owner = frozen::cases()
                            .iter()
                            .find(|case| case.id == row.id)
                            .ok_or("Missing frozen point source")?
                            .owner;
                        assert_eq!(query, Some(0), "{}", row.id);
                        assert_eq!(section_owner, Some(owner), "{}", row.id);
                        counts[1] += 1;
                    } else {
                        counts[2] += 1;
                    }
                }
                other => {
                    return Err(format!(
                        "{} returns {:?} instead of {expected:?}",
                        row.id,
                        other.err()
                    )
                    .into());
                }
            }
        }
        if accepted.is_empty() {
            continue;
        }
        let ids = accepted.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let result = call::<R>(client, &section, &ids, &config)??;
        let physical = [
            frozen::endpoints(result.physical_water_flux_m3_per_s())?,
            frozen::endpoints(result.physical_momentum_flux_m4_per_s2())?,
        ];
        let bound = [
            frozen::endpoints(result.water_flux_bound_m3_per_s())?,
            frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?,
        ];
        let selected = [
            point::<R>(client, result.water_flux_m3_per_s())?,
            point::<R>(client, result.momentum_flux_m4_per_s2())?,
        ];
        let discharge = point::<R>(client, result.trace_discharge_m3_per_s())?;
        let kinds = result.trace_kind().try_to_vec::<i64>()?;
        let speeds = frozen::endpoints(result.wave_speed_m_per_s())?;
        assert_eq!(result.owners().numel(), ids.len());
        for (i, (id, reference)) in accepted.iter().enumerate() {
            let truth = [
                reference.water.ok_or("Flux water truth absent")?,
                reference.momentum.ok_or("Flux momentum truth absent")?,
            ]
            .map(|bits| bits.map(f64::from_bits));
            let mirror = frozen::exact_mirror(id)?;
            for water in [physical[0][i], bound[0][i]] {
                frozen::water_enclosed(water, truth[0], mirror, id);
            }
            if mirror {
                assert_eq!(
                    selected[0][i].map(f64::to_bits),
                    [0, 0],
                    "{id} selected water"
                );
            }
            contains(bound[0][i], selected[0][i], id);
            contains(physical[1][i], truth[1], id);
            contains(bound[1][i], truth[1], id);
            contains(bound[1][i], selected[1][i], id);
            assert_eq!(
                discharge[i], selected[0][i],
                "{id} water flux differs from its trace"
            );
            assert!(
                PrismaticPointTraceKind::from_code(kinds[i]).is_some(),
                "{id} carries trace code {}",
                kinds[i]
            );
            assert!(
                speeds[i][0] <= speeds[i][1],
                "{id} wave speeds {:?}",
                speeds[i]
            );
            counts[0] += 1;
        }
    }
    if coverage.is_full() {
        assert_eq!(counts, [66, 3, 1]);
    }
    Ok(())
}

#[test]
fn configured_cpu_public_point_fluxes_enclose_frozen_references() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    let client = CpuClient::new(CpuDevice::new());
    run::<CpuRuntime>(&client, Coverage::Full)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_public_point_fluxes_enclose_frozen_references() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime>(&client, Coverage::Sample)
}
