pub(super) mod origin;
mod prefix;

#[cfg(test)]
#[path = "oracle_checks.rs"]
mod checks;
#[cfg(test)]
#[path = "oracle_execution.rs"]
mod execution;
#[cfg(test)]
#[path = "fixtures/mod.rs"]
mod fixtures;
#[cfg(test)]
#[path = "oracle_support.rs"]
mod native;

use super::super::super::{boundary_area::BoundaryAreas, hydraulic_geometry::GatheredBands};
use super::super::{SupportedAreaQuery, source_difference::source_difference};
use super::ConditionalBandSegments;
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::PrismaticAreaPointRecovery,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::CertifiedExpansion;

struct SourceOrigin<'a, R: Runtime<DType = DType>> {
    area: &'a CertifiedExpansion<R>,
    delta: &'a CertifiedExpansion<R>,
    target_area: &'a CertifiedExpansion<R>,
    exact_dry: bool,
}

impl<R> ConditionalBandSegments<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn prepare(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        source: &PrismaticAreaPointRecovery<R>,
        endpoint: &SupportedAreaQuery<R>,
        boundaries: &BoundaryAreas<R>,
    ) -> HydraulicResult<Self> {
        let difference = source_difference(a, bands, endpoint, source, boundaries)?;
        let target_area = endpoint.retained_endpoint_area(a, bands, boundaries)?;
        Self::prepare_origin(
            a,
            bands,
            endpoint,
            boundaries,
            SourceOrigin {
                area: source.requested_areas_m2(),
                delta: &difference.lower,
                target_area: &target_area,
                exact_dry: false,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ConditionalBandSegments, checks, execution, fixtures, native};
    use crate::channel::section::prismatic::area_query::{
        endpoints::PhysicalEndpoint, fixtures as queries, pressure::fixtures::CASES as BINDINGS,
        test_support as support,
    };
    use crate::channel::{
        flow::surveyed::error,
        section::{HydraulicGeometry, PrismaticRecoveryConfig},
    };
    use numr::{
        dtype::DType,
        ops::BinaryOps,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };
    use std::cell::Cell;
    use support::{Client, Result, hydraulic};

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let loaded: std::result::Result<fixtures::FixtureSet, fixtures::FixtureError> =
            fixtures::load();
        let evidence = loaded?;
        assert_eq!(evidence.paths.len(), 44);
        assert_eq!(fixtures::FIELD_ORDER.len(), 10);
        let declared_fields: usize = evidence
            .paths
            .iter()
            .flat_map(|path| &path.panels)
            .map(|panel| {
                (panel.points.len() + panel.intervals.len() + panel.endpoint_geometry.len()) * 10
            })
            .sum();
        assert_eq!(declared_fields, 4110);
        let section = support::batch::<R>(&client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let table = hydraulic(error::checked::<R, _>(
            &client,
            "conditional oracle boundary table",
            |a| geometry.boundary_areas(a),
        ))?;
        let mut classes = [0_usize; 4];
        let mut fields = 0;
        for path in &evidence.paths {
            let mut partition: Vec<_> = path
                .panels
                .iter()
                .map(|panel| panel.band_index)
                .chain(path.padded_bands.iter().copied())
                .collect();
            partition.sort_unstable();
            assert_eq!(
                partition,
                (0..fixtures::MAXIMUM_BANDS).collect::<Vec<_>>(),
                "{}",
                path.id
            );
            let binding = BINDINGS
                .get(path.binding_index)
                .ok_or("Missing original pressure binding")?;
            assert_eq!(binding.id, path.binding_id);
            assert_eq!(binding.query_index, path.query_index);
            let original = queries::CASES
                .get(path.query_index)
                .ok_or("Missing original query chart")?;
            assert_eq!(binding.query_id, original.id);
            let physical = match path.physical_endpoint {
                fixtures::PhysicalEndpoint::Lower => PhysicalEndpoint::Lower,
                fixtures::PhysicalEndpoint::Upper => PhysicalEndpoint::Upper,
            };
            let upper = path.physical_endpoint == fixtures::PhysicalEndpoint::Upper;
            assert_eq!(
                path.original_parameter_endpoint,
                usize::from(upper == original.chart.add)
            );
            let input = support::input::<R>(&client, &section, &[original.chart; 3], true, false)?;
            let sources = [[
                binding.source_components[0],
                binding.source_components[1],
                0,
            ]; 3];
            let areas = support::expansion::<R>(&client, &sources, true)?;
            let source = section.recover_area_points_tensor(
                &input.owners,
                &areas,
                &PrismaticRecoveryConfig::default(),
            )?;
            let requests = execution::Requests::new(path);
            let stage = Cell::new("gather strided owners");
            let prepared = error::checked::<R, _>(&client, "conditional oracle preparation", |a| {
                let bands = geometry.gather(a, &input.owners)?;
                stage.set("construct and prove query support");
                let supported = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                stage.set("select query and band rows");
                let row_indices = Tensor::from_slice(&[2_i64, 0, 2], &[3], client.device())?;
                let selected = supported.select(a, &row_indices)?;
                let bands = bands.select(a, &row_indices)?;
                stage.set("construct physical endpoint");
                let endpoint = selected.endpoint(a, &bands, physical)?;
                stage.set("read original rows");
                assert_eq!(endpoint.query.original_rows.try_to_vec::<i64>()?, [2, 0, 2]);
                stage.set("build and select boundary witnesses");
                let boundaries = table.gather(a, &input.owners)?.select(a, &row_indices)?;
                stage.set("prepare conditional segments");
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &endpoint, &boundaries)?;
                stage.set("project and read gap certificates");
                let projected = a.expansion_interval(segments.retained_gap())?;
                checks::gaps(&segments, &native::bounds(&projected)?, path)?;
                stage.set("recover dry endpoint geometry");
                let dry = execution::dry_geometry(
                    a, &geometry, &bands, &source, &endpoint, &segments, path,
                )?;
                Ok((segments, dry))
            })
            .inspect_err(|error| eprintln!("{}: stage={}: {error:?}", path.id, stage.get()));
            let (segments, dry) = hydraulic(prepared)?;
            assert_eq!(
                segments.owners().try_to_vec::<i64>()?,
                vec![original.chart.owner as i64; 3 * fixtures::MAXIMUM_BANDS]
            );
            for (panel, values) in dry {
                for (reference, value) in panel.endpoint_geometry.iter().zip(values) {
                    checks::sample(&value, 0, reference, &path.id);
                    for row in 1..3 {
                        checks::sample(&value, row, reference, &path.id);
                    }
                    fields += 10;
                }
            }
            for panel in &path.panels {
                classes[match panel.kind {
                    fixtures::PanelKind::OrdinaryWet => 0,
                    fixtures::PanelKind::DryFirstBand => 1,
                    fixtures::PanelKind::Zero => 2,
                }] += 1;
            }
            classes[3] += path.padded_bands.len();
            // Prepared segments own their source operands, including originally strided components and owner rows.
            client.copy_into(
                &input.owners,
                &Tensor::zeros(input.owners.shape(), DType::I64, client.device())?,
            )?;
            client.copy_into(
                areas.high(),
                &Tensor::zeros(areas.high().shape(), DType::F64, client.device())?,
            )?;
            client.copy_into(
                source.requested_areas_m2().high(),
                &Tensor::zeros(
                    source.requested_areas_m2().high().shape(),
                    DType::F64,
                    client.device(),
                )?,
            )?;
            if !requests.lanes.is_empty() {
                let value = hydraulic(
                    error::checked::<R, _>(&client, "conditional oracle wet panels", |a| {
                        execution::wet(a, &segments, &requests)
                    })
                    .inspect_err(|error| {
                        eprintln!(
                            "{}: stage=wet panel evaluation/readback: {error:?}",
                            path.id
                        )
                    }),
                )?;
                for (row, reference) in requests.samples.iter().enumerate() {
                    checks::sample(&value, row, reference, &requests.labels[row]);
                    assert_eq!(value.active[row], 1.0, "{}", requests.labels[row]);
                }
                fields += requests.samples.len() / 3 * 10;
            }
        }
        assert_eq!(classes, [35, 13, 38, 46]);
        assert_eq!(fields, declared_fields);
        Ok(())
    }

    #[test]
    fn cpu_conditional_band_oracle_encloses_every_original_panel_field() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        matrix::<CpuRuntime>(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_conditional_band_oracle_encloses_every_original_panel_field() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
