use super::super::wave_curve_types::{ShockDomain, ShockOffset};
use super::*;
use crate::channel::section::{AreaDirection, AreaQuery};

#[derive(Debug, PartialEq)]
struct ValueRow {
    retained: [u64; 3],
    physical: [u64; 2],
    available: u64,
}
#[derive(Debug, PartialEq)]
struct DomainRow {
    source: [u64; 2],
    target: [u64; 2],
    retained_gap: [u64; 3],
    gap: [u64; 2],
    contraction: [u64; 2],
}
#[derive(Debug, PartialEq)]
struct OffsetRow {
    value: ValueRow,
    domain: DomainRow,
}
#[derive(Debug, PartialEq)]
struct EndpointRow {
    curve: ValueRow,
    actual: OffsetRow,
    conditional: OffsetRow,
    equality_limit: ValueRow,
    classification: [u64; 5],
}

fn value_rows<R: Runtime<DType = DType>>(value: &CurveValue<R>) -> Result<Vec<ValueRow>> {
    let retained = components(&value.retained)?;
    let physical = endpoints(&value.physical)?;
    let available = value.available.try_to_vec::<f64>()?;
    Ok(retained
        .into_iter()
        .zip(physical)
        .zip(available)
        .map(|((retained, physical), available)| ValueRow {
            retained,
            physical: physical.map(f64::to_bits),
            available: available.to_bits(),
        })
        .collect())
}

fn domain_rows<R: Runtime<DType = DType>>(value: &ShockDomain<R>) -> Result<Vec<DomainRow>> {
    let source = endpoints(&value.source_area)?;
    let target = endpoints(&value.target_area)?;
    let retained_gap = components(&value.retained_separation)?;
    let gap = endpoints(&value.separation)?;
    let contraction = endpoints(&value.contraction)?;
    Ok(source
        .into_iter()
        .zip(target)
        .zip(retained_gap)
        .zip(gap)
        .zip(contraction)
        .map(
            |((((source, target), retained_gap), gap), contraction)| DomainRow {
                source: source.map(f64::to_bits),
                target: target.map(f64::to_bits),
                retained_gap,
                gap: gap.map(f64::to_bits),
                contraction: contraction.map(f64::to_bits),
            },
        )
        .collect())
}

fn offset_rows<R: Runtime<DType = DType>>(value: &ShockOffset<R>) -> Result<Vec<OffsetRow>> {
    Ok(value_rows(&value.value)?
        .into_iter()
        .zip(domain_rows(&value.domain)?)
        .map(|(value, domain)| OffsetRow { value, domain })
        .collect())
}

fn snapshot<R: Runtime<DType = DType>>(value: &EndpointCurve<R>) -> Result<Vec<EndpointRow>> {
    let curve = value_rows(&value.curve)?;
    let actual = offset_rows(&value.actual_shock)?;
    let conditional = offset_rows(&value.conditional_shock)?;
    let equality_limit = value_rows(&value.equality_limit)?;
    let masks = [
        &value.classification.shock,
        &value.classification.rarefaction,
        &value.classification.equality,
        &value.classification.uncertain,
        &value.classification.dry_source,
    ]
    .into_iter()
    .map(|v| v.try_to_vec::<f64>())
    .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(curve
        .into_iter()
        .zip(actual)
        .zip(conditional)
        .zip(equality_limit)
        .enumerate()
        .map(
            |(row, (((curve, actual), conditional), equality_limit))| EndpointRow {
                curve,
                actual,
                conditional,
                equality_limit,
                classification: std::array::from_fn(|mask| masks[mask][row].to_bits()),
            },
        )
        .collect())
}

#[derive(Debug, PartialEq)]
struct Inputs {
    source: Vec<[u64; 3]>,
    source_owners: Vec<i64>,
    target: Vec<[u64; 3]>,
    scale: Vec<[u64; 3]>,
    parameter: Vec<[u64; 2]>,
    area: Vec<[u64; 2]>,
    above_start: Vec<[u64; 2]>,
    below_end: Vec<[u64; 2]>,
    candidate: Vec<u64>,
    owners: Vec<i64>,
    original_rows: Vec<i64>,
    boundaries: Vec<[u64; 3]>,
}

fn input_snapshot<R>(context: &PotentialContext<'_, R>) -> Result<Inputs>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let target = hydraulic(error::checked::<R, _>(
        context.geometry.client(),
        "mixed query identity snapshot",
        |a| {
            context
                .query
                .retained_endpoint_area(a, context.bands, context.boundaries)
        },
    ))?;
    let bits = |interval: &solvr::common::arithmetic::Interval<R>| -> Result<Vec<[u64; 2]>> {
        Ok(endpoints(interval)?
            .into_iter()
            .map(|v| v.map(f64::to_bits))
            .collect())
    };
    Ok(Inputs {
        source: components(context.source.requested_areas_m2())?,
        source_owners: context.source.owners().try_to_vec::<i64>()?,
        target: components(&target)?,
        scale: components(&context.query.query.scale)?,
        parameter: bits(&context.query.query.parameter)?,
        area: bits(&context.query.area)?,
        above_start: bits(&context.query.above_start)?,
        below_end: bits(&context.query.below_end)?,
        candidate: context
            .query
            .candidate
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect(),
        owners: context.query.query.owners.try_to_vec::<i64>()?,
        original_rows: context.query.query.original_rows.try_to_vec::<i64>()?,
        boundaries: components(context.boundaries.areas())?,
    })
}

fn mixed<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let sections = support::sections::<R>(&client)?;
    let section = &sections[1];
    let geometry = HydraulicGeometry::from_batch(section);
    let owner_storage =
        Tensor::from_slice(&[-1_i64, 0, -1, 0, -1, 0, -1, 0], &[4, 2], client.device())?;
    let owners = owner_storage.narrow(1, 1, 1)?.squeeze(Some(1));
    let original_owners = owner_storage.try_to_vec::<i64>()?;
    let source_area = expansion::<R>(
        &client,
        &[
            [1.0_f64.to_bits(), 0, 0],
            [2.0_f64.to_bits(), 0, 0],
            [1.0_f64.to_bits(), 0, 0],
            [0, 0, 0],
        ],
        true,
    )?;
    let original_area = components(&source_area)?;
    let config = PrismaticRiemannConfig {
        gravity_m_per_s2: 3.0,
        ..PrismaticRiemannConfig::default()
    };
    let source = section.recover_area_points_tensor(&owners, &source_area, &config.recovery)?;
    let parameter_storage = Tensor::from_slice(
        &[f64::NAN, 1.0, f64::NAN, -1.0, f64::NAN, 0.0, f64::NAN, 1.0],
        &[4, 2],
        client.device(),
    )?;
    let parameters = parameter_storage.narrow(1, 1, 1)?.squeeze(Some(1));
    let original_parameters = parameter_storage
        .try_to_vec::<f64>()?
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
        &client,
        "mixed side curve graph",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let scale = a.expansion_point(&Tensor::ones(&[4], a.dtype(), client.device())?)?;
            let parameter = a.point(&parameters)?;
            let query = AreaQuery::source_affine(
                a,
                &bands,
                &source,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &bands)?;
            Ok((bands, boundaries, query))
        },
    ))?;
    let work = WorkLedger::new(&config);
    let context = hydraulic(PotentialContext::new(
        &geometry,
        &bands,
        &source,
        &query,
        &boundaries,
        PotentialEvaluation {
            config: &config,
            panels: 4,
            work: &work,
        },
    ))?;
    let before = input_snapshot(&context)?;
    let batch = hydraulic(wave_curve::endpoint(&context, PhysicalEndpoint::Lower))?;
    assert_eq!(
        batch.classification.shock.try_to_vec::<f64>()?,
        [1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        batch.classification.rarefaction.try_to_vec::<f64>()?,
        [0.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        batch.classification.equality.try_to_vec::<f64>()?,
        [0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        batch.classification.uncertain.try_to_vec::<f64>()?,
        [0.0; 4]
    );
    assert_eq!(
        batch.classification.dry_source.try_to_vec::<f64>()?,
        [0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        batch.curve.available.try_to_vec::<f64>()?,
        [1.0, 1.0, 1.0, 0.0]
    );
    assert_eq!(
        batch.actual_shock.value.available.try_to_vec::<f64>()?,
        [1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        batch
            .conditional_shock
            .value
            .available
            .try_to_vec::<f64>()?,
        [0.0; 4]
    );
    assert_eq!(
        batch.equality_limit.available.try_to_vec::<f64>()?,
        [0.0, 0.0, 1.0, 0.0]
    );
    let batch_rows = snapshot(&batch)?;
    let mut singles = Vec::with_capacity(4);
    assert_eq!(batch_rows.len(), 4);
    for (row, batch_row) in batch_rows.iter().enumerate() {
        let original_row = i64::try_from(row)?;
        let rows = Tensor::from_slice(&[original_row], &[1], client.device())?;
        let (bands, source, query, boundaries) = hydraulic(error::checked::<R, _>(
            &client,
            "singleton side curve selection",
            |a| {
                Ok((
                    context.bands.select(a, &rows)?,
                    context.source.select(a, &rows)?,
                    context.query.select(a, &rows)?,
                    context.boundaries.select(a, &rows)?,
                ))
            },
        ))?;
        assert_eq!(
            query.query.original_rows.try_to_vec::<i64>()?,
            [original_row]
        );
        let singleton = hydraulic(PotentialContext::new(
            &geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: &config,
                panels: 4,
                work: &work,
            },
        ))?;
        let value = hydraulic(wave_curve::endpoint(&singleton, PhysicalEndpoint::Lower))?;
        let rows = snapshot(&value)?;
        assert_eq!(rows.len(), 1);
        assert_eq!(batch_row, &rows[0], "mixed row={row}");
        singles.extend(rows);
    }
    let order = [2_i64, 0, 2, 3, 1];
    let rows = Tensor::from_slice(&order, &[order.len()], client.device())?;
    let (selected_bands, selected_source, selected_query, selected_boundaries) = hydraulic(
        error::checked::<R, _>(&client, "repeated mixed side curve selection", |a| {
            Ok((
                bands.select(a, &rows)?,
                source.select(a, &rows)?,
                query.select(a, &rows)?,
                boundaries.select(a, &rows)?,
            ))
        }),
    )?;
    assert_eq!(
        selected_query.query.original_rows.try_to_vec::<i64>()?,
        order
    );
    let selected = hydraulic(PotentialContext::new(
        &geometry,
        &selected_bands,
        &selected_source,
        &selected_query,
        &selected_boundaries,
        PotentialEvaluation {
            config: &config,
            panels: 4,
            work: &work,
        },
    ))?;
    let selected_before = input_snapshot(&selected)?;
    let result = hydraulic(wave_curve::endpoint(&selected, PhysicalEndpoint::Lower))?;
    for (row, result) in snapshot(&result)?.into_iter().enumerate() {
        assert_eq!(
            result,
            singles[usize::try_from(order[row])?],
            "selected row={row}"
        );
    }
    assert_eq!(selected_before, input_snapshot(&selected)?);
    assert_eq!(before, input_snapshot(&context)?);
    assert_eq!(original_area, components(&source_area)?);
    assert_eq!(original_owners, owner_storage.try_to_vec::<i64>()?);
    assert_eq!(
        original_parameters,
        parameter_storage
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>()
    );
    assert_eq!(work.snapshot(), Default::default());

    let physical_curve = endpoints(&batch.curve.physical)?;
    let physical_offset = endpoints(&batch.actual_shock.value.physical)?;
    let retained_curve =
        super::super::source_tests::support::bounds::<R>(&client, &batch.curve.retained)?;
    let retained_curve = endpoints(&retained_curve)?;
    let retained_offset = super::super::source_tests::support::bounds::<R>(
        &client,
        &batch.actual_shock.value.retained,
    )?;
    let retained_offset = endpoints(&retained_offset)?;
    for (row, id) in [
        (0, "legacy_stationary_compression"),
        (1, "legacy_stationary_expansion"),
    ] {
        let reference = fixtures::CASES
            .iter()
            .find(|case| case.id == id)
            .ok_or("Missing frozen stationary wave reference")?;
        let curve = match reference.expected {
            fixtures::Expected::Shock { curve, offset } => {
                support::check(physical_offset[row], offset, id);
                support::check(retained_offset[row], offset, id);
                curve
            }
            fixtures::Expected::Rare { curve } => curve,
            _ => return Err("Stationary wave reference has an inconsistent branch".into()),
        };
        support::check(physical_curve[row], curve, id);
        support::check(retained_curve[row], curve, id);
    }
    Ok(())
}

#[test]
fn configured_cpu_mixed_side_curves_preserve_selection_and_singleton_certificates() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    mixed::<CpuRuntime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mixed_side_curves_preserve_selection_and_singleton_certificates() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    mixed::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
