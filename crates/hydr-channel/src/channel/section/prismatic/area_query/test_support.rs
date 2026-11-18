use super::fixtures::{AnchorKind, Case, Chart, Extrema, PROFILES};
use super::{AreaAnchor, AreaDirection, AreaQuery, AreaQueryGeometry};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult};
use crate::channel::section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, GatheredBands,
    PrismaticAreaPointRecovery, PrismaticRecoveryConfig, PrismaticSectionBatch,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval, with_interval_arithmetic};

pub(in crate::channel::section::prismatic) type Result<T = ()> =
    std::result::Result<T, Box<dyn std::error::Error>>;
pub(in crate::channel::section::prismatic) trait Client<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + HypotOps<R>
    + MidpointOps<R>
    + RuntimeClient<R>
{
}
impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + RuntimeClient<R>,
{
}

pub(in crate::channel::section::prismatic) fn hydraulic<T>(
    result: HydraulicResult<T>,
) -> Result<T> {
    result.map_err(|error| std::io::Error::other(format!("{error:?}")).into())
}

pub(in crate::channel::section::prismatic) fn batch<R>(
    client: &R::Client,
) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definitions = PROFILES
        .iter()
        .map(|profile| CrossSectionDefinition {
            id: profile.id.into(),
            vertical_datum: "Frozen area-chart datum".into(),
            points: profile
                .coordinates
                .iter()
                .map(|point| CrossSectionPoint {
                    station_m: f64::from_bits(point[0]),
                    elevation_m: f64::from_bits(point[1]),
                })
                .collect(),
            maximum_stage_m: f64::from_bits(profile.cap_stage),
            profile_provenance: "Independent original-coordinate chart extrema".into(),
        })
        .collect();
    Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
}

fn tensor<R>(client: &R::Client, values: &[f64], strided: bool) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    if strided && !values.is_empty() {
        let storage: Vec<_> = values.iter().flat_map(|&value| [f64::NAN, value]).collect();
        let view = Tensor::from_slice(&storage, &[values.len(), 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(view.offset() > 0);
        if values.len() > 1 {
            assert!(!view.is_contiguous());
        }
        Ok(view)
    } else {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            client.device(),
        )?)
    }
}

pub(in crate::channel::section::prismatic) fn expansion<R>(
    client: &R::Client,
    points: &[[u64; 3]],
    strided: bool,
) -> Result<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let components = (0..3)
        .map(|field| {
            tensor::<R>(
                client,
                &points
                    .iter()
                    .map(|point| f64::from_bits(point[field]))
                    .collect::<Vec<_>>(),
                strided,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(with_interval_arithmetic::<R, _, _, _>(
        client,
        DType::F64,
        "private chart scale",
        |a| a.expansion_bounds(&components[0], &components[1], &components[2]),
    )?)
}

pub(in crate::channel::section::prismatic) struct Input<R: Runtime<DType = DType>> {
    pub owners: Tensor<R>,
    kind: AnchorKind,
    direction: AreaDirection,
    band: Tensor<R>,
    source: Option<PrismaticAreaPointRecovery<R>>,
    scale: CertifiedExpansion<R>,
    parameter: Interval<R>,
}

pub(in crate::channel::section::prismatic) fn input<R>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    charts: &[Chart],
    strided: bool,
    malformed_parameter: bool,
) -> Result<Input<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let first = charts.first().ok_or("Private chart batch is empty")?;
    assert!(
        charts
            .iter()
            .all(|chart| chart.kind == first.kind && chart.add == first.add)
    );
    let count = charts.len();
    let owners: Vec<_> = charts
        .iter()
        .map(|chart| i64::try_from(chart.owner))
        .collect::<std::result::Result<_, _>>()?;
    let owners = if strided {
        let storage: Vec<_> = owners.iter().flat_map(|&owner| [i64::MAX, owner]).collect();
        let view = Tensor::from_slice(&storage, &[count, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(view.offset() > 0);
        if count > 1 {
            assert!(!view.is_contiguous());
        }
        view
    } else {
        Tensor::from_slice(&owners, &[count], client.device())?
    };
    let scale = expansion::<R>(
        client,
        &charts.iter().map(|chart| chart.scale).collect::<Vec<_>>(),
        strided,
    )?;
    let upper = tensor::<R>(
        client,
        &charts
            .iter()
            .map(|chart| f64::from_bits(chart.parameter[1]))
            .collect::<Vec<_>>(),
        strided,
    )?;
    let lower = tensor::<R>(
        client,
        &charts
            .iter()
            .map(|chart| f64::from_bits(chart.parameter[if malformed_parameter { 1 } else { 0 }]))
            .collect::<Vec<_>>(),
        strided,
    )?;
    let parameter = with_interval_arithmetic::<R, _, _, _>(
        client,
        DType::F64,
        "private chart parameter",
        |a| a.bounds(&lower, &upper),
    )?;
    if malformed_parameter {
        let invalid = tensor::<R>(
            client,
            &charts
                .iter()
                .map(|chart| f64::from_bits(chart.parameter[0]))
                .collect::<Vec<_>>(),
            false,
        )?;
        client.copy_into(parameter.lower(), &invalid)?;
    }
    let source = if first.kind == AnchorKind::Source {
        let area = expansion::<R>(
            client,
            &charts.iter().map(|chart| chart.source).collect::<Vec<_>>(),
            strided,
        )?;
        Some(section.recover_area_points_tensor(
            &owners,
            &area,
            &PrismaticRecoveryConfig::default(),
        )?)
    } else {
        None
    };
    Ok(Input {
        owners,
        kind: first.kind,
        direction: if first.add {
            AreaDirection::Add
        } else {
            AreaDirection::Subtract
        },
        band: Tensor::from_slice(
            &charts.iter().map(|chart| chart.band).collect::<Vec<_>>(),
            &[count],
            client.device(),
        )?,
        source,
        scale,
        parameter,
    })
}

pub(in crate::channel::section::prismatic) fn query<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    input: &Input<R>,
) -> HydraulicResult<AreaQuery<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    if input.kind == AnchorKind::Source {
        let source = input
            .source
            .as_ref()
            .ok_or_else(|| HydraulicError::Numerical {
                context: "Source chart has no recovered anchor".into(),
            })?;
        return AreaQuery::source_affine(
            a,
            bands,
            source,
            input.direction,
            &input.scale,
            &input.parameter,
        );
    }
    if input.kind == AnchorKind::Cap && matches!(input.direction, AreaDirection::Subtract) {
        return AreaQuery::cap_deficit(a, bands, &input.scale, &input.parameter);
    }
    let anchor = match input.kind {
        AnchorKind::Zero => AreaAnchor::Zero,
        AnchorKind::Cap => AreaAnchor::Cap,
        AnchorKind::BandStart => AreaAnchor::BandStart(input.band.clone()),
        AnchorKind::BandEnd => AreaAnchor::BandEnd(input.band.clone()),
        AnchorKind::Source => {
            return Err(HydraulicError::Numerical {
                context: "Source chart lacks its recovered anchor".into(),
            });
        }
    };
    AreaQuery::band_affine(
        a,
        bands,
        anchor,
        input.direction,
        &input.scale,
        &input.parameter,
    )
}

#[derive(Debug, PartialEq)]
pub(super) struct Snapshot {
    fields: Vec<Vec<[f64; 2]>>,
    local: Vec<Vec<[f64; 2]>>,
    mask: Vec<f64>,
    wet: Vec<f64>,
    dry: Vec<f64>,
    columns: usize,
}

fn endpoints<R: Runtime<DType = DType>>(
    value: &Interval<R>,
    shape: &[usize],
) -> Result<Vec<[f64; 2]>> {
    for endpoint in [value.lower(), value.upper()] {
        assert_eq!(endpoint.shape(), shape);
        assert_eq!(endpoint.dtype(), DType::F64);
    }
    Ok(value
        .lower()
        .try_to_vec::<f64>()?
        .into_iter()
        .zip(value.upper().try_to_vec::<f64>()?)
        .map(|(lower, upper)| [lower, upper])
        .collect())
}

pub(super) fn snapshot<R: Runtime<DType = DType>>(
    value: &AreaQueryGeometry<R>,
) -> Result<Snapshot> {
    let state = &value.state;
    let coordinates = &value.coordinates;
    let count = state.depth.lower().numel();
    assert_eq!(coordinates.candidate_mask().ndim(), 2);
    let columns = coordinates
        .candidate_mask()
        .shape()
        .get(1)
        .copied()
        .ok_or("Missing private chart band dimension")?;
    assert_eq!(coordinates.candidate_mask().shape(), &[count, columns]);
    assert_eq!(coordinates.candidate_mask().dtype(), DType::F64);
    let fields = [
        &state.area,
        &state.forward_area,
        &state.depth,
        &state.top_width,
        &state.integral,
        &state.hydraulic_depth,
    ]
    .into_iter()
    .map(|field| endpoints(field, &[count]))
    .collect::<Result<Vec<_>>>()?;
    let local = [
        coordinates.area_above_start_m2(),
        coordinates.area_below_end_m2(),
        coordinates.depth_above_start_m(),
        coordinates.depth_below_end_m(),
    ]
    .into_iter()
    .map(|field| endpoints(field, &[count, columns]))
    .collect::<Result<Vec<_>>>()?;
    Ok(Snapshot {
        fields,
        local,
        mask: coordinates.candidate_mask().try_to_vec::<f64>()?,
        wet: state.wet_possible.try_to_vec::<f64>()?,
        dry: state.dry_possible.try_to_vec::<f64>()?,
        columns,
    })
}

fn range(actual: [f64; 2], truth: Extrema, label: &str) {
    let minimum = truth.minimum.map(f64::from_bits);
    let maximum = truth.maximum.map(f64::from_bits);
    assert!(
        actual[0].is_finite()
            && actual[1].is_finite()
            && 0.0 <= actual[0]
            && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= minimum[0] && maximum[1] <= actual[1],
        "{label}: {actual:?} excludes minimum={minimum:?}, maximum={maximum:?}"
    );
    let extra = [minimum[1] - actual[0], actual[1] - maximum[0]];
    let allowance = truth.allowances.map(f64::from_bits);
    for endpoint in 0..2 {
        assert!(
            extra[endpoint] <= allowance[endpoint],
            "{label}: endpoint={endpoint}, excess={}, allowance={}, conditioned_scale={:?}",
            extra[endpoint],
            allowance[endpoint],
            truth.conditioned_scale.map(f64::from_bits)
        );
    }
    if truth.zero {
        assert_eq!(actual, [0.0, 0.0], "{label}");
    }
    if truth.positive && minimum[0] > 0.0 {
        assert!(actual[0] > 0.0, "{label}: complete positive range");
    }
}

pub(super) fn check(snapshot: &Snapshot, cases: &[&Case]) {
    assert_eq!(snapshot.wet.len(), cases.len());
    assert_eq!(snapshot.dry.len(), cases.len());
    for (row, case) in cases.iter().enumerate() {
        assert_eq!(
            snapshot.wet[row],
            if case.wet { 1.0 } else { 0.0 },
            "{}",
            case.id
        );
        assert_eq!(
            snapshot.dry[row],
            if case.dry { 1.0 } else { 0.0 },
            "{}",
            case.id
        );
        for (field, (reference, name)) in [
            (0, "area"),
            (0, "forward_area"),
            (1, "depth"),
            (2, "width"),
            (3, "integral"),
            (4, "hydraulic_depth"),
        ]
        .into_iter()
        .enumerate()
        {
            range(
                snapshot.fields[field][row],
                case.fields[reference],
                &format!("{}, {name}", case.id),
            );
        }
        for band in case.bands {
            let index = row * snapshot.columns + band.index;
            assert_eq!(
                snapshot.mask[index], 1.0,
                "{}: physical band={}",
                case.id, band.index
            );
            for (field, name) in [
                "area_above_start",
                "area_below_end",
                "depth_above_start",
                "depth_below_end",
            ]
            .into_iter()
            .enumerate()
            {
                range(
                    snapshot.local[field][index],
                    band.local[field],
                    &format!("{}, band={}, {name}", case.id, band.index),
                );
            }
        }
        for column in 0..snapshot.columns {
            let index = row * snapshot.columns + column;
            assert!(snapshot.mask[index] == 0.0 || snapshot.mask[index] == 1.0);
            if snapshot.mask[index] == 0.0 {
                for field in &snapshot.local {
                    assert_eq!(
                        field[index],
                        [0.0, 0.0],
                        "{}: inactive band={column}",
                        case.id
                    );
                }
            }
        }
    }
}
