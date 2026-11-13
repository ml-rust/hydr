use super::super::super::{AreaAnchor, AreaDirection, AreaQuery, SupportedAreaQuery};
use super::super::ConditionalBandSegments;
use super::Client;
use super::{
    fixtures::{self, ClipAuthority, Panel, PanelKind, Path, Sample},
    native::{self, Snapshot},
};
use crate::channel::{
    flow::surveyed::error::{
        Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection,
    },
    section::{GatheredBands, HydraulicGeometry, PrismaticAreaPointRecovery},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct Requests<'a> {
    pub lanes: Vec<i64>,
    pub parameters: Vec<[f64; 2]>,
    pub samples: Vec<&'a Sample>,
    pub labels: Vec<String>,
}

impl<'a> Requests<'a> {
    pub fn new(path: &'a Path) -> Self {
        let mut requests = Self {
            lanes: Vec::new(),
            parameters: Vec::new(),
            samples: Vec::new(),
            labels: Vec::new(),
        };
        for panel in &path.panels {
            if panel.kind != PanelKind::OrdinaryWet {
                continue;
            }
            assert_eq!(panel.points.len(), fixtures::PARAMETERS.len());
            assert_eq!(panel.intervals.len(), fixtures::PARAMETER_INTERVALS.len());
            for (sample, parameter) in panel
                .points
                .iter()
                .zip(fixtures::PARAMETERS.iter().map(|&p| [p, p]))
                .chain(
                    panel
                        .intervals
                        .iter()
                        .zip(fixtures::PARAMETER_INTERVALS.iter().copied()),
                )
            {
                for repeated in [2_usize, 0, 1] {
                    requests
                        .lanes
                        .push((repeated * fixtures::MAXIMUM_BANDS + panel.band_index) as i64);
                    requests.parameters.push(parameter.map(f64::from_bits));
                    requests.samples.push(sample);
                    requests.labels.push(format!(
                        "{}, band={}, s={:?}, selected_row={repeated}",
                        path.id, panel.band_index, parameter
                    ));
                }
            }
        }
        requests
    }
}

pub(super) fn wet<R>(
    a: &mut Arithmetic<'_, R>,
    segments: &ConditionalBandSegments<R>,
    requests: &Requests<'_>,
) -> HydraulicResult<Snapshot>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let c = a.client();
    let n = requests.lanes.len();
    let lanes = Tensor::from_slice(&requests.lanes, &[n], c.device())?;
    let lower = Tensor::from_slice(
        &requests.parameters.iter().map(|p| p[0]).collect::<Vec<_>>(),
        &[n],
        c.device(),
    )?;
    let upper = Tensor::from_slice(
        &requests.parameters.iter().map(|p| p[1]).collect::<Vec<_>>(),
        &[n],
        c.device(),
    )?;
    let s = a.bounds(&lower, &upper)?;
    let panel = segments
        .panel(a, &lanes, &s)
        .inspect_err(|error| eprintln!("conditional oracle wet geometry: {error:?}"))?;
    native::snapshot(a, &panel)
        .inspect_err(|error| eprintln!("conditional oracle wet field readback: {error:?}"))
}

type DryGeometry<'a> = Vec<(&'a Panel, Vec<Snapshot>)>;

pub(super) fn dry_geometry<'a, R>(
    a: &mut Arithmetic<'_, R>,
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    source: &PrismaticAreaPointRecovery<R>,
    endpoint: &SupportedAreaQuery<R>,
    segments: &ConditionalBandSegments<R>,
    path: &'a Path,
) -> HydraulicResult<DryGeometry<'a>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let c = a.client();
    let n = bands.section_owner.numel();
    let mut result = Vec::new();
    for panel in &path.panels {
        if panel.kind != PanelKind::DryFirstBand {
            continue;
        }
        assert_eq!(panel.band_index, 0);
        assert_eq!(panel.endpoint_geometry.len(), 2);
        let lane = Tensor::from_slice(&[0_i64], &[1], c.device())?;
        let zero = Tensor::zeros(&[1], DType::F64, c.device())?;
        let one = Tensor::ones(&[1], DType::F64, c.device())?;
        let whole = a.bounds(&zero, &one)?;
        assert!(
            matches!(segments.panel(a, &lane, &whole), Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::QuadratureUnresolved, query: Some(2), ref context, .. }) if context.contains("bottom regularization")),
            "{}",
            path.id
        );
        let zero = Tensor::zeros(&[n], DType::F64, c.device())?;
        let scale = a.expansion_point(&zero)?;
        let parameter = a.point(&zero)?;
        let mut values = Vec::new();
        for (source_endpoint, authority) in [(true, panel.source_clip), (false, panel.target_clip)]
        {
            let query = match authority {
                ClipAuthority::BandStart | ClipAuthority::BandEnd => {
                    let index =
                        Tensor::from_slice(&vec![panel.band_index as i64; n], &[n], c.device())?;
                    let anchor = if authority == ClipAuthority::BandStart {
                        AreaAnchor::BandStart(index)
                    } else {
                        AreaAnchor::BandEnd(index)
                    };
                    AreaQuery::band_affine(
                        a,
                        bands,
                        anchor,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, bands)?
                }
                ClipAuthority::Interior if source_endpoint => AreaQuery::source_affine(
                    a,
                    bands,
                    source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, bands)?,
                ClipAuthority::Interior => {
                    let rows = Tensor::from_slice(
                        &(0..n).map(|i| i as i64).collect::<Vec<_>>(),
                        &[n],
                        c.device(),
                    )?;
                    endpoint.select(a, &rows)?
                }
            };
            let recovered = geometry.recover_query(a, bands, &query)?;
            values.push(native::geometry_snapshot(a, &recovered, panel.band_index)?);
        }
        result.push((panel, values));
    }
    Ok(result)
}
