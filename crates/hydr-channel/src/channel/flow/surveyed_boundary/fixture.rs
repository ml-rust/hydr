//! One-cell surveyed reaches on the frozen three-band profile, and their end-cell interiors.

use super::{
    interior::Interior,
    point::{Evaluation, TraceDepth, TracePoint},
};
use crate::channel::flow::{
    ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReach,
    ChannelReachDefinition, ChannelSurveyedConfig, ChannelTensorBoundary, SurveyedChannelCell,
    geometry::{SurveyedGeometry, TensorGeometry},
    surveyed::{error::checked, work::WorkLedger},
    surveyed_update::tests::{Client, TestResult},
};
use crate::channel::{CrossSectionDefinition, CrossSectionPoint};
use numr::{dtype::DType, runtime::Runtime};

pub(super) const GRAVITY: f64 = 9.81;
/// Cap depth of the three-band profile, in metres.
pub(super) const CAP: f64 = 3.0;
/// Simpson panels that resolve every three-band trace of the frozen references.
pub(super) const PANELS: usize = 512;

/// The `three-bands` survey: knots at depths 1 and 2, cap 3, bed at `bed`.
fn profile(bed: f64) -> CrossSectionDefinition {
    let points = [
        (-4.0, 3.0),
        (-2.5, 2.0),
        (-2.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (2.5, 2.0),
        (4.0, 3.0),
    ];
    CrossSectionDefinition {
        id: "three-bands".into(),
        vertical_datum: "Synthetic SI datum".into(),
        points: points
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m: elevation_m + bed,
            })
            .collect(),
        maximum_stage_m: 3.0 + bed,
        profile_provenance: "Frozen three-band acceptance survey".into(),
    }
}

/// One unit-length cell of area `area` and global discharge `discharge`, bed at `bed`.
pub(super) fn reach<R>(
    client: R::Client,
    area: f64,
    discharge: f64,
    bed: f64,
) -> TestResult<ChannelReach<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    Ok(ChannelReach::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "Synthetic SI datum".into(),
            provenance: "Surveyed trace roots".into(),
            geometry: ChannelGeometry::PrismaticSurveyed {
                profile: profile(bed),
                cells: vec![SurveyedChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    manning_n: 0.0,
                }],
                numerical: ChannelSurveyedConfig::default(),
            },
            lateral_bindings: Vec::new(),
            config: ChannelConfig {
                gravity: GRAVITY,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![area],
            longitudinal_momentum_m4_per_s: vec![discharge],
        },
    )?)
}

/// The surveyed geometry of `reach`.
pub(super) fn geometry<R: Runtime<DType = DType>>(
    reach: &ChannelReach<R>,
) -> TestResult<&SurveyedGeometry<R>> {
    match &reach.context.geometry {
        TensorGeometry::Surveyed(geometry) => Ok(geometry),
        _ => Err("surveyed geometry required".into()),
    }
}

/// The end-cell interior of `reach` at `endpoint`.
pub(super) fn interior<R>(reach: &ChannelReach<R>, endpoint: usize) -> TestResult<Interior<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let context = &reach.context;
    let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
    Ok(context.surveyed_interior(geometry(reach)?, &fields, endpoint)?)
}

/// The tensor policy of `policy`.
pub(super) fn policy<R>(
    reach: &ChannelReach<R>,
    policy: &ChannelBoundary,
) -> TestResult<ChannelTensorBoundary<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    Ok(ChannelTensorBoundary::from_host(
        &reach.context.client,
        policy,
    )?)
}

/// A fresh ledger of `reach` at the fixture panel count.
pub(super) fn ledger<R: Runtime<DType = DType>>(reach: &ChannelReach<R>) -> TestResult<WorkLedger> {
    Ok(WorkLedger::new(&geometry(reach)?.riemann))
}

/// The trace state over the normalized depths `[lower, upper]`.
pub(super) fn state<R>(
    reach: &ChannelReach<R>,
    interior: &Interior<R>,
    evaluation: &Evaluation<'_>,
    lower: f64,
    upper: f64,
) -> TestResult<TracePoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let context = &reach.context;
    let (lower, upper) = (context.tensor(&[lower])?, context.tensor(&[upper])?);
    let t = checked(&context.client, "fixture normalized depth", |a| {
        Ok(a.bounds(&lower, &upper)?)
    })
    .map_err(|error| format!("{error:?}"))?;
    Ok(context.surveyed_trace_point(interior, TraceDepth::Normalized(&t), evaluation)?)
}

/// Host `[lower, upper]` of a one-element interval.
pub(super) fn ends<R: Runtime<DType = DType>>(
    value: &solvr::common::arithmetic::Interval<R>,
) -> TestResult<[f64; 2]> {
    Ok([value.lower().item::<f64>()?, value.upper().item::<f64>()?])
}
