use super::super::super::AreaQueryGeometry;
use super::super::ConditionalBandPanel;
use crate::channel::flow::surveyed::{
    equation_ops::{gravity_depth_root, scalar},
    error::{Arithmetic, HydraulicResult},
};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::Interval;

#[derive(Debug, PartialEq)]
pub(super) struct Snapshot {
    pub fields: Vec<Vec<[f64; 2]>>,
    pub gap: Vec<[f64; 2]>,
    pub active: Vec<f64>,
}

pub(super) fn bounds<R: Runtime<DType = DType>>(
    value: &Interval<R>,
) -> HydraulicResult<Vec<[f64; 2]>> {
    Ok(value
        .lower()
        .try_to_vec::<f64>()?
        .into_iter()
        .zip(value.upper().try_to_vec::<f64>()?)
        .map(|(lower, upper)| [lower, upper])
        .collect())
}

pub(super) fn snapshot<R>(
    a: &mut Arithmetic<'_, R>,
    panel: &ConditionalBandPanel<R>,
) -> HydraulicResult<Snapshot>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let gravity = scalar(a, 1.0)?;
    let sound = gravity_depth_root(a, &gravity, &panel.hydraulic_depth, None)?;
    let fields = [
        &panel.area,
        &panel.depth,
        &panel.top_width,
        &panel.integral,
        &panel.hydraulic_depth,
        &sound,
        panel.coordinates.area_above_start_m2(),
        panel.coordinates.area_below_end_m2(),
        panel.coordinates.depth_above_start_m(),
        panel.coordinates.depth_below_end_m(),
    ]
    .into_iter()
    .map(bounds)
    .collect::<HydraulicResult<Vec<_>>>()?;
    Ok(Snapshot {
        fields,
        gap: bounds(&panel.gap)?,
        active: panel.active.try_to_vec::<f64>()?,
    })
}

pub(super) fn geometry_snapshot<R>(
    a: &mut Arithmetic<'_, R>,
    geometry: &AreaQueryGeometry<R>,
    band: usize,
) -> HydraulicResult<Snapshot>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let gravity = scalar(a, 1.0)?;
    let sound = gravity_depth_root(a, &gravity, &geometry.state.hydraulic_depth, None)?;
    let mut fields = [
        &geometry.state.area,
        &geometry.state.depth,
        &geometry.state.top_width,
        &geometry.state.integral,
        &geometry.state.hydraulic_depth,
        &sound,
    ]
    .into_iter()
    .map(bounds)
    .collect::<HydraulicResult<Vec<_>>>()?;
    let n = geometry.state.area.lower().numel();
    for value in [
        geometry.coordinates.area_above_start_m2(),
        geometry.coordinates.area_below_end_m2(),
        geometry.coordinates.depth_above_start_m(),
        geometry.coordinates.depth_below_end_m(),
    ] {
        let local = a.bounds(
            &value.lower().narrow(1, band, 1)?.squeeze(Some(1)),
            &value.upper().narrow(1, band, 1)?.squeeze(Some(1)),
        )?;
        fields.push(bounds(&local)?);
    }
    Ok(Snapshot {
        fields,
        gap: vec![[0.0; 2]; n],
        active: vec![1.0; n],
    })
}
