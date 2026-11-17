use super::super::CrossSectionError;
use super::{
    area_point_query::{
        PrismaticAreaPointBandCoordinates, PrismaticAreaPointBandProperties,
        PrismaticAreaPointRecoveredProperties, PrismaticAreaPointRecovery,
    },
    projection::Projection,
};
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

struct Bands {
    count: usize,
    candidate: Vec<f64>,
    area_above: Bounds,
    area_below: Bounds,
    depth_above: Bounds,
    depth_below: Bounds,
}

pub(super) fn properties<R: Runtime<DType = DType>>(
    query: &PrismaticAreaPointRecovery<R>,
    device: &R::Device,
) -> Result<Vec<PrismaticAreaPointRecoveredProperties>, CrossSectionError> {
    let projection = Projection::read(
        &query.owners,
        query.requested_areas.high(),
        &query.dry,
        &query.fields,
        device,
    )?;
    let count = projection.len();
    let low = read(
        query.requested_areas.low(),
        &[count],
        device,
        "requested area low",
    )?;
    let radius = read(
        query.requested_areas.abs_error(),
        &[count],
        device,
        "requested area radius",
    )?;
    let bands = Bands::read(
        &query.coordinates,
        count,
        query.context.maximum_bands,
        device,
    )?;
    (0..count)
        .map(|index| {
            let row = projection.row(index)?;
            let low = value(&low, index, "requested area low")?;
            let radius = value(&radius, index, "requested area radius")?;
            if !low.is_finite() || radius != 0.0 {
                return Err(invalid(
                    index,
                    "area components require finite low and zero radius",
                ));
            }
            let definition = query
                .context
                .source
                .definitions
                .get(row.owner)
                .ok_or_else(|| invalid(index, "owner exceeds section count"))?;
            Ok(PrismaticAreaPointRecoveredProperties {
                section_id: definition.id.clone(),
                requested_area_high_m2: row.requested,
                requested_area_low_m2: low,
                dry: row.dry,
                depth_m: row.depth,
                forward_area_m2: row.area,
                top_width_m: (!row.dry).then_some(row.top_width),
                hydrostatic_integral_m3: row.integral,
                hydraulic_depth_m: (!row.dry).then_some(row.hydraulic_depth),
                bands: bands.row(index, row.dry)?,
            })
        })
        .collect()
}

impl Bands {
    fn read<R: Runtime<DType = DType>>(
        coordinates: &PrismaticAreaPointBandCoordinates<R>,
        count: usize,
        bands: usize,
        device: &R::Device,
    ) -> Result<Self, CrossSectionError> {
        let shape = &[count, bands];
        let bounds = |value: &Interval<R>, name: &str| -> Result<Bounds, CrossSectionError> {
            Ok(Bounds {
                lower: read(value.lower(), shape, device, name)?,
                upper: read(value.upper(), shape, device, name)?,
            })
        };
        Ok(Self {
            count: bands,
            candidate: read(&coordinates.candidate, shape, device, "band candidate mask")?,
            area_above: bounds(&coordinates.area_above_start, "area above start")?,
            area_below: bounds(&coordinates.area_below_end, "area below end")?,
            depth_above: bounds(&coordinates.depth_above_start, "depth above start")?,
            depth_below: bounds(&coordinates.depth_below_end, "depth below end")?,
        })
    }

    fn row(
        &self,
        row: usize,
        dry: bool,
    ) -> Result<Vec<PrismaticAreaPointBandProperties>, CrossSectionError> {
        let mut result = Vec::new();
        for band_index in 0..self.count {
            let index = row
                .checked_mul(self.count)
                .and_then(|offset| offset.checked_add(band_index))
                .ok_or_else(|| invalid(row, "band index overflows usize"))?;
            let mask = value(&self.candidate, index, "band candidate mask")?;
            if mask != 0.0 && mask != 1.0 {
                return Err(invalid(
                    row,
                    "band candidate mask requires binary candidates",
                ));
            }
            let area_above_start_m2 = self.area_above.row(index)?;
            let area_below_end_m2 = self.area_below.row(index)?;
            let depth_above_start_m = self.depth_above.row(index)?;
            let depth_below_end_m = self.depth_below.row(index)?;
            if mask == 0.0 {
                if [
                    area_above_start_m2,
                    area_below_end_m2,
                    depth_above_start_m,
                    depth_below_end_m,
                ]
                .iter()
                .any(|bounds| *bounds != [0.0, 0.0])
                {
                    return Err(invalid(
                        row,
                        "inactive band coordinates require zero placeholders",
                    ));
                }
            } else {
                if dry && (area_above_start_m2 != [0.0, 0.0] || depth_above_start_m != [0.0, 0.0]) {
                    return Err(invalid(
                        row,
                        "dry endpoint coordinates require zero lower distances",
                    ));
                }
                result.push(PrismaticAreaPointBandProperties {
                    band_index,
                    area_above_start_m2,
                    area_below_end_m2,
                    depth_above_start_m,
                    depth_below_end_m,
                });
            }
        }
        if !dry && result.is_empty() {
            return Err(invalid(row, "wet row requires a compatible band"));
        }
        Ok(result)
    }
}

impl Bounds {
    fn row(&self, index: usize) -> Result<[f64; 2], CrossSectionError> {
        let lower = value(&self.lower, index, "coordinate lower")?;
        let upper = value(&self.upper, index, "coordinate upper")?;
        if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || lower > upper {
            return Err(invalid(
                index,
                "coordinates require finite ordered nonnegative bounds",
            ));
        }
        Ok([lower, upper])
    }
}

fn read<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
    shape: &[usize],
    device: &R::Device,
    name: &str,
) -> Result<Vec<f64>, CrossSectionError> {
    if tensor.shape() != shape || tensor.dtype() != DType::F64 || !tensor.device().is_same(device) {
        return Err(CrossSectionError::numerical(format!(
            "prismatic area-point output {name}: shape={:?}, dtype={:?}, device={}. Rebuild immutable output with shape={shape:?}, F64, device={}",
            tensor.shape(),
            tensor.dtype(),
            tensor.device().id(),
            device.id()
        )));
    }
    Ok(tensor.contiguous()?.try_to_vec::<f64>()?)
}

fn value(values: &[f64], index: usize, name: &str) -> Result<f64, CrossSectionError> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| invalid(index, name))
}

fn invalid(index: usize, reason: &str) -> CrossSectionError {
    CrossSectionError::numerical(format!(
        "prismatic area-point output query={index}: {reason}. Rebuild the complete immutable recovery"
    ))
}
