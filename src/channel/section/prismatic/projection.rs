use super::super::CrossSectionError;
use super::query::Fields;
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

pub(super) struct Projection {
    owners: Vec<i64>,
    requested: Vec<f64>,
    dry: Vec<f64>,
    depth: Bounds,
    area: Bounds,
    top_width: Bounds,
    integral: Bounds,
    hydraulic_depth: Bounds,
}

pub(super) struct Row {
    pub(super) owner: usize,
    pub(super) requested: f64,
    pub(super) dry: bool,
    pub(super) depth: [f64; 2],
    pub(super) area: [f64; 2],
    pub(super) top_width: [f64; 2],
    pub(super) integral: [f64; 2],
    pub(super) hydraulic_depth: [f64; 2],
}

impl Projection {
    pub(super) fn read<R: Runtime<DType = DType>>(
        owners: &Tensor<R>,
        requested: &Tensor<R>,
        dry: &Tensor<R>,
        fields: &Fields<R>,
        device: &R::Device,
    ) -> Result<Self, CrossSectionError> {
        let count = owners.numel();
        metadata(owners, count, DType::I64, device, "owners")?;
        let owners = owners.contiguous()?.try_to_vec::<i64>()?;
        let read = |value: &Tensor<R>, name: &str| -> Result<Vec<f64>, CrossSectionError> {
            metadata(value, count, DType::F64, device, name)?;
            Ok(value.contiguous()?.try_to_vec::<f64>()?)
        };
        let bounds = |lower: &Tensor<R>, upper: &Tensor<R>, name: &str| {
            Ok::<_, CrossSectionError>(Bounds {
                lower: read(lower, name)?,
                upper: read(upper, name)?,
            })
        };
        Ok(Self {
            owners,
            requested: read(requested, "requested operand")?,
            dry: read(dry, "dry mask")?,
            depth: bounds(fields.depth.lower(), fields.depth.upper(), "depth bounds")?,
            area: bounds(fields.area.lower(), fields.area.upper(), "area bounds")?,
            top_width: bounds(
                fields.top_width.lower(),
                fields.top_width.upper(),
                "width bounds",
            )?,
            integral: bounds(
                fields.integral.lower(),
                fields.integral.upper(),
                "integral bounds",
            )?,
            hydraulic_depth: bounds(
                fields.hydraulic_depth.lower(),
                fields.hydraulic_depth.upper(),
                "hydraulic depth bounds",
            )?,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.owners.len()
    }

    pub(super) fn row(&self, index: usize) -> Result<Row, CrossSectionError> {
        let owner = self
            .owners
            .get(index)
            .copied()
            .ok_or_else(|| missing("owners", index))?;
        let owner = usize::try_from(owner).map_err(|_| {
            CrossSectionError::numerical(format!(
                "prismatic output owner={owner}, query={index}: rebuild the query with nonnegative section owners"
            ))
        })?;
        let requested = value(&self.requested, "requested operand", index)?;
        if !requested.is_finite() || requested < 0.0 {
            return Err(CrossSectionError::numerical(format!(
                "prismatic output requested operand={requested}, query={index}: rebuild the query with finite nonnegative operands"
            )));
        }
        let dry = value(&self.dry, "dry mask", index)?;
        if dry != 0.0 && dry != 1.0 {
            return Err(CrossSectionError::numerical(format!(
                "prismatic output dry mask={dry}, query={index}: rebuild the query with binary dry masks"
            )));
        }
        Ok(Row {
            owner,
            requested,
            dry: dry == 1.0,
            depth: self.depth.row("depth bounds", index)?,
            area: self.area.row("area bounds", index)?,
            top_width: self.top_width.row("width bounds", index)?,
            integral: self.integral.row("integral bounds", index)?,
            hydraulic_depth: self.hydraulic_depth.row("hydraulic depth bounds", index)?,
        })
    }
}

impl Bounds {
    fn row(&self, name: &str, index: usize) -> Result<[f64; 2], CrossSectionError> {
        let lower = value(&self.lower, name, index)?;
        let upper = value(&self.upper, name, index)?;
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(CrossSectionError::numerical(format!(
                "prismatic output {name}=[{lower},{upper}], query={index}: rebuild the query with finite ordered bounds"
            )));
        }
        Ok([lower, upper])
    }
}

fn value(values: &[f64], name: &str, index: usize) -> Result<f64, CrossSectionError> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| missing(name, index))
}

fn missing(name: &str, index: usize) -> CrossSectionError {
    CrossSectionError::numerical(format!(
        "prismatic output {name} lacks query={index}: rebuild the complete query"
    ))
}

fn metadata<R: Runtime<DType = DType>>(
    value: &Tensor<R>,
    count: usize,
    dtype: DType,
    device: &R::Device,
    name: &str,
) -> Result<(), CrossSectionError> {
    if value.shape() != [count] || value.dtype() != dtype || !value.device().is_same(device) {
        return Err(CrossSectionError::numerical(format!(
            "prismatic output {name} shape={:?}, dtype={:?}, device_id={}: rebuild the query with shape=[{count}], dtype={dtype:?}, device_id={}",
            value.shape(),
            value.dtype(),
            value.device().id(),
            device.id()
        )));
    }
    Ok(())
}
