use super::contact::Fields;
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

pub(super) struct Projection {
    perimeter: Bounds,
    radius: Bounds,
}

impl Projection {
    pub fn read<R: Runtime<DType = DType>>(
        fields: &Fields<R>,
        count: usize,
        device: &R::Device,
    ) -> Result<Self, CrossSectionError> {
        count
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CrossSectionError::numerical(
                    "prismatic perimeter output bytes overflow usize. Split the query batch",
                )
            })?;
        let read = |tensor: &Tensor<R>, name: &str| {
            if tensor.shape() != [count]
                || tensor.dtype() != DType::F64
                || !tensor.device().is_same(device)
            {
                return Err(CrossSectionError::numerical(format!(
                    "prismatic output {name} shape={:?}, dtype={:?}, device_id={}: rebuild with shape=[{count}], dtype=F64, device_id={}",
                    tensor.shape(),
                    tensor.dtype(),
                    tensor.device().id(),
                    device.id()
                )));
            }
            let values = tensor.contiguous()?.try_to_vec::<f64>()?;
            if values.len() != count {
                return Err(CrossSectionError::numerical(format!(
                    "prismatic output {name} count={}, expected={count}: rebuild the complete query",
                    values.len()
                )));
            }
            Ok(values)
        };
        Ok(Self {
            perimeter: Bounds {
                lower: read(fields.perimeter.lower(), "perimeter lower")?,
                upper: read(fields.perimeter.upper(), "perimeter upper")?,
            },
            radius: Bounds {
                lower: read(fields.radius.lower(), "radius lower")?,
                upper: read(fields.radius.upper(), "radius upper")?,
            },
        })
    }

    pub fn row(
        &self,
        index: usize,
        dry: bool,
    ) -> Result<([f64; 2], Option<[f64; 2]>), CrossSectionError> {
        let perimeter = self.perimeter.row(index, "perimeter")?;
        let radius = self.radius.row(index, "radius")?;
        if dry && (perimeter != [0.0, 0.0] || radius != [0.0, 0.0]) {
            return Err(CrossSectionError::numerical(format!(
                "prismatic dry contact perimeter={perimeter:?}, radius={radius:?}, query={index}: rebuild with zero dry placeholders"
            )));
        }
        if !dry && perimeter[0] <= 0.0 {
            return Err(CrossSectionError::numerical(format!(
                "prismatic wet perimeter={perimeter:?}, query={index}: rebuild with a represented positive perimeter lower bound"
            )));
        }
        Ok((perimeter, (!dry).then_some(radius)))
    }
}

impl Bounds {
    fn row(&self, index: usize, name: &str) -> Result<[f64; 2], CrossSectionError> {
        let value = self.lower.get(index).zip(self.upper.get(index));
        let (&lower, &upper) = value.ok_or_else(|| {
            CrossSectionError::numerical(format!(
                "prismatic output {name} lacks query={index}: rebuild the complete query"
            ))
        })?;
        if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || lower > upper {
            return Err(CrossSectionError::numerical(format!(
                "prismatic output {name}=[{lower},{upper}], query={index}: rebuild with finite ordered nonnegative bounds"
            )));
        }
        Ok([lower, upper])
    }
}
