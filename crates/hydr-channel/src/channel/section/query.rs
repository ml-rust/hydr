use super::{
    CrossSectionError, CrossSectionProperties, context::Context,
    evaluation::TensorSectionProperties,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Complete immutable tensor geometry for one ordered stage query.
#[derive(Debug)]
pub struct CrossSectionQuery<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(crate) fields: TensorSectionProperties<R>,
}

impl<R: Runtime<DType = DType>> CrossSectionQuery<R> {
    /// Explicitly copy ordered properties to the host, with absent dry width and derived depths.
    pub fn properties(&self) -> Result<Vec<CrossSectionProperties>, CrossSectionError> {
        let count = self.context.definitions.len();
        let read = |field: &Tensor<R>| -> Result<Vec<f64>, CrossSectionError> {
            if field.shape() != [count] || field.dtype() != DType::F64 {
                return Err(CrossSectionError::numerical(
                    "geometry output shape differs from ordered sections. Rebuild the query",
                ));
            }
            Ok(field.contiguous()?.try_to_vec::<f64>()?)
        };
        let stages = read(&self.fields.stage)?;
        let wet = read(&self.fields.wet)?;
        let area = read(&self.fields.area)?;
        let width = read(&self.fields.width)?;
        let perimeter = read(&self.fields.perimeter)?;
        let radius = read(&self.fields.radius)?;
        let depth = read(&self.fields.hydraulic_depth)?;
        let integral = read(&self.fields.integral)?;
        Ok(self
            .context
            .definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| {
                let represented = wet[index] != 0.0;
                CrossSectionProperties {
                    section_id: definition.id.clone(),
                    query_stage_m: stages[index],
                    dry: !represented,
                    area_m2: area[index],
                    top_width_m: represented.then_some(width[index]),
                    wetted_perimeter_m: perimeter[index],
                    hydraulic_radius_m: represented.then_some(radius[index]),
                    hydraulic_depth_m: represented.then_some(depth[index]),
                    hydrostatic_integral_m3: integral[index],
                }
            })
            .collect())
    }
}
