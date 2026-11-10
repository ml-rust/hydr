use super::{CrossSectionError, context::Context, segments};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ScatterReduceOp, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorSectionProperties<R: Runtime<DType = DType>> {
    pub stage: Tensor<R>,
    pub wet: Tensor<R>,
    pub area: Tensor<R>,
    pub width: Tensor<R>,
    pub perimeter: Tensor<R>,
    pub radius: Tensor<R>,
    pub hydraulic_depth: Tensor<R>,
    pub integral: Tensor<R>,
}

pub(super) fn evaluate<R: Runtime<DType = DType>>(
    context: &Context<R>,
    stages: &Tensor<R>,
) -> Result<TensorSectionProperties<R>, CrossSectionError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    context.stage_input(stages)?;
    let c = &context.client;
    let wet = c.gt(stages, &context.bed)?;
    // Below-bed stages represent no water and must not overflow irrelevant depth subtraction.
    let admitted_stage = c.where_cond(&wet, stages, &context.bed)?;
    let segment_stage = c.index_select(&admitted_stage, 0, &context.owners)?;
    let segments = segments::evaluate(context, &segment_stage)?;
    let sum = |value: &Tensor<R>| {
        c.scatter_reduce(
            &context.zero_sections,
            0,
            &context.owners,
            value,
            ScatterReduceOp::Sum,
            true,
        )
    };
    let area = sum(&segments.area)?;
    let width = sum(&segments.width)?;
    let perimeter = sum(&segments.perimeter)?;
    let integral = sum(&segments.integral)?;
    let section_fields = [
        (&area, "section area"),
        (&width, "section top width"),
        (&perimeter, "section wetted perimeter"),
        (&integral, "section hydrostatic integral"),
    ];
    let mut section_rejection = context.field_rejection(section_fields[0].0, &wet)?;
    for (field, _) in section_fields.iter().skip(1) {
        section_rejection =
            c.maximum(&section_rejection, &context.field_rejection(field, &wet)?)?;
    }
    if context.any(&section_rejection)? {
        for (field, quantity) in section_fields {
            context.section_finite(field, quantity)?;
            context.reject_section(&c.mul(&wet, &c.le(field, &context.zero)?)?, field, quantity)?;
        }
    }
    let radius = c.div(&area, &c.where_cond(&wet, &perimeter, &context.one)?)?;
    let hydraulic_depth = c.div(&area, &c.where_cond(&wet, &width, &context.one)?)?;
    let derived_fields = [
        (&radius, "hydraulic radius"),
        (&hydraulic_depth, "hydraulic depth"),
    ];
    let mut derived_rejection = context.field_rejection(derived_fields[0].0, &wet)?;
    for (field, _) in derived_fields.iter().skip(1) {
        derived_rejection =
            c.maximum(&derived_rejection, &context.field_rejection(field, &wet)?)?;
    }
    if context.any(&derived_rejection)? {
        for (field, quantity) in derived_fields {
            context.section_finite(field, quantity)?;
            context.reject_section(&c.mul(&wet, &c.le(field, &context.zero)?)?, field, quantity)?;
        }
    }
    Ok(TensorSectionProperties {
        stage: stages.clone(),
        wet,
        area,
        width,
        perimeter,
        radius,
        hydraulic_depth,
        integral,
    })
}
