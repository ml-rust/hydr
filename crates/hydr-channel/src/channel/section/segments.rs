use super::{CrossSectionError, context::Context};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct SegmentProperties<R: Runtime<DType = DType>> {
    pub area: Tensor<R>,
    pub width: Tensor<R>,
    pub perimeter: Tensor<R>,
    pub integral: Tensor<R>,
}

pub(super) fn evaluate<R: Runtime<DType = DType>>(
    context: &Context<R>,
    stage: &Tensor<R>,
) -> Result<SegmentProperties<R>, CrossSectionError>
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
    let c = &context.client;
    let zero = &context.zero;
    let one = &context.one;
    let wet = c.gt(stage, &context.low)?;
    let full = c.mul(&wet, &c.ge(stage, &context.high)?)?;
    let partial = c.sub(&wet, &full)?;
    let left_depth = c.maximum(&c.sub(stage, &context.left_elevation)?, zero)?;
    let right_depth = c.maximum(&c.sub(stage, &context.right_elevation)?, zero)?;
    let maximum_depth = c.maximum(&left_depth, &right_depth)?;
    let partial_depth = c.where_cond(&partial, &maximum_depth, zero)?;
    let partial_span = c.where_cond(&partial, &context.span, one)?;
    let partial_fraction = c.div(&partial_depth, &partial_span)?;
    let fraction = c.where_cond(&full, one, &partial_fraction)?;
    let above_one = c.gt(&fraction, one)?;
    let fraction_rejection = c.maximum(&context.field_rejection(&fraction, &wet)?, &above_one)?;
    if context.any(&fraction_rejection)? {
        context.segment_finite(&fraction, "wetted segment fraction")?;
        context.reject_segment(
            &c.mul(&wet, &c.le(&fraction, zero)?)?,
            &fraction,
            "positive wetted segment fraction",
        )?;
        context.reject_segment(&above_one, &fraction, "wetted segment fraction above one")?;
    }
    let width = c.mul(&context.width, &fraction)?;
    let perimeter = c.mul(&context.length, &fraction)?;
    let area = c.mul(&width, &c.midpoint(&left_depth, &right_depth)?)?;
    let depth_denominator = c.where_cond(&wet, &maximum_depth, one)?;
    let q = c.div(&c.minimum(&left_depth, &right_depth)?, &depth_denominator)?;
    let one_plus_q = c.add_scalar(&q, 1.0)?;
    let coefficient = c.div(
        &c.add(&one_plus_q, &c.mul(&q, &q)?)?,
        &c.mul_scalar(&one_plus_q, 3.0)?,
    )?;
    let integral = c.mul(&area, &c.mul(&maximum_depth, &coefficient)?)?;
    let area_support = c.mul(&wet, &c.gt(&context.width, zero)?)?;
    let fields = [
        (&width, "wet horizontal width", &area_support),
        (&perimeter, "wetted perimeter", &wet),
        (&area, "wetted area", &area_support),
        (&integral, "hydrostatic pressure integral", &area_support),
    ];
    let mut field_rejection = context.field_rejection(fields[0].0, fields[0].2)?;
    for (field, _, support) in fields.iter().skip(1) {
        field_rejection = c.maximum(&field_rejection, &context.field_rejection(field, support)?)?;
    }
    if context.any(&field_rejection)? {
        for (field, quantity, support) in fields {
            context.segment_finite(field, quantity)?;
            context.reject_segment(&c.mul(support, &c.le(field, zero)?)?, field, quantity)?;
        }
    }
    Ok(SegmentProperties {
        area,
        width,
        perimeter,
        integral,
    })
}
