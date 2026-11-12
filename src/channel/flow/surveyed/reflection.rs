use super::PrismaticRiemannRejection;
use super::equation_ops::{choose, reject_mask, scalar};
use super::error::{Arithmetic, HydraulicResult};
use super::sampling::SampledSolution;
use super::types::CapturedInput;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};

pub(crate) fn tighten<R>(
    a: &mut Arithmetic<'_, R>,
    input: &CapturedInput<R>,
    solution: &mut SampledSolution<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = scalar(a, 0.0)?;
    let equal_area = c.eq(&input.left_area, &input.right_area)?;
    let wet = c.gt(&input.left_area, zero.upper())?;
    let opposite_discharge = c.eq(&input.left_discharge, &c.neg(&input.right_discharge)?)?;
    let reflected = c.mul(&c.mul(&equal_area, &wet)?, &opposite_discharge)?;
    // Equal original wet areas and opposite discharges reflect the same-profile entropy solution, proving Q(0)=0.
    // Completed sample coverage must already enclose this value, including a symmetric vacuum's dry trace.
    for value in [&solution.water_flux, &solution.trace_discharge] {
        let excludes_zero = c.maximum(
            &c.gt(value.lower(), zero.upper())?,
            &c.lt(value.upper(), zero.lower())?,
        )?;
        reject_mask(
            a,
            &c.mul(&reflected, &excludes_zero)?,
            PrismaticRiemannRejection::SamplingUnresolved,
            "Reflected original states require zero discharge, but the completed sample excludes zero. Review sampled candidate enclosures",
        )?;
    }
    solution.water_flux = choose(a, &reflected, &zero, &solution.water_flux)?;
    solution.trace_discharge = choose(a, &reflected, &zero, &solution.trace_discharge)?;
    Ok(())
}
