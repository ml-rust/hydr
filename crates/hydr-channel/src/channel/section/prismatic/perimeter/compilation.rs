use super::super::{
    compilation_bounds::{nonnegative, select, stack},
    context::Context,
    inputs::{BankInputs, Inputs, column},
    query_input::{self, Arithmetic, QueryResult},
    topology::Topology,
};
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

#[derive(Debug)]
pub(super) struct ContactTable<R: Runtime<DType = DType>> {
    pub start: Interval<R>,
    pub thickness: Interval<R>,
    pub left: Interval<R>,
    pub right: Interval<R>,
    pub cap: Interval<R>,
}

pub(super) fn compile<R>(
    context: &Context<R>,
    topology: &Topology,
) -> Result<ContactTable<R>, CrossSectionError>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    query_input::checked(context, "prismatic perimeter compilation", |a| {
        let c = a.client();
        let count = topology.profiles.len();
        let inputs = Inputs::new(c, topology)?;
        let zero_tensor = Tensor::zeros(&[count], DType::F64, c.device())?;
        let one_tensor = Tensor::ones(&[count], DType::F64, c.device())?;
        let zero = a.point(&zero_tensor)?;
        let one = a.point(&one_tensor)?;
        let left_bottom = a.point(&inputs.bottom_left)?;
        let right_bottom = a.point(&inputs.bottom_right)?;
        let bottom = a.sub(&right_bottom, &left_bottom)?;
        let mut perimeter = nonnegative(a, bottom, &zero_tensor)?;
        let mut starts = Vec::with_capacity(topology.maximum_bands);
        let mut thicknesses = Vec::with_capacity(topology.maximum_bands);
        let mut lefts = Vec::with_capacity(topology.maximum_bands);
        let mut rights = Vec::with_capacity(topology.maximum_bands);
        for index in 0..topology.maximum_bands {
            let valid = column(&inputs.valid, index, count)?;
            let bottom = a.point(&column(&inputs.band_bottom, index, count)?)?;
            let top = a.point(&column(&inputs.band_top, index, count)?)?;
            let thickness = a.sub(&top, &bottom)?;
            let thickness = select(a, &valid, &thickness, &zero)?;
            let left = bank_contact(
                a,
                context,
                &inputs.left_bank,
                (index, count),
                &thickness,
                (&zero, &one),
            )?;
            let right = bank_contact(
                a,
                context,
                &inputs.right_bank,
                (index, count),
                &thickness,
                (&zero, &one),
            )?;
            starts.push(select(a, &valid, &perimeter, &zero)?);
            let increment = a.add(&left, &right)?;
            perimeter = a.add(&perimeter, &increment)?;
            thicknesses.push(thickness);
            lefts.push(left);
            rights.push(right);
        }
        Ok(ContactTable {
            start: stack(a, &starts)?,
            thickness: stack(a, &thicknesses)?,
            left: stack(a, &lefts)?,
            right: stack(a, &rights)?,
            cap: perimeter,
        })
    })
}

fn bank_contact<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    bank: &BankInputs<R>,
    position: (usize, usize),
    thickness: &Interval<R>,
    constants: (&Interval<R>, &Interval<R>),
) -> QueryResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (index, count) = position;
    let (zero, one) = constants;
    let c = a.client();
    let vertical = column(&bank.vertical, index, count)?;
    let sloping = c.eq(&vertical, zero.lower())?;
    if !context.source.any(&sloping)? {
        return Ok(a.bounds(thickness.lower(), thickness.upper())?);
    }
    // Mask original coordinates before arithmetic. Vertical banks require no norm or quotient.
    let point = |a: &mut Arithmetic<'_, R>, table: &Tensor<R>, fallback: &Tensor<R>| {
        a.point(&c.where_cond(&sloping, &column(table, index, count)?, fallback)?)
    };
    let left = point(a, &bank.left_station, zero.lower())?;
    let right = point(a, &bank.right_station, zero.lower())?;
    let low = point(a, &bank.low_elevation, zero.lower())?;
    let high = point(a, &bank.high_elevation, one.lower())?;
    let dx = a.sub(&right, &left)?;
    let dz = a.sub(&high, &low)?;
    let dx2 = a.mul(&dx, &dx)?;
    let dz2 = a.mul(&dz, &dz)?;
    let square = a.add(&dx2, &dz2)?;
    let square = nonnegative(a, square, zero.lower())?;
    let length = a.sqrt_nonnegative(&square)?;
    let numerator = select(a, &sloping, thickness, zero)?;
    let fraction = a.div_positive_interval(&numerator, &dz)?;
    // Original bank endpoints span this band, proving the exact fraction lies in [0, 1].
    let fraction = a.bounds(
        &c.maximum(fraction.lower(), zero.lower())?,
        &c.minimum(fraction.upper(), one.upper())?,
    )?;
    let contact = a.mul(&fraction, &length)?;
    let contact = nonnegative(a, contact, zero.lower())?;
    Ok(select(a, &sloping, &contact, thickness)?)
}
