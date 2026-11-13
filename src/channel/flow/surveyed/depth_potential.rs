use super::PrismaticRiemannRejection;
use super::band_lanes::IntegrationLanes;
use super::equation_ops::{
    choose, gravity_depth_root, reject_mask, scalar, scale, select_interval,
};
use super::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

#[cfg(test)]
#[path = "depth_potential_wet.rs"]
mod wet;
#[cfg(test)]
pub(crate) use wet::WetDepthPotential;

pub(super) struct DepthPotential<'input, R: Runtime<DType = DType>> {
    lanes: &'input IntegrationLanes<R>,
    owner: &'input Tensor<R>,
    parameter: &'input Interval<R>,
}

impl<'input, R> DepthPotential<'input, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn from_lanes(
        lanes: &'input IntegrationLanes<R>,
        owner: &'input Tensor<R>,
        parameter: &'input Interval<R>,
    ) -> Self {
        Self {
            lanes,
            owner,
            parameter,
        }
    }

    pub(super) fn value(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Interval<R>> {
        let (n, d, prefactor) = coefficients(a, self.lanes, self.owner, self.parameter)?;
        value_from_coefficients(a, &n, &d, &prefactor)
    }

    pub(super) fn fourth_derivative(
        &self,
        a: &mut Arithmetic<'_, R>,
    ) -> HydraulicResult<Interval<R>> {
        let (n, d, prefactor) = coefficients(a, self.lanes, self.owner, self.parameter)?;
        let zero = scalar(a, 0.0)?;
        let fourth = fourth_from_coefficients(a, &n, &d, &prefactor, &zero)?;
        let active = a.client().index_select(&self.lanes.active, 0, self.owner)?;
        choose(a, &active, &fourth, &zero)
    }
}

fn value_from_coefficients<R>(
    a: &mut Arithmetic<'_, R>,
    n: &[Interval<R>; 5],
    d: &[Interval<R>; 5],
    prefactor: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = scalar(a, 0.0)?;
    reject_mask(
        a,
        &a.client().le(d[0].lower(), zero.lower())?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Surveyed integrand denominator lacks a positive lower bound. Refine geometry or inspect shallow conditioning",
    )?;
    let ratio = a.div_positive_interval(&n[0], &d[0])?;
    let root = a.sqrt_nonnegative(&ratio)?;
    Ok(a.mul(prefactor, &root)?)
}

fn fourth_from_coefficients<R>(
    a: &mut Arithmetic<'_, R>,
    n: &[Interval<R>; 5],
    d: &[Interval<R>; 5],
    prefactor: &Interval<R>,
    zero: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    reject_mask(
        a,
        &a.client().le(d[0].lower(), zero.lower())?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Surveyed derivative denominator lacks a positive lower bound. Refine geometry or inspect shallow conditioning",
    )?;
    // At every x, N = D*Q. Induction encloses Q^(k)(x)/k! throughout the complete panel.
    let mut ratio = Vec::with_capacity(5);
    for k in 0..=4 {
        let mut numerator = a.bounds(n[k].lower(), n[k].upper())?;
        for j in 1..=k {
            let product = a.mul(&d[j], &ratio[k - j])?;
            numerator = a.sub(&numerator, &product)?;
        }
        ratio.push(a.div_positive_interval(&numerator, &d[0])?);
    }
    let y0 = a.sqrt_nonnegative(&ratio[0])?;
    reject_mask(
        a,
        &a.client().le(y0.lower(), zero.lower())?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Surveyed derivative square root lacks a positive lower bound. Refine geometry or inspect shallow conditioning",
    )?;
    let denominator = scale(a, &y0, 2.0)?;
    let mut root = Vec::with_capacity(5);
    root.push(y0);
    // Q = Y*Y gives 2*Y0*Yk = Qk - sum(Yj*Y(k-j)), pointwise for each derivative order.
    for k in 1..=4 {
        let mut numerator = a.bounds(ratio[k].lower(), ratio[k].upper())?;
        for j in 1..k {
            let product = a.mul(&root[j], &root[k - j])?;
            numerator = a.sub(&numerator, &product)?;
        }
        root.push(a.div_positive_interval(&numerator, &denominator)?);
    }
    let coefficient = a.mul(prefactor, &root[4])?;
    let fourth = scale(a, &coefficient, 24.0)?;
    Ok(fourth)
}

type Coefficients<R> = ([Interval<R>; 5], [Interval<R>; 5], Interval<R>);

fn coefficients<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &IntegrationLanes<R>,
    owner: &Tensor<R>,
    x: &Interval<R>,
) -> HydraulicResult<Coefficients<R>>
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
    let bottom = c.index_select(&lanes.bottom, 0, owner)?;
    let d = select_interval(a, &lanes.extent, owner)?;
    let width = select_interval(a, &lanes.width, owner)?;
    let slope = select_interval(a, &lanes.slope, owner)?;
    let area = select_interval(a, &lanes.area, owner)?;
    let zero = scalar(a, 0.0)?;
    let md = a.mul(&slope, &d)?;
    let md_half = scale(a, &md, 0.5)?;
    let md2 = a.mul(&md, &d)?;
    let md2_half = scale(a, &md2, 0.5)?;
    let td = a.mul(&width, &d)?;
    // Bottom h=d*x² removes the endpoint singularity before differentiation.
    let n0 = a.bounds(width.lower(), width.upper())?;
    let n1 = choose(a, &bottom, &zero, &md)?;
    let n2 = choose(a, &bottom, &md, &zero)?;
    let d0 = choose(a, &bottom, &width, &area)?;
    let d1 = choose(a, &bottom, &zero, &td)?;
    let d2 = choose(a, &bottom, &md_half, &md2_half)?;
    let bottom_prefactor = gravity_depth_root(a, &lanes.gravity, &d, None)?;
    let bottom_prefactor = scale(a, &bottom_prefactor, 2.0)?;
    let sqrt_g = a.sqrt_nonnegative(&lanes.gravity)?;
    let upper_prefactor = a.mul(&d, &sqrt_g)?;
    let prefactor = choose(a, &bottom, &bottom_prefactor, &upper_prefactor)?;
    let n = polynomial_coefficients(a, &n0, &n1, &n2, x)?;
    let denominator = polynomial_coefficients(a, &d0, &d1, &d2, x)?;
    Ok((n, denominator, prefactor))
}

fn polynomial_coefficients<R>(
    a: &mut Arithmetic<'_, R>,
    p0: &Interval<R>,
    p1: &Interval<R>,
    p2: &Interval<R>,
    x: &Interval<R>,
) -> HydraulicResult<[Interval<R>; 5]>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let p2x = a.mul(p2, x)?;
    let linear = a.add(p1, &p2x)?;
    let variable = a.mul(x, &linear)?;
    let value = a.add(p0, &variable)?;
    let twice = scale(a, &p2x, 2.0)?;
    let first = a.add(p1, &twice)?;
    Ok([
        value,
        first,
        a.bounds(p2.lower(), p2.upper())?,
        scalar(a, 0.0)?,
        scalar(a, 0.0)?,
    ])
}
