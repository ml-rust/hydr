use super::{fourth_from_coefficients, polynomial_coefficients, value_from_coefficients};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{choose, reject_mask, scalar, scale, select_interval},
        error::{Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection},
    },
    section::WetDepthGeometry,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

pub(crate) struct WetDepthPotential<R: Runtime<DType = DType>> {
    numerator: [Interval<R>; 5],
    denominator: [Interval<R>; 5],
    prefactor: Interval<R>,
    active: Tensor<R>,
}

struct WetDomain<R: Runtime<DType = DType>> {
    original_rows: Tensor<R>,
    owners: Tensor<R>,
}

impl<R> WetDomain<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    fn reject(
        &self,
        a: &Arithmetic<'_, R>,
        mask: &Tensor<R>,
        context: &str,
    ) -> HydraulicResult<()> {
        if mask.numel() == 0 {
            return Ok(());
        }
        let indices = a.client().nonzero(mask)?.flatten()?;
        if indices.numel() != 0 {
            let first = indices.narrow(0, 0, 1)?;
            let original = a
                .client()
                .index_select(&self.original_rows, 0, &first)?
                .item::<i64>()?;
            let owner = a
                .client()
                .index_select(&self.owners, 0, &first)?
                .item::<i64>()?;
            return Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::QuadratureUnresolved,
                query: usize::try_from(original).ok(),
                section_owner: usize::try_from(owner).ok(),
                context: context.into(),
            });
        }
        Ok(())
    }
}

impl<R> WetDepthPotential<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn from_geometry(
        a: &mut Arithmetic<'_, R>,
        geometry: &WetDepthGeometry<R>,
        owner: &Tensor<R>,
        parameter: &Interval<R>,
        gravity: f64,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        if owner.ndim() != 1
            || owner.dtype() != DType::I64
            || !owner.device().is_same(c.device())
            || [parameter.lower(), parameter.upper()].iter().any(|value| {
                value.shape() != owner.shape()
                    || value.dtype() != DType::F64
                    || !value.device().is_same(c.device())
            })
        {
            return Err(HydraulicError::Numerical { context: "wet depth potential metadata differs from rank-one I64 owners and matching F64 parameters. Supply matching native operands".into() });
        }
        let domain = WetDomain {
            original_rows: c.index_select(geometry.original_rows(), 0, owner)?,
            owners: c.index_select(geometry.owners(), 0, owner)?,
        };
        let parameter = a.bounds(parameter.lower(), parameter.upper())?;
        let zero = scalar(a, 0.0)?;
        let one = scalar(a, 1.0)?;
        domain.reject(a, &c.maximum(&c.lt(parameter.lower(), zero.lower())?, &c.gt(parameter.upper(), one.upper())?)?,
            "wet depth potential parameter lies outside [0,1]. Supply a panel inside its proved physical segment")?;
        let active = c.index_select(geometry.active_mask(), 0, owner)?;
        let width = select_interval(a, geometry.start_width(), owner)?;
        let area = select_interval(a, geometry.start_area(), owner)?;
        let slope = select_interval(a, geometry.slope(), owner)?;
        let depth = select_interval(a, geometry.physical_depth_gap(), owner)?;
        let md = a.mul(&slope, &depth)?;
        let md2 = a.mul(&md, &depth)?;
        let half_md2 = scale(a, &md2, 0.5)?;
        let td = a.mul(&width, &depth)?;
        let gravity = scalar(a, gravity)?;
        reject_mask(
            a,
            &c.le(gravity.lower(), zero.lower())?,
            PrismaticRiemannRejection::QuadratureUnresolved,
            "wet depth potential gravity lacks a positive lower bound. Supply checked positive gravity",
        )?;
        let sqrt_g = a.sqrt_nonnegative(&gravity)?;
        // Signed traversal belongs in the prefactor, never inside a gravity-depth square root.
        let prefactor = a.mul(&depth, &sqrt_g)?;
        let mut numerator = polynomial_coefficients(a, &width, &md, &zero, &parameter)?;
        let mut denominator = polynomial_coefficients(a, &area, &td, &half_md2, &parameter)?;
        let width_hull = select_interval(a, geometry.width_hull(), owner)?;
        let area_hull = select_interval(a, geometry.area_hull(), owner)?;
        let intersect = |a: &mut Arithmetic<'_, R>,
                         value: &Interval<R>,
                         physical: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            let lower = c.maximum(value.lower(), physical.lower())?;
            let upper = c.minimum(value.upper(), physical.upper())?;
            domain.reject(a, &c.gt(&lower, &upper)?, "wet polynomial value conflicts with its proved physical hull. Rebuild compatible endpoint geometry")?;
            Ok(a.bounds(&lower, &upper)?)
        };
        // These constrain the same physical zeroth values. Higher coefficients retain their complete derivative uncertainty.
        numerator[0] = intersect(a, &numerator[0], &width_hull)?;
        denominator[0] = intersect(a, &denominator[0], &area_hull)?;
        domain.reject(a, &c.maximum(&c.le(numerator[0].lower(), zero.lower())?, &c.le(denominator[0].lower(), zero.lower())?)?,
            "wet polynomial area or width lacks a positive lower bound. Tighten the physical endpoint certificates")?;
        Ok(Self {
            numerator,
            denominator,
            prefactor,
            active,
        })
    }

    pub(crate) fn value(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Interval<R>> {
        value_from_coefficients(a, &self.numerator, &self.denominator, &self.prefactor)
    }

    pub(crate) fn fourth_derivative(
        &self,
        a: &mut Arithmetic<'_, R>,
    ) -> HydraulicResult<Interval<R>> {
        let zero = scalar(a, 0.0)?;
        let fourth = fourth_from_coefficients(
            a,
            &self.numerator,
            &self.denominator,
            &self.prefactor,
            &zero,
        )?;
        choose(a, &self.active, &fourth, &zero)
    }
}
