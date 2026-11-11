use super::algebra::{Bounds, Interior};
use crate::channel::flow::{ChannelError, context::Context, geometry::TensorGeometry};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    pub(super) fn boundary_triangle(&self) -> bool {
        matches!(self.geometry, TensorGeometry::Triangular(_))
    }
    pub(super) fn boundary_sound(&self, h: &Bounds<R>) -> Result<Bounds<R>, ChannelError> {
        let root_g = self.bsqrt(&Bounds::exact(
            &self.constant(&h.lo, self.definition.config.gravity)?,
        ))?;
        let sound = self.bmul(&root_g, &self.bsqrt(h)?)?;
        if self.boundary_triangle() {
            self.bdiv(
                &sound,
                &self.bsqrt(&Bounds::exact(&self.constant(&h.lo, 2.0)?))?,
            )
        } else {
            Ok(sound)
        }
    }
    pub(super) fn boundary_area(
        &self,
        h: &Bounds<R>,
        coefficient: &Tensor<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        let coefficient = if self.boundary_triangle() {
            Bounds {
                lo: crate::channel::flow::triangular::envelope::lower::<R>(
                    &self.client,
                    coefficient,
                )?,
                hi: crate::channel::flow::triangular::envelope::upper::<R>(
                    &self.client,
                    coefficient,
                )?,
            }
        } else {
            Bounds::exact(coefficient)
        };
        let a = self.bmul(&coefficient, h)?;
        if self.boundary_triangle() {
            self.bmul(&a, h)
        } else {
            Ok(a)
        }
    }
    pub(super) fn boundary_pressure(
        &self,
        h: &Bounds<R>,
        coefficient: &Tensor<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        let factor = self.bdiv(
            &Bounds::exact(&self.constant(&h.lo, self.definition.config.gravity)?),
            &Bounds::exact(
                &self.constant(&h.lo, if self.boundary_triangle() { 3.0 } else { 2.0 })?,
            ),
        )?;
        self.bmul(
            &self.bmul(&factor, &self.boundary_area(h, coefficient)?)?,
            h,
        )
    }
    pub(super) fn boundary_curve_nominal(
        &self,
        h: &Tensor<R>,
        hi: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        match &self.geometry {
            TensorGeometry::Triangular(g) => self.triangular_curve_nominal(g, h, hi),
            TensorGeometry::Rectangular(_) => Ok(crate::numerics::curve(
                &self.client,
                h,
                hi,
                self.definition.config.gravity,
            )?),
        }
    }
    fn boundary_curve_point(
        &self,
        h: &Tensor<R>,
        hi: &Tensor<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        if let TensorGeometry::Triangular(g) = &self.geometry {
            let (lo, upper) = self.triangular_curve_enclosure(g, h, hi)?;
            return Ok(Bounds { lo, hi: upper });
        }
        let c = &self.client;
        let hb = Bounds::exact(h);
        let ib = Bounds::exact(hi);
        let rare = self.bscale(
            &self.bsub(&self.boundary_sound(&hb)?, &self.boundary_sound(&ib)?)?,
            2.0,
        )?;
        let sh = Bounds::exact(&c.maximum(h, hi)?);
        let ratio = self.bdiv(&ib, &sh)?;
        let factor = self.bsqrt(&self.bscale(
            &self.badd(&ratio, &Bounds::exact(&self.constant(h, 1.0)?))?,
            0.5,
        )?)?;
        let shock = self.bmul(
            &self.bmul(
                &self.bsqrt(&Bounds::exact(
                    &self.constant(h, self.definition.config.gravity)?,
                ))?,
                &self.bdiv(&self.bsub(&sh, &ib)?, &self.bsqrt(&ib)?)?,
            )?,
            &factor,
        )?;
        let mask = c.gt(h, hi)?;
        Ok(Bounds {
            lo: c.where_cond(&mask, &shock.lo, &rare.lo)?,
            hi: c.where_cond(&mask, &shock.hi, &rare.hi)?,
        })
    }
    pub(super) fn boundary_curve(
        &self,
        h: &Bounds<R>,
        hi: &Bounds<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        let low = self.boundary_curve_point(&h.lo, &hi.hi)?;
        let high = self.boundary_curve_point(&h.hi, &hi.lo)?;
        Ok(Bounds {
            lo: low.lo,
            hi: high.hi,
        })
    }
    pub(super) fn boundary_velocity(
        &self,
        h: &Bounds<R>,
        interior: &Interior<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        self.bsub(
            &interior.velocity_bounds,
            &self.boundary_curve(h, &interior.depth_bounds)?,
        )
    }
    pub(super) fn boundary_subcritical(
        &self,
        h: &Bounds<R>,
        u: &Bounds<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        let c = &self.client;
        let sound = self.boundary_sound(h)?;
        let scale = c.maximum(&self.babs(u)?, &sound.hi)?;
        let margin = c.mul_scalar(&scale, self.definition.config.boundary.critical_relative)?;
        self.boundary_require(
            &c.lt(&self.bsub(u, &sound)?.hi, &c.neg(&margin)?)?,
            "incoming characteristic is unresolved",
        )?;
        self.boundary_require(
            &c.gt(&self.badd(u, &sound)?.lo, &margin)?,
            "outgoing characteristic is unresolved",
        )?;
        Ok(sound)
    }
}
