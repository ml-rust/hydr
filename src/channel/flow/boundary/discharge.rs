use super::algebra::{Bounds, Interior};
use crate::channel::flow::{
    ChannelError,
    context::Context,
    triangular::envelope::{lower, upper},
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions};
use std::cell::RefCell;

pub(super) struct Depth<R: Runtime<DType = DType>> {
    pub value: Tensor<R>,
    pub bounds: Bounds<R>,
    pub approximate: bool,
    pub iterations: usize,
}
pub(super) struct RootInput<R: Runtime<DType = DType>> {
    pub interior: Interior<R>,
    pub coefficient: Tensor<R>,
    pub target: Tensor<R>,
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
}

pub(super) enum DepthRequest<R: Runtime<DType = DType>> {
    Known(Box<Depth<R>>),
    Root(Box<RootInput<R>>),
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(super) fn boundary_rate(
        &self,
        h: &Tensor<R>,
        interior: &Interior<R>,
        coefficient: &Tensor<R>,
    ) -> Result<Bounds<R>, ChannelError> {
        self.bmul(
            &self.boundary_area(&Bounds::exact(h), coefficient)?,
            &self.boundary_velocity(&Bounds::exact(h), interior)?,
        )
    }
    pub(super) fn boundary_discharge_prepare(
        &self,
        interior: &Interior<R>,
        coefficient: &Tensor<R>,
        cap: &Tensor<R>,
        target: &Tensor<R>,
    ) -> Result<DepthRequest<R>, ChannelError> {
        let c = &self.client;
        let hi = &interior.depth;
        let zero = self.constant(hi, 0.0)?;
        let count = if self.boundary_triangle() { 4.0 } else { 2.0 };
        let ci = self.boundary_sound(&interior.depth_bounds)?;
        let sonic = self.bdiv(
            &self.badd(&interior.velocity_bounds, &self.bscale(&ci, count)?)?,
            &Bounds::exact(&self.constant(hi, count + 1.0)?),
        )?;
        let sonic_root = self.bdiv(
            &sonic,
            &self.bsqrt(&Bounds::exact(
                &self.constant(hi, self.definition.config.gravity)?,
            ))?,
        )?;
        let sonic_root = if self.boundary_triangle() {
            self.bmul(
                &sonic_root,
                &self.bsqrt(&Bounds::exact(&self.constant(hi, 2.0)?))?,
            )?
        } else {
            sonic_root
        };
        let hsonic = self.bmul(&sonic_root, &sonic_root)?;
        let lo = hsonic.hi;
        self.boundary_require(&c.gt(&lo, &zero)?, "lower sonic enclosure is unresolved")?;
        let factor = if self.boundary_triangle() { 4.0 } else { 9.0 };
        let capped = c.le(
            &c.div(cap, &self.constant(cap, factor)?)?,
            &interior.depth_bounds.hi,
        )?;
        let safe = c.minimum(
            &interior.depth_bounds.hi,
            &c.div(cap, &self.constant(cap, factor)?)?,
        )?;
        let search = c.minimum(
            cap,
            &c.where_cond(&capped, cap, &upper::<R>(c, &c.mul_scalar(&safe, factor)?)?)?,
        )?;
        self.boundary_require(&c.lt(&lo, &search)?, "discharge depth bracket is empty")?;
        let flo = self.bsub(
            &self.boundary_rate(&lo, interior, coefficient)?,
            &Bounds::exact(target),
        )?;
        let fhi = self.bsub(
            &self.boundary_rate(&search, interior, coefficient)?,
            &Bounds::exact(target),
        )?;
        self.boundary_require(
            &c.ge(&flo.lo, &zero)?,
            "requested discharge reaches the lower sonic uncertainty band",
        )?;
        if !self.all(&c.le(&fhi.hi, &zero)?)? {
            self.boundary_require(
                &capped,
                "artificial upper bracket lacks a directed residual sign",
            )?;
            self.boundary_require(
                &c.le(&fhi.lo, &zero)?,
                "requested discharge lies below the certified cap range",
            )?;
            return Ok(DepthRequest::Known(Box::new(Depth {
                value: cap.clone(),
                bounds: Bounds::exact(cap),
                approximate: true,
                iterations: 0,
            })));
        }
        Ok(DepthRequest::Root(Box::new(RootInput {
            interior: interior.clone(),
            coefficient: coefficient.clone(),
            target: target.clone(),
            lower: lo,
            upper: search,
        })))
    }

    pub(super) fn boundary_discharge_solve(
        &self,
        inputs: &[&RootInput<R>],
    ) -> Result<Vec<Depth<R>>, ChannelError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let c = &self.client;
        macro_rules! pack {
            ($field:ident) => {{
                let refs: Vec<_> = inputs.iter().map(|input| &input.$field).collect();
                c.cat(&refs, 0)?
            }};
        }
        macro_rules! pack_state {
            ($field:ident) => {{
                let refs: Vec<_> = inputs.iter().map(|input| &input.interior.$field).collect();
                c.cat(&refs, 0)?
            }};
        }
        macro_rules! pack_bounds {
            ($field:ident,$side:ident) => {{
                let refs: Vec<_> = inputs
                    .iter()
                    .map(|input| &input.interior.$field.$side)
                    .collect();
                c.cat(&refs, 0)?
            }};
        }
        let interior = Interior {
            depth: pack_state!(depth),
            velocity: pack_state!(velocity),
            area: pack_state!(area),
            discharge: pack_state!(discharge),
            depth_bounds: Bounds {
                lo: pack_bounds!(depth_bounds, lo),
                hi: pack_bounds!(depth_bounds, hi),
            },
            velocity_bounds: Bounds {
                lo: pack_bounds!(velocity_bounds, lo),
                hi: pack_bounds!(velocity_bounds, hi),
            },
        };
        let interior = &interior;
        let hi = &interior.depth;
        let ui = &interior.velocity;
        let coefficient = &pack!(coefficient);
        let target = &pack!(target);
        let lo = pack!(lower);
        let search = pack!(upper);
        let zero = self.constant(hi, 0.0)?;
        let diagnostic = RefCell::new(None);
        let root = c.bisect_batched(
            |h| {
                let result = (|| -> Result<Tensor<R>, ChannelError> {
                    let velocity = c.sub(ui, &self.boundary_curve_nominal(h, hi)?)?;
                    let area = if self.boundary_triangle() {
                        c.mul(&c.mul(coefficient, h)?, h)?
                    } else {
                        c.mul(coefficient, h)?
                    };
                    Ok(c.sub(&c.mul(&area, &velocity)?, target)?)
                })();
                result.map_err(|error| {
                    if let Ok(mut slot) = diagnostic.try_borrow_mut() {
                        *slot = Some(error);
                    }
                    numr::error::Error::InvalidArgument {
                        arg: "channel boundary discharge",
                        reason: "physical root evaluation rejected the candidate".into(),
                    }
                })
            },
            &lo,
            &search,
            &BatchedBisectionOptions {
                atol: f64::from_bits(1),
                rtol: 4.0 * f64::EPSILON,
                max_iter: self.definition.config.root_iterations,
                max_expansions: 0,
                expansion_factor: 2.0,
            },
        );
        if let Some(error) = diagnostic.into_inner() {
            return Err(error);
        }
        let root=root.map_err(|error|ChannelError::numerical(format!("scalar channel Solvr discharge root: {error}. Increase root_iterations or rescale inputs")))?;
        let value = c.add(
            &root.lower,
            &c.mul_scalar(&c.sub(&root.upper, &root.lower)?, 0.5)?,
        )?;
        let reference = c.add(&c.mul_scalar(&root.lower, 0.5)?, &c.mul_scalar(&lo, 0.5)?)?;
        let sound = self.boundary_sound(&Bounds::exact(&reference))?;
        let velocity = self.boundary_velocity(&Bounds::exact(&reference), interior)?;
        let gap = self.bsub(&sound, &velocity)?;
        let derivative_area = if self.boundary_triangle() {
            self.bscale(
                &self.bdiv(
                    &self.boundary_area(&Bounds::exact(&reference), coefficient)?,
                    &Bounds::exact(&reference),
                )?,
                2.0,
            )?
        } else {
            Bounds::exact(coefficient)
        };
        let derivative = self.bmul(&derivative_area, &gap)?;
        self.boundary_require(
            &c.gt(&derivative.lo, &zero)?,
            "discharge derivative lower bound is unresolved",
        )?;
        let left_residual = self.bsub(
            &self.boundary_rate(&root.lower, interior, coefficient)?,
            &Bounds::exact(target),
        )?;
        let right_residual = self.bsub(
            &self.boundary_rate(&root.upper, interior, coefficient)?,
            &Bounds::exact(target),
        )?;
        let residual = c.maximum(&self.babs(&left_residual)?, &self.babs(&right_residual)?)?;
        let delta = upper::<R>(
            c,
            &c.mul_scalar(&upper::<R>(c, &c.div(&residual, &derivative.lo)?)?, 2.0)?,
        )?;
        let room = lower::<R>(c, &c.sub(&root.lower, &reference)?)?;
        self.boundary_require(
            &c.le(&delta, &room)?,
            "one discharge residual correction exceeds its derivative range",
        )?;
        let a = c.maximum(&lo, &lower::<R>(c, &c.sub(&root.lower, &delta)?)?)?;
        let b = c.minimum(&search, &upper::<R>(c, &c.add(&root.upper, &delta)?)?)?;
        let ra = self.bsub(
            &self.boundary_rate(&a, interior, coefficient)?,
            &Bounds::exact(target),
        )?;
        let rb = self.bsub(
            &self.boundary_rate(&b, interior, coefficient)?,
            &Bounds::exact(target),
        )?;
        self.boundary_require(
            &c.ge(&ra.lo, &zero)?,
            "root lower residual lacks a directed sign",
        )?;
        self.boundary_require(
            &c.le(&rb.hi, &zero)?,
            "root upper residual lacks a directed sign",
        )?;
        let depth_width = upper::<R>(c, &c.sub(&b, &a)?)?;
        let depth_tolerance = c.add_scalar(
            &c.mul_scalar(&b, self.definition.config.root_relative_depth)?,
            self.definition.config.root_absolute_depth,
        )?;
        self.boundary_require(
            &c.le(&depth_width, &depth_tolerance)?,
            "certified discharge bracket exceeds depth controls",
        )?;
        (0..inputs.len())
            .map(|index| {
                Ok(Depth {
                    value: value.narrow(0, index, 1)?,
                    bounds: Bounds::exact(&value.narrow(0, index, 1)?),
                    approximate: false,
                    iterations: root.iterations,
                })
            })
            .collect()
    }
}
