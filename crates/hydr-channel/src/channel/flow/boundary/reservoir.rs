use super::{
    algebra::{Bounds, Interior},
    depth::{Depth, DepthRequest},
    kernel::Lane,
    reservoir_head::ReservoirHeads,
    types::NativeReservoirTraces,
};
use crate::channel::flow::{
    ChannelError,
    context::Context,
    triangular::envelope::{lower, upper},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
    },
    runtime::Runtime,
    tensor::Tensor,
};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions};
use std::cell::RefCell;

#[derive(Clone)]
pub(in crate::channel::flow) struct Reservoir<R: Runtime<DType = DType>> {
    pub head: Tensor<R>,
    pub bounds: Bounds<R>,
    pub coefficient: Tensor<R>,
}
pub(super) struct RootInput<R: Runtime<DType = DType>> {
    pub interior: Interior<R>,
    pub reservoir: Reservoir<R>,
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
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
        + TwoSumOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(super) fn boundary_reservoir_prepare(
        &self,
        interior: &Interior<R>,
        cap: &Tensor<R>,
        elevation: &Tensor<R>,
        bed: &Tensor<R>,
        k: &Tensor<R>,
    ) -> Result<(DepthRequest<R>, Reservoir<R>), ChannelError> {
        let c = &self.client;
        let z = self.constant(cap, 0.0)?;
        let (head, tail) = c.two_sum(elevation, &c.neg(bed)?)?;
        self.finite(&head, "reservoir local head")?;
        let bounds = Bounds {
            lo: c.where_cond(&c.ge(&tail, &z)?, &head, &lower::<R>(c, &head)?)?,
            hi: c.where_cond(&c.le(&tail, &z)?, &head, &upper::<R>(c, &head)?)?,
        };
        let reservoir = Reservoir {
            head,
            bounds,
            coefficient: k.clone(),
        };
        // Exact outgoing cap equality requires the subtraction witness and outgoing velocity.
        if self.all(&c.eq(&reservoir.head, cap)?)? && self.all(&c.eq(&tail, &z)?)? {
            let u = self.boundary_velocity(&Bounds::exact(cap), interior)?;
            if self.all(&c.ge(&u.lo, &z)?)? {
                return Ok((
                    DepthRequest::Known(Box::new(Depth {
                        value: cap.clone(),
                        bounds: Bounds::exact(cap),
                        approximate: false,
                        iterations: 0,
                    })),
                    reservoir,
                ));
            }
        }
        if self.all(&c.eq(&interior.velocity, &z)?)?
            && self.all(&c.eq(&interior.discharge, &z)?)?
            && self.all(&c.eq(&reservoir.bounds.lo, &interior.depth)?)?
            && self.all(&c.eq(&reservoir.bounds.hi, &interior.depth)?)?
        {
            // At rest, the head and interior-depth hull encloses the unique physical root.
            // Retain section-recovery uncertainty while evaluating the original nominal depth.
            let bounds = Bounds {
                lo: c.minimum(&reservoir.bounds.lo, &interior.depth_bounds.lo)?,
                hi: c.maximum(&reservoir.bounds.hi, &interior.depth_bounds.hi)?,
            };
            let width = upper::<R>(c, &c.sub(&bounds.hi, &bounds.lo)?)?;
            let tolerance = c.add_scalar(
                &c.mul_scalar(&bounds.hi, self.definition.config.root_relative_depth)?,
                self.definition.config.root_absolute_depth,
            )?;
            self.finite(&width, "reservoir rest depth enclosure width")?;
            self.finite(&tolerance, "reservoir rest depth tolerance")?;
            self.boundary_require(
                &c.le(&width, &tolerance)?,
                "certified reservoir bracket exceeds depth controls",
            )?;
            return Ok((
                DepthRequest::Known(Box::new(Depth {
                    value: interior.depth.clone(),
                    bounds,
                    approximate: false,
                    iterations: 0,
                })),
                reservoir,
            ));
        }
        let count = if self.boundary_triangle() { 4.0 } else { 2.0 };
        let sonic = self.bdiv(
            &self.badd(
                &interior.velocity_bounds,
                &self.bscale(&self.boundary_sound(&interior.depth_bounds)?, count)?,
            )?,
            &Bounds::exact(&self.constant(cap, count + 1.0)?),
        )?;
        let mut root = self.bdiv(
            &sonic,
            &self.bsqrt(&Bounds::exact(
                &self.constant(cap, self.definition.config.gravity)?,
            ))?,
        )?;
        if self.boundary_triangle() {
            root = self.bmul(
                &root,
                &self.bsqrt(&Bounds::exact(&self.constant(cap, 2.0)?))?,
            )?;
        }
        let lo = self.bmul(&root, &root)?.hi;
        let search = c.minimum(cap, &upper::<R>(c, &reservoir.bounds.hi)?)?;
        self.boundary_require(
            &c.gt(&lo, &z)?,
            "reservoir lower sonic enclosure is unresolved",
        )?;
        self.boundary_require(&c.lt(&lo, &search)?, "reservoir depth bracket is empty")?;
        let flo = self.reservoir_residual(&lo, interior, &reservoir.bounds, k)?;
        let fhi = self.reservoir_residual(&search, interior, &reservoir.bounds, k)?;
        self.boundary_require(
            &c.le(&flo.hi, &z)?,
            "reservoir head reaches the lower sonic uncertainty band",
        )?;
        self.boundary_require(
            &c.ge(&fhi.lo, &z)?,
            "reservoir upper bracket lacks a directed sign or cap membership is ambiguous",
        )?;
        Ok((
            DepthRequest::Reservoir(Box::new(RootInput {
                interior: interior.clone(),
                reservoir: reservoir.clone(),
                lower: lo,
                upper: search,
            })),
            reservoir,
        ))
    }
    pub(super) fn boundary_reservoir_solve(
        &self,
        inputs: &[&RootInput<R>],
    ) -> Result<Vec<Depth<R>>, ChannelError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let c = &self.client;
        macro_rules! pack {
            ($($field:ident).+) => {{
                let refs: Vec<_> = inputs.iter().map(|x| &x.$($field).+).collect();
                c.cat(&refs, 0)?
            }};
        }
        let interior = Interior {
            depth: pack!(interior.depth),
            velocity: pack!(interior.velocity),
            area: pack!(interior.area),
            discharge: pack!(interior.discharge),
            depth_bounds: Bounds {
                lo: pack!(interior.depth_bounds.lo),
                hi: pack!(interior.depth_bounds.hi),
            },
            velocity_bounds: Bounds {
                lo: pack!(interior.velocity_bounds.lo),
                hi: pack!(interior.velocity_bounds.hi),
            },
        };
        let head = pack!(reservoir.head);
        let hb = Bounds {
            lo: pack!(reservoir.bounds.lo),
            hi: pack!(reservoir.bounds.hi),
        };
        let k = pack!(reservoir.coefficient);
        let lo = pack!(lower);
        let hi = pack!(upper);
        let z = self.constant(&head, 0.0)?;
        let diagnostic = RefCell::new(None);
        let result = c.bisect_batched(
            |h| {
                self.reservoir_nominal(h, &interior, &head, &k)
                    .map_err(|error| {
                        if let Ok(mut slot) = diagnostic.try_borrow_mut() {
                            *slot = Some(error);
                        }
                        numr::error::Error::InvalidArgument {
                            arg: "channel reservoir head",
                            reason: "physical root evaluation rejected the candidate".into(),
                        }
                    })
            },
            &lo,
            &hi,
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
        let root = result.map_err(|error| ChannelError::numerical(format!("scalar channel Solvr reservoir root: {error}. Increase root_iterations or rescale inputs")))?;
        let left = self.reservoir_residual(&root.lower, &interior, &hb, &k)?;
        let right = self.reservoir_residual(&root.upper, &interior, &hb, &k)?;
        // The increasing head derivative is at least one throughout the wet branch.
        let delta = upper::<R>(
            c,
            &c.mul_scalar(&c.maximum(&self.babs(&left)?, &self.babs(&right)?)?, 2.0)?,
        )?;
        let a = c.maximum(&lo, &lower::<R>(c, &c.sub(&root.lower, &delta)?)?)?;
        let b = c.minimum(&hi, &upper::<R>(c, &c.add(&root.upper, &delta)?)?)?;
        let ra = self.reservoir_residual(&a, &interior, &hb, &k)?;
        let rb = self.reservoir_residual(&b, &interior, &hb, &k)?;
        self.boundary_require(
            &c.le(&ra.hi, &z)?,
            "reservoir root lower residual lacks a directed sign",
        )?;
        self.boundary_require(
            &c.ge(&rb.lo, &z)?,
            "reservoir root upper residual lacks a directed sign",
        )?;
        let width = upper::<R>(c, &c.sub(&b, &a)?)?;
        let tolerance = c.add_scalar(
            &c.mul_scalar(&b, self.definition.config.root_relative_depth)?,
            self.definition.config.root_absolute_depth,
        )?;
        self.boundary_require(
            &c.le(&width, &tolerance)?,
            "certified reservoir bracket exceeds depth controls",
        )?;
        let value = c.add(
            &root.lower,
            &c.mul_scalar(&c.sub(&root.upper, &root.lower)?, 0.5)?,
        )?;
        (0..inputs.len())
            .map(|index| {
                Ok(Depth {
                    value: value.narrow(0, index, 1)?,
                    bounds: Bounds {
                        lo: a.narrow(0, index, 1)?,
                        hi: b.narrow(0, index, 1)?,
                    },
                    approximate: false,
                    iterations: root.iterations,
                })
            })
            .collect()
    }
    pub(in crate::channel::flow) fn boundary_reservoir_receipt(
        &self,
        lane: &mut Lane<R>,
        reservoir: &Reservoir<R>,
        bounds: &Bounds<R>,
    ) -> Result<(), ChannelError> {
        let c = &self.client;
        let trace = &lane.trace;
        let h = Bounds::exact(&trace.depth_m);
        let area = Bounds::exact(&trace.area_m2);
        let u = self.bdiv(&Bounds::exact(&trace.outward_discharge_m3_per_s), &area)?;
        let ReservoirHeads {
            velocity_head: velocity,
            entrance_loss_head: loss,
            dissipation_head: dissipation,
        } = self.reservoir_heads(&u, &reservoir.coefficient)?;
        let total = self.reservoir_add_head(&self.reservoir_add_head(&h, &velocity)?, &loss)?;
        let residual = self.babs(&self.bsub(&total, &reservoir.bounds)?)?;
        let scale = c.maximum(
            &c.maximum(&trace.depth_m, &self.babs(&reservoir.bounds)?)?,
            &c.maximum(&velocity.hi, &loss.hi)?,
        )?;
        let config = &self.definition.config.boundary;
        let tolerance = c.add_scalar(
            &c.mul_scalar(&scale, config.reservoir_head_relative)?,
            config.reservoir_head_absolute_m,
        )?;
        self.finite(&residual, "reservoir delivered head residual")?;
        self.finite(&dissipation.hi, "reservoir dissipation head")?;
        self.boundary_require(
            &c.le(&residual, &tolerance)?,
            "delivered reservoir head exceeds dedicated controls",
        )?;
        lane.trace.reservoir = Some(NativeReservoirTraces {
            depth_lower_m: bounds.lo.clone(),
            depth_upper_m: bounds.hi.clone(),
            head_compatibility_abs_bound_m: residual,
            dissipation_head_lower_m: dissipation.lo,
            dissipation_head_upper_m: dissipation.hi,
            endpoints: trace.endpoints.clone(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{context, interior};
    use super::*;
    use std::sync::Arc;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn unequal_or_uncertain_heads_and_nonzero_motion_retain_numerical_solving() -> TestResult {
        for triangle in [false, true] {
            let ctx = context(triangle, 0.0)?;
            for (head, bed, velocity, discharge) in [
                (f64::from_bits(1.0_f64.to_bits() - 1), 0.0, 0.0, 0.0),
                (f64::from_bits(1.0_f64.to_bits() + 1), 0.0, 0.0, 0.0),
                (1.0, 0.0, 1e-3, 1e-3),
                (1.0, 0.0, 1e-3, 0.0),
                (1.0, 0.0, 0.0, 1e-3),
                (1.0, 2.0_f64.powi(-54), 0.0, 0.0),
            ] {
                let state = interior(&ctx, velocity, discharge)?;
                let (request, reservoir) = ctx.boundary_reservoir_prepare(
                    &state,
                    &ctx.tensor(&[4.0])?,
                    &ctx.tensor(&[head])?,
                    &ctx.tensor(&[bed])?,
                    &ctx.tensor(&[1.0])?,
                )?;
                assert!(matches!(request, DepthRequest::Reservoir(_)));
                if bed > 0.0 {
                    assert_eq!(reservoir.head.item::<f64>()?, 1.0);
                    assert!(reservoir.bounds.lo.item::<f64>()? < 1.0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn resting_root_retains_existing_depth_width_controls() -> TestResult {
        let mut ctx = context(false, 0.0)?;
        Arc::make_mut(&mut ctx.definition)
            .config
            .root_absolute_depth = f64::from_bits(1);
        Arc::make_mut(&mut ctx.definition)
            .config
            .root_relative_depth = 0.0;
        let result = ctx.boundary_reservoir_prepare(
            &interior(&ctx, 0.0, 0.0)?,
            &ctx.tensor(&[4.0])?,
            &ctx.tensor(&[1.0])?,
            &ctx.tensor(&[0.0])?,
            &ctx.tensor(&[0.0])?,
        );
        assert!(matches!(result, Err(ChannelError::Numerical { .. })));
        Ok(())
    }
}
