use super::super::envelope::{lower, upper};
use super::sample_bounds::Sample;
use crate::channel::flow::{
    ChannelError, context::Context, riemann_diagnostics::RiemannDiagnosticState,
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

pub(super) struct Candidate<R: Runtime<DType = DType>> {
    pub possible: Tensor<R>,
    pub selected: Tensor<R>,
    pub sample: Sample<R>,
}

pub(in crate::channel::flow::triangular) struct SampleFlux<R: Runtime<DType = DType>> {
    pub water: Tensor<R>,
    pub momentum: Tensor<R>,
    pub water_bound: Tensor<R>,
    pub momentum_bound: Tensor<R>,
    pub diagnostics: RiemannDiagnosticState<R>,
}

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
    pub(super) fn triangular_sample_flux(
        &self,
        candidates: Vec<Candidate<R>>,
        iterations: usize,
    ) -> Result<SampleFlux<R>, ChannelError> {
        let first = candidates.first().ok_or_else(|| {
            ChannelError::numerical(
                "Godunov sample candidates absent: inspect Riemann classification",
            )
        })?;
        let c = &self.client;
        let zero = self.constant(&first.possible, 0.0)?;
        let mut water = zero.clone();
        let mut momentum = zero.clone();
        let mut water_bound = zero.clone();
        let mut momentum_bound = zero.clone();
        let mut ql = zero.clone();
        let mut qh = zero.clone();
        let mut ml = zero.clone();
        let mut mh = zero.clone();
        let mut present = zero.clone();
        let mut selected = zero.clone();
        for candidate in &candidates {
            let sample = &candidate.sample;
            let physical = self.triangular_sample_physical(
                &sample.face,
                &sample.area_bound,
                &sample.discharge_bound,
            )?;
            let (lower_product, upper_advective) = self.triangular_signed_product(
                &sample.lower_discharge,
                &sample.upper_discharge,
                &sample.lower_velocity,
                &sample.upper_velocity,
            )?;
            let lower_advective = c.maximum(&lower_product, &zero)?;
            let gp = c.div(
                &self.constant(&zero, self.definition.config.gravity)?,
                &self.constant(&zero, 3.0)?,
            )?;
            let gl = c.maximum(&lower::<R>(c, &gp)?, &zero)?;
            let gu = upper::<R>(c, &gp)?;
            let lower_pressure = lower::<R>(
                c,
                &c.mul(
                    &c.maximum(&lower::<R>(c, &c.mul(&gl, &sample.lower_area)?)?, &zero)?,
                    &sample.lower_depth,
                )?,
            )?;
            let upper_pressure = upper::<R>(
                c,
                &c.mul(
                    &upper::<R>(c, &c.mul(&gu, &sample.upper_area)?)?,
                    &sample.upper_depth,
                )?,
            )?;
            let lower_momentum = c.maximum(
                &lower::<R>(c, &c.add(&lower_advective, &lower_pressure)?)?,
                &zero,
            )?;
            let upper_momentum = upper::<R>(c, &c.add(&upper_advective, &upper_pressure)?)?;
            let merge = |old: &Tensor<R>,
                         new: &Tensor<R>,
                         minimum: bool|
             -> numr::error::Result<Tensor<R>> {
                let union = if minimum {
                    c.minimum(old, new)?
                } else {
                    c.maximum(old, new)?
                };
                c.where_cond(
                    &candidate.possible,
                    &c.where_cond(&present, &union, new)?,
                    old,
                )
            };
            ql = merge(&ql, &sample.lower_discharge, true)?;
            qh = merge(&qh, &sample.upper_discharge, false)?;
            ml = merge(&ml, &lower_momentum, true)?;
            mh = merge(&mh, &upper_momentum, false)?;
            present = c.maximum(&present, &candidate.possible)?;
            selected = c.maximum(&selected, &candidate.selected)?;
            water = c.where_cond(&candidate.selected, &sample.face.discharge, &water)?;
            momentum = c.where_cond(&candidate.selected, &physical.momentum, &momentum)?;
            water_bound =
                c.where_cond(&candidate.selected, &sample.discharge_bound, &water_bound)?;
            momentum_bound = c.where_cond(
                &candidate.selected,
                &physical.momentum_bound,
                &momentum_bound,
            )?;
        }
        if !self.all(&c.mul(&present, &selected)?)? {
            return Err(ChannelError::numerical(
                "Godunov interface sample unresolved: reduce the interval or rescale states",
            ));
        }
        let water_width = c.maximum(&upper::<R>(c, &c.sub(&qh, &ql)?)?, &zero)?;
        let momentum_width = c.maximum(&upper::<R>(c, &c.sub(&mh, &ml)?)?, &zero)?;
        for (name, value, lo, hi, width, absolute) in [
            (
                "water",
                &water,
                &ql,
                &qh,
                &water_width,
                self.definition.config.riemann_water_flux_absolute_m3_per_s,
            ),
            (
                "momentum",
                &momentum,
                &ml,
                &mh,
                &momentum_width,
                self.definition
                    .config
                    .riemann_momentum_flux_absolute_m4_per_s2,
            ),
        ] {
            self.finite(width, "Godunov sampled flux width")?;
            let scale = c.maximum(&c.abs(lo)?, &c.abs(hi)?)?;
            let budget = c.add_scalar(
                &c.mul_scalar(&scale, self.definition.config.riemann_flux_relative)?,
                absolute,
            )?;
            let admitted = c.mul(
                &c.le(width, &budget)?,
                &c.mul(&c.ge(value, lo)?, &c.le(value, hi)?)?,
            )?;
            if !self.all(&admitted)? {
                let mut lane = 0;
                let mut count = admitted.numel();
                if count == 0 {
                    return Err(ChannelError::numerical(
                        "Godunov rejected-lane array is empty: inspect face construction",
                    ));
                }
                while count > 1 {
                    let half = count / 2;
                    let first_half = admitted.narrow(0, lane, half)?;
                    if c.min(&first_half, &[0], false)?.item::<f64>()? != 1.0 {
                        count = half;
                    } else {
                        lane += half;
                        count -= half;
                    }
                }
                let read = |tensor: &Tensor<R>| tensor.narrow(0, lane, 1)?.item::<f64>();
                let mut states = Vec::new();
                for (candidate_index, candidate) in candidates.iter().enumerate() {
                    let possible = read(&candidate.possible)?;
                    let selected = read(&candidate.selected)?;
                    if possible != 0.0 || selected != 0.0 {
                        let sample = &candidate.sample;
                        states.push(format!(
                            "candidate={candidate_index}, possible={possible}, selected={selected}, A={}, h={} in [{},{}], u={} in [{},{}], Q={} in [{},{}]",
                            read(&sample.face.area)?, read(&sample.face.depth)?, read(&sample.lower_depth)?, read(&sample.upper_depth)?,
                            read(&sample.face.velocity)?, read(&sample.lower_velocity)?, read(&sample.upper_velocity)?,
                            read(&sample.face.discharge)?, read(&sample.lower_discharge)?, read(&sample.upper_discharge)?,
                        ));
                    }
                }
                return Err(ChannelError::numerical(format!(
                    "Godunov {name} sampled flux enclosure rejects lane={lane}, value={}, bounds=[{},{}], width={}, budget={}, root_iterations={iterations}, states=[{}]: increase root iterations or rescale states",
                    read(value)?,
                    read(lo)?,
                    read(hi)?,
                    read(width)?,
                    read(&budget)?,
                    states.join(" | "),
                )));
            }
        }
        Ok(SampleFlux {
            water,
            momentum,
            water_bound,
            momentum_bound,
            diagnostics: RiemannDiagnosticState {
                water_flux_width: water_width,
                momentum_flux_width: momentum_width,
                root_iterations: iterations,
            },
        })
    }
}
