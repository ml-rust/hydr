//! Immutable persistent batch lifecycle and accepted-output projections.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::kernel::{TensorSoilContext, TensorSoilState};
use super::parameters::METRES_PER_FOOT;
use super::{
    ModifiedGreenAmptParameters, NativeSoilInput, NativeSoilState, SoilError, SoilInput, SoilOutput,
};
use std::sync::Arc;

/// Persistent native soil memory with an explicitly configured one-thread numr CPU client.
#[derive(Debug, Clone)]
pub struct ModifiedGreenAmptBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<TensorSoilContext<R>>,
    pub(super) state: TensorSoilState<R>,
}

/// Complete candidate state. Discarding it leaves the borrowed accepted batch unchanged.
#[derive(Debug)]
pub struct ModifiedGreenAmptTrial<R: Runtime<DType = DType>> {
    pub(super) next: ModifiedGreenAmptBatch<R>,
    pub(super) native_rate: Tensor<R>,
    pub(super) native_depth: Tensor<R>,
    pub(super) si_rate: Tensor<R>,
    pub(super) si_depth: Tensor<R>,
}

impl<R> ModifiedGreenAmptBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Explicitly select the original EPA initialized antecedent state.
    pub fn initialized(
        client: R::Client,
        parameters: &[ModifiedGreenAmptParameters],
    ) -> Result<Self, SoilError> {
        let states: Vec<_> = parameters
            .iter()
            .map(|p| NativeSoilState {
                imd: p.deficit,
                f_ft: 0.0,
                fu_ft: 0.0,
                saturated: false,
                t_s: 0.0,
            })
            .collect();
        Self::from_native_states(client, parameters, &states)
    }

    /// Import complete checked native memory without inferred history or clamping.
    pub fn from_native_states(
        client: R::Client,
        parameters: &[ModifiedGreenAmptParameters],
        states: &[NativeSoilState],
    ) -> Result<Self, SoilError> {
        let context = Arc::new(TensorSoilContext::<R>::new(client, parameters)?);
        let state = TensorSoilState::<R>::from_native(&context, states)?;
        Ok(Self { context, state })
    }

    /// Immutable checked parameters in cell order.
    pub fn parameters(&self) -> &[ModifiedGreenAmptParameters] {
        &self.context.parameters
    }

    /// Project complete native state at an application output or restart boundary.
    pub fn native_states(&self) -> Result<Vec<NativeSoilState>, SoilError> {
        self.state.project(&self.context)
    }

    /// Compute an immutable SI-input trial using tensor division for the native bridge.
    pub fn trial(
        &self,
        dt_s: f64,
        inputs: &[SoilInput],
    ) -> Result<ModifiedGreenAmptTrial<R>, SoilError> {
        self.check_inputs(dt_s, inputs.len())?;
        for (index, input) in inputs.iter().enumerate() {
            check_input(index, input.incoming_rate_m_per_s, input.ponded_depth_m)?;
        }
        let rain = self.context.tensor(
            &inputs
                .iter()
                .map(|x| x.incoming_rate_m_per_s)
                .collect::<Vec<_>>(),
        )?;
        let depth = self
            .context
            .tensor(&inputs.iter().map(|x| x.ponded_depth_m).collect::<Vec<_>>())?;
        let c = &self.context.client;
        self.compute(
            dt_s,
            c.div_scalar(&rain, METRES_PER_FOOT)?,
            c.div_scalar(&depth, METRES_PER_FOOT)?,
        )
    }

    /// Compute an immutable trial from exact feet/second and feet inputs.
    pub fn trial_native(
        &self,
        dt_s: f64,
        inputs: &[NativeSoilInput],
    ) -> Result<ModifiedGreenAmptTrial<R>, SoilError> {
        self.check_inputs(dt_s, inputs.len())?;
        for (index, input) in inputs.iter().enumerate() {
            check_input(index, input.incoming_rate_ft_per_s, input.ponded_depth_ft)?;
        }
        self.compute(
            dt_s,
            self.context.tensor(
                &inputs
                    .iter()
                    .map(|x| x.incoming_rate_ft_per_s)
                    .collect::<Vec<_>>(),
            )?,
            self.context
                .tensor(&inputs.iter().map(|x| x.ponded_depth_ft).collect::<Vec<_>>())?,
        )
    }

    pub(super) fn check_inputs(&self, dt: f64, count: usize) -> Result<(), SoilError> {
        if !dt.is_finite() || dt <= 0.0 || count != self.parameters().len() {
            return Err(SoilError::input(format!(
                "dt={dt}, inputs={count}: require positive finite dt and {} cells",
                self.parameters().len()
            )));
        }
        Ok(())
    }

    fn compute(
        &self,
        dt: f64,
        rain: Tensor<R>,
        depth: Tensor<R>,
    ) -> Result<ModifiedGreenAmptTrial<R>, SoilError> {
        self.compute_traced(dt, rain, depth).map(|(trial, _)| trial)
    }

    pub(super) fn compute_traced(
        &self,
        dt: f64,
        rain: Tensor<R>,
        depth: Tensor<R>,
    ) -> Result<
        (
            ModifiedGreenAmptTrial<R>,
            super::kernel::transition::NativeTrace<R>,
        ),
        SoilError,
    > {
        self.context.finite(&rain, "native incoming rate (ft/s)")?;
        self.context.finite(&depth, "native ponded depth (ft)")?;
        let (state, native_rate, trace) =
            super::kernel::advance(&self.context, &self.state, dt, &rain, &depth)?;
        let c = &self.context.client;
        let native_depth = c.mul_scalar(&native_rate, dt)?;
        let si_rate = c.mul_scalar(&native_rate, METRES_PER_FOOT)?;
        let si_depth = c.mul_scalar(&si_rate, dt)?;
        let positive = c.gt(&native_rate, &self.context.zero)?;
        for (quantity, value) in [
            ("native interval depth (ft)", &native_depth),
            ("SI infiltration rate (m/s)", &si_rate),
            ("SI interval depth (m)", &si_depth),
        ] {
            self.context.finite(value, quantity)?;
            self.context.reject_mask(
                &c.mul(&positive, &c.eq(value, &self.context.zero)?)?,
                quantity,
                &[("value", value), ("native rate (ft/s)", &native_rate)],
                &[("dt (s)", dt)],
                "Positive infiltration became zero. Rescale the interval or parameters",
            )?;
        }
        Ok((
            ModifiedGreenAmptTrial {
                next: Self {
                    context: Arc::clone(&self.context),
                    state,
                },
                native_rate,
                native_depth,
                si_rate,
                si_depth,
            },
            trace,
        ))
    }
}

impl<R> ModifiedGreenAmptTrial<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Project raw empirical interval rates and depths in both unit systems.
    pub fn outputs(&self) -> Result<Vec<SoilOutput>, SoilError> {
        let native_rate = self.native_rate.try_to_vec::<f64>()?;
        let native_depth = self.native_depth.try_to_vec::<f64>()?;
        let rate = self.si_rate.try_to_vec::<f64>()?;
        let depth = self.si_depth.try_to_vec::<f64>()?;
        self.next.context.check_projection_lengths(&[
            ("native infiltration rate (ft/s)", native_rate.len()),
            ("native interval depth (ft)", native_depth.len()),
            ("SI infiltration rate (m/s)", rate.len()),
            ("SI interval depth (m)", depth.len()),
        ])?;
        Ok(native_rate
            .into_iter()
            .zip(native_depth)
            .zip(rate)
            .zip(depth)
            .map(
                |(((native_rate_ft_per_s, native_depth_ft), rate_m_per_s), depth_m)| SoilOutput {
                    native_rate_ft_per_s,
                    native_depth_ft,
                    rate_m_per_s,
                    depth_m,
                },
            )
            .collect())
    }

    /// Accept the complete candidate state. This never modifies the original batch.
    pub fn into_next(self) -> ModifiedGreenAmptBatch<R> {
        self.next
    }
}

fn check_input(index: usize, rate: f64, depth: f64) -> Result<(), SoilError> {
    if !rate.is_finite() || rate < 0.0 || !depth.is_finite() || depth < 0.0 {
        return Err(SoilError::input(format!(
            "cell={index}, rate={rate}, depth={depth}: require nonnegative finite incoming rate and ponding"
        )).at_cell(index));
    }
    Ok(())
}
