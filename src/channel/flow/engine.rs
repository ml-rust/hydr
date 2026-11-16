use super::{
    ChannelCellResult, ChannelConfig, ChannelError, ChannelInitialState, ChannelLateralLedger,
    ChannelReachDefinition, ChannelStepDiagnostics, ChannelWaterLedger,
    conserved_point::ConservedPoint, context::Context, receipt::LastInterval,
    sources::LateralState,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use std::sync::Arc;

/// Accepted one-dimensional channel state on the caller's runtime.
#[derive(Debug)]
pub struct ChannelReach<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) accepted: Accepted<R>,
}

#[derive(Clone, Debug)]
pub(super) struct Accepted<R: Runtime<DType = DType>> {
    pub volume: ConservedPoint<R>,
    pub momentum: ConservedPoint<R>,
    pub lateral: LateralState<R>,
    pub time: f64,
    pub ledger: ChannelWaterLedger,
    pub steps: usize,
    pub last: Option<Arc<LastInterval<R>>>,
}

impl<R: Runtime<DType = DType>> Clone for ChannelReach<R> {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            accepted: self.accepted.clone(),
        }
    }
}

/// One accepted interval, including its complete next continuation state.
#[derive(Clone, Debug)]
pub struct ChannelStep<R: Runtime<DType = DType>> {
    /// Candidate accepted without modifying its input reach.
    pub next: ChannelReach<R>,
}

impl<R: Runtime<DType = DType>> ChannelReach<R> {
    /// Immutable ordered geometry and numerical controls.
    pub fn definition(&self) -> &ChannelReachDefinition {
        &self.context.definition
    }
    /// Numerical controls used for this reach.
    pub fn config(&self) -> &ChannelConfig {
        &self.context.definition.config
    }
    /// Accepted elapsed time, in seconds.
    pub fn time(&self) -> f64 {
        self.accepted.time
    }
    /// Number of accepted intervals since construction.
    pub fn accepted_steps(&self) -> usize {
        self.accepted.steps
    }
    /// Cumulative water accounting from the original run origin.
    pub fn ledger(&self) -> &ChannelWaterLedger {
        &self.accepted.ledger
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    /// Construct explicit channel cells and conservative state without a default runtime.
    pub fn new(
        client: R::Client,
        definition: ChannelReachDefinition,
        initial: ChannelInitialState,
    ) -> Result<Self, ChannelError>
    where
        R::Client: HypotOps<R> + TensorOps<R> + NextRepresentableOps<R>,
    {
        let context = Arc::new(Context::new(client, definition, initial)?);
        Self::from_context(Arc::clone(&context), &context.original_initial)
    }

    pub(super) fn from_context(
        context: Arc<Context<R>>,
        initial: &ChannelInitialState,
    ) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R> + NextRepresentableOps<R>,
    {
        let zeros = vec![0.0; context.count()];
        Self::from_parts(context, initial, &zeros, &zeros)
    }

    /// Build accepted state from the initial high parts and explicit low parts.
    pub(super) fn from_parts(
        context: Arc<Context<R>>,
        initial: &ChannelInitialState,
        volume_low: &[f64],
        momentum_low: &[f64],
    ) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R> + NextRepresentableOps<R>,
    {
        if !initial.time_s.is_finite()
            || initial.volume_m3.len() != context.count()
            || initial.longitudinal_momentum_m4_per_s.len() != context.count()
            || volume_low.len() != context.count()
            || momentum_low.len() != context.count()
        {
            return Err(ChannelError::input(format!(
                "initial time={}, volume count={}, momentum count={}, cells={}: supply finite time and one value per cell",
                initial.time_s,
                initial.volume_m3.len(),
                initial.longitudinal_momentum_m4_per_s.len(),
                context.count()
            )));
        }
        let volume =
            ConservedPoint::from_parts(&context, "volume", &initial.volume_m3, volume_low)?;
        let momentum = ConservedPoint::from_parts(
            &context,
            "momentum",
            &initial.longitudinal_momentum_m4_per_s,
            momentum_low,
        )?;
        context.fields(&volume.high, &momentum.high)?;
        let storage = context.total(&volume.high)?;
        let time = initial.time_s;
        let lateral = context.lateral.initial_state(&context.client)?;
        Ok(Self {
            context,
            accepted: Accepted {
                volume,
                momentum,
                lateral,
                time,
                ledger: ChannelWaterLedger {
                    initial_storage_m3: storage,
                    storage_m3: storage,
                    boundary_in_m3: 0.0,
                    boundary_out_m3: 0.0,
                    lateral_in_m3: 0.0,
                    lateral_out_m3: 0.0,
                    lateral_counter_abs_bound_m3: 0.0,
                    lateral_assembly_abs_bound_m3: 0.0,
                    residual_m3: 0.0,
                    numerical_transfer_underflow_bound_m3: 0.0,
                },
                steps: 0,
                last: None,
            },
        })
    }

    /// Copy accepted hydraulic results from the selected device.
    pub fn state(&self) -> Result<Vec<ChannelCellResult>, ChannelError>
    where
        R::Client: TensorOps<R> + NextRepresentableOps<R>,
    {
        let fields = self
            .context
            .fields(&self.accepted.volume.high, &self.accepted.momentum.high)?;
        let volume = self.accepted.volume.high.try_to_vec::<f64>()?;
        let momentum = self.accepted.momentum.high.try_to_vec::<f64>()?;
        let depth = fields.depth.try_to_vec::<f64>()?;
        let velocity = fields.velocity.try_to_vec::<f64>()?;
        let discharge = fields.discharge.try_to_vec::<f64>()?;
        Ok((0..self.context.count())
            .map(|i| {
                let wet = volume[i] > 0.0;
                ChannelCellResult {
                    volume_m3: volume[i],
                    longitudinal_momentum_m4_per_s: momentum[i],
                    depth_m: depth[i],
                    stage_m: wet
                        .then_some(self.definition().geometry.cell(i).bed_elevation_m + depth[i]),
                    velocity_m_per_s: wet.then_some(velocity[i]),
                    discharge_m3_per_s: discharge[i],
                    dry: !wet,
                }
            })
            .collect())
    }

    /// Copy per-binding cumulative lateral volumes and their rounding bounds.
    pub fn lateral_ledgers(&self) -> Result<Vec<ChannelLateralLedger>, ChannelError> {
        self.context
            .lateral
            .project_ledgers(&self.context.client, &self.accepted.lateral)
    }

    /// Copy the bounded final interval receipt from the selected device.
    pub fn last_step(&self) -> Result<Option<ChannelStepDiagnostics>, ChannelError> {
        self.accepted
            .last
            .as_ref()
            .map(|last| last.project(&self.context.client, &self.context.lateral))
            .transpose()
    }
}
