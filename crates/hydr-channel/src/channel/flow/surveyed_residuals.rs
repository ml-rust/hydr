//! Expanded cell water and momentum residuals of one surveyed interval.

use super::{
    ChannelError,
    conserved_point::ConservedPoint,
    conserved_update::{ExpansionClient, exact, expanded, magnitude, transfer},
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Both parts of the face points and the friction change of one surveyed interval.
pub(super) struct ExpandedImpulses<'a, R: Runtime<DType = DType>> {
    pub face_volume_low: &'a Tensor<R>,
    pub face_momentum_low: &'a Tensor<R>,
    /// Low part of the friction impulse.
    pub friction_low: &'a Tensor<R>,
    /// Per-cell radius discarded by the water state selection.
    pub selection_bound: &'a Tensor<R>,
}

/// Exact cell states and shared face points that enter the expanded residuals.
pub(super) struct ResidualInputs<'a, R: Runtime<DType = DType>> {
    pub volume: &'a ConservedPoint<R>,
    pub previous_volume: &'a ConservedPoint<R>,
    pub momentum: &'a ConservedPoint<R>,
    pub previous_momentum: &'a ConservedPoint<R>,
    pub face_volume: &'a Tensor<R>,
    pub face_momentum: &'a Tensor<R>,
    /// High part of the friction impulse.
    pub friction: &'a Tensor<R>,
    pub expanded: &'a ExpandedImpulses<'a, R>,
    /// Lateral exchange, when it is active.
    pub lateral: Option<LateralImpulses<'a, R>>,
}

/// Lateral water and momentum that enter one surveyed interval.
pub(super) struct LateralImpulses<'a, R: Runtime<DType = DType>> {
    pub incoming_volume: &'a Tensor<R>,
    pub withdrawal_volume: &'a Tensor<R>,
    pub incoming_momentum: &'a Tensor<R>,
    /// Two-part momentum the withdrawal removes, as applied to the state.
    pub withdrawn_momentum: &'a ConservedPoint<R>,
}

/// Upper bounds on the magnitude of the cell water and momentum residuals.
///
/// Each residual is the state change minus the selected face transfers, friction, and lateral
/// exchange, evaluated as expansions. The momentum withdrawal is the applied two-part value.
/// Its distance from the receipt momentum is a lateral assembly bound.
pub(super) fn expanded_residuals<R>(
    client: &R::Client,
    count: usize,
    input: &ResidualInputs<'_, R>,
) -> Result<(Tensor<R>, Tensor<R>), ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    expanded(client, "surveyed local water and momentum residuals", |a| {
        let volume = exact(a, &input.volume.high, &input.volume.low)?;
        let previous = exact(a, &input.previous_volume.high, &input.previous_volume.low)?;
        let change = a.expansion_sub(&volume, &previous)?;
        let faces = exact(a, input.face_volume, input.expanded.face_volume_low)?;
        let transfer_water = transfer(a, &faces, count)?;
        let mut water = a.expansion_sub(&change, &transfer_water)?;

        let momentum = exact(a, &input.momentum.high, &input.momentum.low)?;
        let previous = exact(
            a,
            &input.previous_momentum.high,
            &input.previous_momentum.low,
        )?;
        let change = a.expansion_sub(&momentum, &previous)?;
        let faces = exact(a, input.face_momentum, input.expanded.face_momentum_low)?;
        let transfer_momentum = transfer(a, &faces, count)?;
        let friction = exact(a, input.friction, input.expanded.friction_low)?;
        let change = a.expansion_sub(&change, &transfer_momentum)?;
        let mut momentum = a.expansion_sub(&change, &friction)?;

        if let Some(lateral) = &input.lateral {
            let incoming = a.expansion_point(lateral.incoming_volume)?;
            let withdrawal = a.expansion_point(lateral.withdrawal_volume)?;
            let exchange = a.expansion_sub(&incoming, &withdrawal)?;
            water = a.expansion_sub(&water, &exchange)?;
            let incoming = a.expansion_point(lateral.incoming_momentum)?;
            let withdrawn = &lateral.withdrawn_momentum;
            let withdrawal = exact(a, &withdrawn.high, &withdrawn.low)?;
            let lateral = a.expansion_sub(&incoming, &withdrawal)?;
            momentum = a.expansion_sub(&momentum, &lateral)?;
        }
        Ok((magnitude(a, &water)?, magnitude(a, &momentum)?))
    })
}
