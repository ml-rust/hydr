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
    /// Momentum high part before friction.
    pub friction_from: &'a Tensor<R>,
    /// Momentum high part after friction.
    pub friction_to: &'a Tensor<R>,
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
    pub expanded: &'a ExpandedImpulses<'a, R>,
    /// Incoming and withdrawn water and momentum, when lateral exchange is active.
    pub lateral: Option<[&'a Tensor<R>; 4]>,
}

/// Upper bounds on the magnitude of the cell water and momentum residuals.
///
/// Each residual is the state change minus the selected face transfers, friction, and lateral
/// exchange, evaluated as expansions.
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
        let to = a.expansion_point(input.expanded.friction_to)?;
        let from = a.expansion_point(input.expanded.friction_from)?;
        let friction = a.expansion_sub(&to, &from)?;
        let change = a.expansion_sub(&change, &transfer_momentum)?;
        let mut momentum = a.expansion_sub(&change, &friction)?;

        if let Some([incoming, withdrawal, incoming_momentum, withdrawal_momentum]) = input.lateral
        {
            let incoming = a.expansion_point(incoming)?;
            let withdrawal = a.expansion_point(withdrawal)?;
            let lateral = a.expansion_sub(&incoming, &withdrawal)?;
            water = a.expansion_sub(&water, &lateral)?;
            let incoming = a.expansion_point(incoming_momentum)?;
            let withdrawal = a.expansion_point(withdrawal_momentum)?;
            let lateral = a.expansion_sub(&incoming, &withdrawal)?;
            momentum = a.expansion_sub(&momentum, &lateral)?;
        }
        Ok((magnitude(a, &water)?, magnitude(a, &momentum)?))
    })
}
