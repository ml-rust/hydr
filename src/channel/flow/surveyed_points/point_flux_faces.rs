use super::{
    point_candidate_types::PointCandidate,
    point_flux_association,
    point_flux_masks::{assign, copy, equals},
    point_flux_types::PointFluxCertificates,
    point_quadrature_types::{PointQuadratureOutcome as Outcome, PointQuadratureProgress},
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

/// One quadrature continuation a face can demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Continuation {
    Star,
    Fan(SonicSide),
}

impl Continuation {
    pub(super) const ALL: [Self; 3] = [
        Self::Star,
        Self::Fan(SonicSide::Left),
        Self::Fan(SonicSide::Right),
    ];

    pub(super) fn index(self) -> usize {
        match self {
            Self::Star => 0,
            Self::Fan(side) => 1 + side.index(),
        }
    }

    /// The continuation one quadrature progress belongs to.
    pub(super) fn of<R: Runtime<DType = DType>>(progress: &PointQuadratureProgress<R>) -> Self {
        match progress.side {
            None => Self::Star,
            Some(side) => Self::Fan(side),
        }
    }
}

/// Outcomes after which a further advance cannot change a continuation.
const TERMINAL: [Outcome; 4] = [
    Outcome::LevelLimit,
    Outcome::WorkLimit,
    Outcome::NotApplicable,
    Outcome::Unresolved,
];

/// Face-order accumulators. A reselection overwrites its faces wholesale.
/// Arrays indexed by `Continuation::index` hold one tensor per continuation.
pub(super) struct Faces<R: Runtime<DType = DType>> {
    pub(super) owner: Tensor<R>,
    pub(super) association: Tensor<R>,
    pub(super) status: Tensor<R>,
    pub(super) physical_valid: Tensor<R>,
    pub(super) joint_valid: Tensor<R>,
    pub(super) physical_water: Interval<R>,
    pub(super) physical_momentum: Interval<R>,
    pub(super) joint_water: Interval<R>,
    pub(super) joint_momentum: Interval<R>,
    pub(super) selection: Tensor<R>,
    /// One where the owning selection holds a possible candidate of the continuation.
    pub(super) possible: [Tensor<R>; 3],
    /// Last outcome of each continuation. `NotApplicable` when never advanced.
    pub(super) outcome: [Tensor<R>; 3],
    /// Panel level of each continuation after its last advance. -1 when never advanced.
    pub(super) level: [Tensor<R>; 3],
    /// One when a further advance cannot change the continuation.
    pub(super) terminal: [Tensor<R>; 3],
    /// Last sonic outcome and level of either side.
    pub(super) sonic_outcome: Tensor<R>,
    pub(super) sonic_level: Tensor<R>,
    /// One when the continuation's depth bracket was offered the tightest tolerance.
    pub(super) tightened: [Tensor<R>; 3],
    pub(super) selections: Vec<PointFluxCertificates<R>>,
}

impl<R> Faces<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn new(
        a: &mut Arithmetic<'_, R>,
        count: usize,
        initial: PointFluxCertificates<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let owner = initial.candidates.invocation_owner.clone();
        let shape = owner.shape().to_vec();
        let association = point_flux_association::inverse(a, count, &owner)?;
        let possible = possible(a, count, &initial)?;
        let outcome = || {
            Tensor::full_scalar(
                &shape,
                DType::I64,
                Outcome::NotApplicable as i64 as f64,
                c.device(),
            )
        };
        let level = || Tensor::full_scalar(&shape, DType::I64, -1., c.device());
        let dtype = a.dtype();
        let zero = || Tensor::zeros(&shape, dtype, c.device());
        Ok(Self {
            association,
            status: initial.status.clone(),
            physical_valid: initial.physical_valid.clone(),
            joint_valid: initial.joint_valid.clone(),
            physical_water: copy(a, &initial.physical_water)?,
            physical_momentum: copy(a, &initial.physical_momentum)?,
            joint_water: copy(a, &initial.joint_water)?,
            joint_momentum: copy(a, &initial.joint_momentum)?,
            selection: Tensor::zeros(&shape, DType::I64, c.device())?,
            possible,
            outcome: [outcome()?, outcome()?, outcome()?],
            level: [level()?, level()?, level()?],
            terminal: [zero()?, zero()?, zero()?],
            sonic_outcome: outcome()?,
            sonic_level: level()?,
            tightened: [zero()?, zero()?, zero()?],
            selections: vec![initial],
            owner,
        })
    }

    pub(super) fn overwrite(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        count: usize,
        selected: PointFluxCertificates<R>,
    ) -> HydraulicResult<()> {
        let c = a.client();
        let rows = &selected.candidates.invocation_owner;
        let positions = point_flux_association::positions(a, &self.association, rows)?;
        let put = |target: &Tensor<R>, value: &Tensor<R>| c.index_put(target, 0, &positions, value);
        self.status = put(&self.status, &selected.status)?;
        self.physical_valid = put(&self.physical_valid, &selected.physical_valid)?;
        self.joint_valid = put(&self.joint_valid, &selected.joint_valid)?;
        let possible = possible(a, count, &selected)?;
        for (target, value) in self.possible.iter_mut().zip(&possible) {
            *target = put(target, value)?;
        }
        self.selection = put(
            &self.selection,
            &Tensor::full_scalar(
                rows.shape(),
                DType::I64,
                self.selections.len() as f64,
                c.device(),
            )?,
        )?;
        self.physical_water = assign(
            a,
            &self.physical_water,
            &positions,
            &selected.physical_water,
        )?;
        self.physical_momentum = assign(
            a,
            &self.physical_momentum,
            &positions,
            &selected.physical_momentum,
        )?;
        self.joint_water = assign(a, &self.joint_water, &positions, &selected.joint_water)?;
        self.joint_momentum = assign(
            a,
            &self.joint_momentum,
            &positions,
            &selected.joint_momentum,
        )?;
        self.selections.push(selected);
        Ok(())
    }

    /// Records one continuation advance and returns its count of advanced faces.
    pub(super) fn record(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        progress: &PointQuadratureProgress<R>,
    ) -> HydraulicResult<usize> {
        let c = a.client();
        let index = Continuation::of(progress).index();
        let positions =
            point_flux_association::positions(a, &self.association, &progress.invocation_owner)?;
        self.outcome[index] =
            c.index_put(&self.outcome[index], 0, &positions, &progress.outcome)?;
        self.level[index] = c.index_put(&self.level[index], 0, &positions, &progress.new_level)?;
        if progress.side.is_some() {
            self.sonic_outcome =
                c.index_put(&self.sonic_outcome, 0, &positions, &progress.outcome)?;
            self.sonic_level =
                c.index_put(&self.sonic_level, 0, &positions, &progress.new_level)?;
        }
        let mut terminal = Tensor::zeros(progress.outcome.shape(), a.dtype(), c.device())?;
        for outcome in TERMINAL {
            terminal = c.maximum(&terminal, &equals(a, &progress.outcome, outcome as i64)?)?;
        }
        let previous = c.index_select(&self.terminal[index], 0, &positions)?;
        self.terminal[index] = c.index_put(
            &self.terminal[index],
            0,
            &positions,
            &c.maximum(&previous, &terminal)?,
        )?;
        let advanced = equals(a, &progress.outcome, Outcome::Advanced as i64)?;
        Ok(c.nonzero(&advanced)?.numel())
    }

    /// Records one depth tightening and returns its count of narrowed faces.
    pub(super) fn record_tightening(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        progress: &PointQuadratureProgress<R>,
    ) -> HydraulicResult<usize> {
        let c = a.client();
        let index = Continuation::of(progress).index();
        let owners = &progress.invocation_owner;
        let positions = point_flux_association::positions(a, &self.association, owners)?;
        self.tightened[index] = c.index_put(
            &self.tightened[index],
            0,
            &positions,
            &Tensor::ones(owners.shape(), a.dtype(), c.device())?,
        )?;
        let advanced = equals(a, &progress.outcome, Outcome::Advanced as i64)?;
        let narrowed = c.mul(&advanced, &progress.bracket_improved)?;
        Ok(c.nonzero(&narrowed)?.numel())
    }
}

/// Possible-candidate masks per continuation, in the order of the selection's rows.
/// Every star candidate counts, and a fan counts on its active rows.
fn possible<R>(
    a: &mut Arithmetic<'_, R>,
    count: usize,
    selected: &PointFluxCertificates<R>,
) -> HydraulicResult<[Tensor<R>; 3]>
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
    let rows = &selected.candidates.invocation_owner;
    let local = point_flux_association::inverse(a, count, rows)?;
    let zero = || Tensor::zeros(rows.shape(), a.dtype(), c.device());
    let mut masks = [zero()?, zero()?, zero()?];
    for candidate in &selected.candidates.candidates {
        let (continuation, owners) = match candidate {
            PointCandidate::Star(star) => (Continuation::Star, star.invocation_owner.clone()),
            PointCandidate::Fan(fan) => {
                let certificate = &fan.certificate;
                let active = c.nonzero(&fan.active)?.flatten()?;
                (
                    Continuation::Fan(certificate.side),
                    c.index_select(&certificate.invocation_owner, 0, &active)?,
                )
            }
            _ => continue,
        };
        if owners.numel() == 0 {
            continue;
        }
        let positions = point_flux_association::positions(a, &local, &owners)?;
        let index = continuation.index();
        masks[index] = c.index_put(
            &masks[index],
            0,
            &positions,
            &Tensor::ones(owners.shape(), a.dtype(), c.device())?,
        )?;
    }
    Ok(masks)
}
