use super::{
    point_flux_association,
    point_flux_types::{
        PointFluxCertificates, PointFluxKind, PointFluxStatus, ResolvedPointFluxes,
    },
    point_root_session::{PointRootSession, invalid},
    point_wave_envelope,
    source_sonic_quadrature_types::{
        SourceSonicQuadratureOutcome as Outcome, SourceSonicQuadratureProgress,
    },
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::Interval,
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

const SIDES: [SonicSide; 2] = [SonicSide::Left, SonicSide::Right];

impl<R> PointRootSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    /// Selects every face, then refines sonic fans until no face exceeds its flux budget.
    /// Each round advances each fan side once and reselects the demanded faces.
    /// The rounds stop when no face advances, and never exceed `max_refinements`.
    /// Each face then takes the wave-speed envelope of its owning selection.
    pub(super) fn resolve_point_fluxes(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<ResolvedPointFluxes<R>> {
        let work = self.sources.evaluation.work;
        let geometry = self.sources.geometry;
        let c = geometry.client();
        let points_before = self.callback_points();
        let residual_before = work.snapshot().residual_points;
        let initial = self.select_point_fluxes(owners)?;
        let count = self.invocation_rows.numel();
        let mut faces = checked::<R, _>(c, "point flux controller face state", |a| {
            Faces::new(a, count, initial)
        })?;
        let blocked = checked::<R, _>(c, "point flux controller positive-area faces", |a| {
            let pending = equals(
                a,
                &faces.status,
                PointFluxStatus::PendingPositiveArea as i64,
            )?;
            Ok(c.index_select(&faces.owner, 0, &c.nonzero(&pending)?.flatten()?)?)
        })?;
        let mut positive_support = None;
        if blocked.numel() != 0 {
            let support = self.advance_point_flux_positive_support(&blocked)?;
            self.invocation
                .check(&support.candidates.invocation_witness)?;
            let reselected = self.select_point_fluxes(&blocked)?;
            checked::<R, _>(c, "point flux controller positive-area reselection", |a| {
                faces.overwrite(a, count, reselected)
            })?;
            positive_support = Some(support);
        }
        let mut rounds = 0;
        for round in 1..=self.sources.evaluation.config.max_refinements {
            let (demand, sides) =
                checked::<R, _>(c, "point flux controller fan demand", |a| faces.demand(a))?;
            if demand.numel() == 0 {
                break;
            }
            let mut advanced = 0usize;
            for side in SIDES {
                let owners = &sides[side.index()];
                if owners.numel() == 0 {
                    continue;
                }
                let progress = self.advance_source_sonic_quadrature(owners, side)?;
                let moved = checked::<R, _>(c, "point flux controller sonic outcomes", |a| {
                    faces.record(a, &progress)
                })?;
                advanced = advanced.checked_add(moved).ok_or_else(|| {
                    invalid("point flux controller advance count overflows usize")
                })?;
            }
            if advanced == 0 {
                break;
            }
            work.begin_refinement(self.max_panels()?, round)?;
            rounds = round;
            let reselected = self.select_point_fluxes(&demand)?;
            checked::<R, _>(c, "point flux controller fan reselection", |a| {
                faces.overwrite(a, count, reselected)
            })?;
        }
        let (envelope, status) = checked::<R, _>(c, "point flux controller final status", |a| {
            let status = faces.final_status(a)?;
            point_wave_envelope::resolve(a, count, &faces.owner, &faces.selections, &status)
        })?;
        let executed = work
            .snapshot()
            .residual_points
            .checked_sub(residual_before)
            .ok_or_else(|| invalid("point flux controller residual work decreased"))?;
        let recorded = self
            .callback_points()
            .checked_sub(points_before)
            .ok_or_else(|| invalid("point flux controller callback count decreased"))?;
        if executed != recorded {
            return Err(invalid(
                "point flux controller callback points differ from executed residual work",
            ));
        }
        Ok(ResolvedPointFluxes {
            invocation_owner: faces.owner,
            status,
            physical_valid: faces.physical_valid,
            joint_valid: faces.joint_valid,
            physical_water: faces.physical_water,
            physical_momentum: faces.physical_momentum,
            joint_water: faces.joint_water,
            joint_momentum: faces.joint_momentum,
            selection: faces.selection,
            selections: faces.selections,
            sonic_outcome: faces.outcome,
            sonic_level: faces.level,
            wave_speed_m_per_s: envelope.wave_speed_m_per_s,
            wave_valid: envelope.valid,
            positive_support,
            rounds,
            callback_points: self.callback_points(),
            point_limit: self.point_limit(),
        })
    }

    /// Largest Simpson panel count among the cached source levels.
    fn max_panels(&self) -> HydraulicResult<usize> {
        match self
            .sources
            .panel_cache
            .iter()
            .map(|(index, _)| *index)
            .max()
        {
            Some(index) => Ok(self.sources.panel_level(index)?.panels()),
            None => Ok(self.sources.evaluation.panels),
        }
    }
}

/// Face-order accumulators. A reselection overwrites its faces wholesale.
struct Faces<R: Runtime<DType = DType>> {
    owner: Tensor<R>,
    association: Tensor<R>,
    status: Tensor<R>,
    physical_valid: Tensor<R>,
    joint_valid: Tensor<R>,
    physical_water: Interval<R>,
    physical_momentum: Interval<R>,
    joint_water: Interval<R>,
    joint_momentum: Interval<R>,
    selection: Tensor<R>,
    /// Side index of the selected fan, or -1 when the selected point is not a fan.
    fan: Tensor<R>,
    outcome: Tensor<R>,
    level: Tensor<R>,
    /// One when a further advance cannot change the face.
    settled: Tensor<R>,
    selections: Vec<PointFluxCertificates<R>>,
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
    fn new(
        a: &mut Arithmetic<'_, R>,
        count: usize,
        initial: PointFluxCertificates<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let owner = initial.candidates.invocation_owner.clone();
        let shape = owner.shape().to_vec();
        let association = point_flux_association::inverse(a, count, &owner)?;
        let fan = fan_codes(a, count, &initial)?;
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
            fan,
            outcome: Tensor::full_scalar(
                &shape,
                DType::I64,
                Outcome::NotApplicable as i64 as f64,
                c.device(),
            )?,
            level: Tensor::full_scalar(&shape, DType::I64, -1., c.device())?,
            settled: Tensor::zeros(&shape, a.dtype(), c.device())?,
            selections: vec![initial],
            owner,
        })
    }

    fn overwrite(
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
        self.fan = put(&self.fan, &fan_codes(a, count, &selected)?)?;
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

    /// Over-budget fan faces that can still advance, all together and split by side.
    fn demand(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<(Tensor<R>, [Tensor<R>; 2])> {
        let c = a.client();
        let budget = equals(a, &self.status, PointFluxStatus::FluxBudgetExceeded as i64)?;
        let open = flag(
            a,
            &c.eq(&self.settled, &Tensor::zeros(&[], a.dtype(), c.device())?)?,
        )?;
        let demand = c.mul(&budget, &open)?;
        let empty = Tensor::empty(&[0], DType::I64, c.device())?;
        let mut sides = [empty.clone(), empty];
        let mut any = Tensor::zeros(demand.shape(), a.dtype(), c.device())?;
        for side in SIDES {
            let mask = c.mul(&demand, &equals(a, &self.fan, side.index() as i64)?)?;
            any = c.maximum(&any, &mask)?;
            let rows = c.nonzero(&mask)?.flatten()?;
            sides[side.index()] = c.index_select(&self.owner, 0, &rows)?;
        }
        let all = c.index_select(&self.owner, 0, &c.nonzero(&any)?.flatten()?)?;
        Ok((all, sides))
    }

    /// Records one advance and returns its count of advanced faces.
    fn record(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        progress: &SourceSonicQuadratureProgress<R>,
    ) -> HydraulicResult<usize> {
        let c = a.client();
        let positions =
            point_flux_association::positions(a, &self.association, &progress.invocation_owner)?;
        self.outcome = c.index_put(&self.outcome, 0, &positions, &progress.outcome)?;
        self.level = c.index_put(&self.level, 0, &positions, &progress.new_level)?;
        let mut terminal = Tensor::zeros(progress.outcome.shape(), a.dtype(), c.device())?;
        for outcome in [
            Outcome::LevelLimit,
            Outcome::WorkLimit,
            Outcome::NotApplicable,
        ] {
            terminal = c.maximum(&terminal, &equals(a, &progress.outcome, outcome as i64)?)?;
        }
        let previous = c.index_select(&self.settled, 0, &positions)?;
        self.settled = c.index_put(
            &self.settled,
            0,
            &positions,
            &c.maximum(&previous, &terminal)?,
        )?;
        let advanced = equals(a, &progress.outcome, Outcome::Advanced as i64)?;
        Ok(c.nonzero(&advanced)?.numel())
    }

    /// Maps each face still over budget to the reason its continuation stopped.
    fn final_status(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Tensor<R>> {
        let c = a.client();
        let shape = self.status.shape();
        let full = |value: PointFluxStatus| {
            Tensor::full_scalar(shape, DType::I64, value as i64 as f64, c.device())
        };
        let exceeded = full(PointFluxStatus::FluxBudgetExceeded)?;
        let reason = c.where_cond(
            &c.eq(&self.outcome, &code(a, Outcome::WorkLimit as i64)?)?,
            &full(PointFluxStatus::WorkLimit)?,
            &c.where_cond(
                &c.eq(&self.outcome, &code(a, Outcome::Unresolved as i64)?)?,
                &full(PointFluxStatus::RootBracketUnresolved)?,
                &exceeded,
            )?,
        )?;
        let reason = c.where_cond(&c.lt(&self.fan, &code(a, 0)?)?, &exceeded, &reason)?;
        Ok(c.where_cond(
            &c.eq(
                &self.status,
                &code(a, PointFluxStatus::FluxBudgetExceeded as i64)?,
            )?,
            &reason,
            &self.status,
        )?)
    }
}

/// Selected fan side index per selection row, or -1.
fn fan_codes<R>(
    a: &mut Arithmetic<'_, R>,
    count: usize,
    selected: &PointFluxCertificates<R>,
) -> HydraulicResult<Tensor<R>>
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
    let mut codes = Tensor::full_scalar(rows.shape(), DType::I64, -1., c.device())?;
    for point in &selected.selected {
        let PointFluxKind::Fan(side) = point.kind else {
            continue;
        };
        let positions = point_flux_association::positions(a, &local, &point.invocation_owner)?;
        let value = Tensor::full_scalar(
            positions.shape(),
            DType::I64,
            side.index() as f64,
            c.device(),
        )?;
        codes = c.index_put(&codes, 0, &positions, &value)?;
    }
    Ok(codes)
}

fn code<R>(a: &Arithmetic<'_, R>, value: i64) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(Tensor::full_scalar(
        &[],
        DType::I64,
        value as f64,
        a.client().device(),
    )?)
}

/// Numeric zero-one mask of the I64 entries equal to `value`.
pub(super) fn equals<R>(
    a: &Arithmetic<'_, R>,
    values: &Tensor<R>,
    value: i64,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    flag(a, &a.client().eq(values, &code(a, value)?)?)
}

/// Converts a comparison result to a numeric zero-one mask.
pub(super) fn flag<R>(a: &Arithmetic<'_, R>, condition: &Tensor<R>) -> HydraulicResult<Tensor<R>>
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
    Ok(c.where_cond(
        condition,
        &Tensor::ones(condition.shape(), a.dtype(), c.device())?,
        &Tensor::zeros(condition.shape(), a.dtype(), c.device())?,
    )?)
}

fn copy<R>(a: &mut Arithmetic<'_, R>, value: &Interval<R>) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.bounds(value.lower(), value.upper())?)
}

fn assign<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    positions: &Tensor<R>,
    value: &Interval<R>,
) -> HydraulicResult<Interval<R>>
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
    let lower = c.index_put(target.lower(), 0, positions, value.lower())?;
    let upper = c.index_put(target.upper(), 0, positions, value.upper())?;
    Ok(a.bounds(&lower, &upper)?)
}
