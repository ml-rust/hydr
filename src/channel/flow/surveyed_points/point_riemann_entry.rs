use super::{
    point_flux_association,
    point_flux_types::{PointFluxKind, ResolvedPointFluxes},
    point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
    point_riemann_rejection,
    point_root_session::{PointRootSession, invalid},
    point_sources::PointSourceParts,
    source_sonic_types::SonicSide,
};
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig,
        error::{Arithmetic, HydraulicResult, checked},
        work::WorkLedger,
    },
    section::PrismaticSectionBatch,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::{CertifiedExpansion, Interval},
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

/// Resolves every face from four exact conserved points, or rejects the whole call.
pub(crate) fn solve<R>(
    section: &PrismaticSectionBatch<R>,
    owners: &Tensor<R>,
    inputs: [&CertifiedExpansion<R>; 4],
    config: &PrismaticRiemannConfig,
    work: &WorkLedger,
) -> HydraulicResult<PrismaticPointRiemannResult<R>>
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
    config.check()?;
    let parts = PointSourceParts::capture(section, owners, inputs, config)?;
    let c = parts.geometry.client();
    let count = parts.count();
    let fluxes = if count == 0 {
        checked::<R, _>(c, "point Riemann empty faces", Faces::empty)?
    } else {
        let rows = parts.sequential_rows()?;
        let mut pair = parts.pair(config, work, &rows)?;
        let mut session = PointRootSession::new(&mut pair, &rows, config.root.max_points)?;
        let resolved = session.resolve_point_fluxes(&rows)?;
        point_riemann_rejection::check(&session, &resolved)?;
        checked::<R, _>(c, "point Riemann selected traces", |a| {
            Faces::scatter(a, count, &resolved)
        })?
    };
    let input = parts.velocity.input();
    checked::<R, _>(c, "point Riemann result", |a| {
        let copy = |a: &mut Arithmetic<'_, R>, value: &CertifiedExpansion<R>| {
            a.expansion_bounds(value.high(), value.low(), value.abs_error())
        };
        let owners = Tensor::empty(input.owners.shape(), DType::I64, c.device())?;
        c.copy_into(&owners, &input.owners)?;
        Ok(PrismaticPointRiemannResult {
            owners,
            left_area: copy(a, &input.left_area)?,
            left_discharge: copy(a, &input.left_discharge)?,
            right_area: copy(a, &input.right_area)?,
            right_discharge: copy(a, &input.right_discharge)?,
            water_flux: fluxes.water,
            momentum_flux: fluxes.momentum,
            trace_area: fluxes.area,
            trace_discharge: fluxes.discharge,
            trace_kind: fluxes.kind,
            physical_water_flux: fluxes.physical_water,
            physical_momentum_flux: fluxes.physical_momentum,
            water_flux_bound: fluxes.joint_water,
            momentum_flux_bound: fluxes.joint_momentum,
            wave_speed: fluxes.wave_speed,
            work: work.snapshot(),
        })
    })
}

/// Query-order fields of the selected trace and the face bounds.
struct Faces<R: Runtime<DType = DType>> {
    water: CertifiedExpansion<R>,
    momentum: CertifiedExpansion<R>,
    area: CertifiedExpansion<R>,
    discharge: CertifiedExpansion<R>,
    kind: Tensor<R>,
    physical_water: Interval<R>,
    physical_momentum: Interval<R>,
    joint_water: Interval<R>,
    joint_momentum: Interval<R>,
    wave_speed: Interval<R>,
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
    fn empty(a: &mut Arithmetic<'_, R>) -> HydraulicResult<Self> {
        let c = a.client();
        let zero = Tensor::zeros(&[0], DType::F64, c.device())?;
        let point = |a: &mut Arithmetic<'_, R>| a.expansion_bounds(&zero, &zero, &zero);
        let interval = |a: &mut Arithmetic<'_, R>| a.bounds(&zero, &zero);
        Ok(Self {
            water: point(a)?,
            momentum: point(a)?,
            area: point(a)?,
            discharge: point(a)?,
            kind: Tensor::zeros(&[0], DType::I64, c.device())?,
            physical_water: interval(a)?,
            physical_momentum: interval(a)?,
            joint_water: interval(a)?,
            joint_momentum: interval(a)?,
            wave_speed: interval(a)?,
        })
    }

    /// Writes each face's owning selected point at its invocation owner.
    fn scatter(
        a: &mut Arithmetic<'_, R>,
        count: usize,
        resolved: &ResolvedPointFluxes<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let association = point_flux_association::inverse(a, count, &resolved.invocation_owner)?;
        let zeros = || Tensor::zeros(&[count], DType::F64, c.device());
        // Components of the water, momentum, trace area and trace discharge points.
        let mut points = Vec::with_capacity(4);
        for _ in 0..4 {
            points.push([zeros()?, zeros()?, zeros()?]);
        }
        let mut kind = Tensor::full_scalar(&[count], DType::I64, -1., c.device())?;
        let mut filled = zeros()?;
        for (index, selection) in resolved.selections.iter().enumerate() {
            let owning = Tensor::full_scalar(&[], DType::I64, index as f64, c.device())?;
            for point in &selection.selected {
                let positions =
                    point_flux_association::positions(a, &association, &point.invocation_owner)?;
                let owned = c.eq(
                    &c.index_select(&resolved.selection, 0, &positions)?,
                    &owning,
                )?;
                let rows = c.nonzero(&owned)?.flatten()?;
                if rows.numel() == 0 {
                    continue;
                }
                let target = c.index_select(&point.invocation_owner, 0, &rows)?;
                let values = [
                    &point.discharge,
                    &point.momentum,
                    &point.area,
                    &point.discharge,
                ];
                for (components, value) in points.iter_mut().zip(values) {
                    for (component, source) in
                        components
                            .iter_mut()
                            .zip([value.high(), value.low(), value.abs_error()])
                    {
                        let selected = c.index_select(source, 0, &rows)?;
                        *component = c.index_put(component, 0, &target, &selected)?;
                    }
                }
                let code = Tensor::full_scalar(
                    target.shape(),
                    DType::I64,
                    trace_code(point.kind) as i64 as f64,
                    c.device(),
                )?;
                kind = c.index_put(&kind, 0, &target, &code)?;
                let ones = Tensor::ones(target.shape(), DType::F64, c.device())?;
                filled = c.index_put(&filled, 0, &target, &ones)?;
            }
        }
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        if c.nonzero(&c.eq(&filled, &zero)?)?.numel() != 0 {
            return Err(invalid("accepted face lacks its owning selected trace"));
        }
        let owner = &resolved.invocation_owner;
        let interval = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| {
            let lower = c.index_put(&zeros()?, 0, owner, value.lower())?;
            let upper = c.index_put(&zeros()?, 0, owner, value.upper())?;
            HydraulicResult::Ok(a.bounds(&lower, &upper)?)
        };
        let physical_water = interval(a, &resolved.physical_water)?;
        let physical_momentum = interval(a, &resolved.physical_momentum)?;
        let joint_water = interval(a, &resolved.joint_water)?;
        let joint_momentum = interval(a, &resolved.joint_momentum)?;
        let wave_speed = interval(a, &resolved.wave_speed_m_per_s)?;
        let mut expansions = Vec::with_capacity(4);
        for components in &points {
            expansions.push(a.expansion_bounds(&components[0], &components[1], &components[2])?);
        }
        let [water, momentum, area, discharge]: [CertifiedExpansion<R>; 4] = expansions
            .try_into()
            .map_err(|_| invalid("selected trace expansion count differs from four"))?;
        Ok(Self {
            water,
            momentum,
            area,
            discharge,
            kind,
            physical_water,
            physical_momentum,
            joint_water,
            joint_momentum,
            wave_speed,
        })
    }
}

fn trace_code(kind: PointFluxKind) -> PrismaticPointTraceKind {
    match kind {
        PointFluxKind::Exterior(SonicSide::Left) => PrismaticPointTraceKind::LeftState,
        PointFluxKind::Fan(SonicSide::Left) => PrismaticPointTraceKind::LeftFan,
        PointFluxKind::Star => PrismaticPointTraceKind::Star,
        PointFluxKind::Fan(SonicSide::Right) => PrismaticPointTraceKind::RightFan,
        PointFluxKind::Exterior(SonicSide::Right) => PrismaticPointTraceKind::RightState,
        PointFluxKind::Dry => PrismaticPointTraceKind::Dry,
    }
}

#[cfg(test)]
#[path = "point_riemann_entry_tests.rs"]
mod tests;
