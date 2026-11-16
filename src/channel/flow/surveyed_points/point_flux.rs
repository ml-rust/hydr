use super::{
    point_candidate_types::PointCandidateStatus,
    point_flux_assembly::{Assembly, hull},
    point_flux_association, point_flux_operands,
    point_flux_types::{PointFluxCertificates, PointFluxKind, PointFluxStatus},
    point_root_session::PointRootSession,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
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
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

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
    pub(super) fn select_point_fluxes(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<PointFluxCertificates<R>> {
        let candidates = self.certify_point_candidates(owners)?;
        self.invocation.check(&candidates.invocation_witness)?;
        let c = self.sources.geometry.client();
        let (mut physical, mut joint, mut chosen, mut pending, complete, association) =
            checked::<R, _>(c, "point flux coverage initialization", |a| {
                let count = candidates.invocation_owner.numel();
                let zero = Tensor::zeros(&[count], a.dtype(), c.device())?;
                let complete = c.where_cond(
                    &c.eq(
                        &candidates.status,
                        &Tensor::full_scalar(
                            &[],
                            DType::I64,
                            PointCandidateStatus::Complete as i64 as f64,
                            c.device(),
                        )?,
                    )?,
                    &Tensor::ones(candidates.status.shape(), a.dtype(), c.device())?,
                    &Tensor::zeros(candidates.status.shape(), a.dtype(), c.device())?,
                )?;
                let association = point_flux_association::inverse(
                    a,
                    self.invocation_rows.numel(),
                    &candidates.invocation_owner,
                )?;
                Ok((
                    Assembly::new(a, count)?,
                    Assembly::new(a, count)?,
                    zero.clone(),
                    zero,
                    complete,
                    association,
                ))
            })?;
        let mut selected = Vec::new();
        for index in 0..candidates.candidates.len() {
            let operand = checked::<R, _>(c, "point flux complete candidate operands", |a| {
                point_flux_operands::candidate(a, &candidates, index)
            })?;
            let Some(operand) = operand else {
                continue;
            };
            let (safe_rows, select_rows) =
                checked::<R, _>(c, "point flux positive area and coherent selection", |a| {
                    let positions =
                        point_flux_association::positions(a, &association, &operand.owners)?;
                    let zero = Tensor::zeros(operand.owners.shape(), a.dtype(), c.device())?;
                    let safe = if operand.kind == PointFluxKind::Dry {
                        Tensor::ones(operand.owners.shape(), a.dtype(), c.device())?
                    } else {
                        c.gt(operand.query.area.lower(), &zero)?
                    };
                    let unsafe_mask = c.eq(&safe, &zero)?;
                    let unsafe_mask = c.where_cond(
                        &unsafe_mask,
                        &Tensor::ones(operand.owners.shape(), a.dtype(), c.device())?,
                        &zero,
                    )?;
                    let previous = c.index_select(&pending, 0, &positions)?;
                    pending = c.index_put(
                        &pending,
                        0,
                        &positions,
                        &c.maximum(&previous, &unsafe_mask)?,
                    )?;
                    let ready = c.mul(&safe, &c.index_select(&complete, 0, &positions)?)?;
                    let available = c.eq(&c.index_select(&chosen, 0, &positions)?, &zero)?;
                    let take = c.mul(&ready, &available)?;
                    Ok((c.nonzero(&safe)?.flatten()?, c.nonzero(&take)?.flatten()?))
                })?;
            if safe_rows.numel() != 0 {
                checked::<R, _>(c, "point flux complete physical candidate hull", |a| {
                    let safe = point_flux_operands::select(a, &operand, &safe_rows)?;
                    let momentum = if safe.kind == PointFluxKind::Dry {
                        a.point(&Tensor::zeros(safe.owners.shape(), a.dtype(), c.device())?)?
                    } else {
                        let side = match safe.kind {
                            PointFluxKind::Exterior(side) | PointFluxKind::Fan(side) => {
                                side.index()
                            }
                            _ => 0,
                        };
                        let pressure = super::point_flux_pressure::physical(
                            a,
                            self.sources,
                            &safe.parent,
                            &safe.query,
                            side,
                        )?;
                        let velocity =
                            a.div_positive_interval(&safe.discharge, &safe.query.area)?;
                        let mut advective = a.mul(&safe.discharge, &velocity)?;
                        let stage = self.sources.select_rows(&safe.parent, &safe.query)?;
                        if let Some([area, discharge]) = &safe.captured {
                            let (_, retained) =
                                super::point_flux_arithmetic::advection(a, area, discharge)?;
                            advective = super::point_flux_arithmetic::intersect_advection(
                                a, &advective, &retained,
                            )?;
                        } else if safe.kind == PointFluxKind::Star {
                            let uniform_rows = c.nonzero(&stage.velocity.uniform)?.flatten()?;
                            if uniform_rows.numel() != 0 {
                                let parent = c.index_select(&safe.parent, 0, &uniform_rows)?;
                                let uniform = self.sources.select_sources(a, &parent)?;
                                let input = uniform.velocity.input();
                                let (_, retained) = super::point_flux_arithmetic::advection(
                                    a,
                                    &input.left_area,
                                    &input.left_discharge,
                                )?;
                                let ordinary =
                                    crate::channel::flow::surveyed::equation_ops::select_interval(
                                        a,
                                        &advective,
                                        &uniform_rows,
                                    )?;
                                let intersection =
                                    super::point_flux_arithmetic::intersect_advection(
                                        a, &ordinary, &retained,
                                    )?;
                                advective = a.bounds(
                                    &c.index_put(
                                        advective.lower(),
                                        0,
                                        &uniform_rows,
                                        intersection.lower(),
                                    )?,
                                    &c.index_put(
                                        advective.upper(),
                                        0,
                                        &uniform_rows,
                                        intersection.upper(),
                                    )?,
                                )?;
                            }
                        }
                        let momentum = a.add(&advective, &pressure)?;
                        let depth = self
                            .sources
                            .geometry
                            .recover_query(a, &stage.bands, &safe.query)?
                            .state
                            .depth;
                        super::point_flux_quantum::momentum(
                            a,
                            super::point_flux_quantum::MomentumInputs {
                                area: &safe.query.area,
                                discharge: &safe.discharge,
                                depth_upper: depth.upper(),
                                pressure: &pressure,
                                gravity: self.sources.evaluation.config.gravity_m_per_s2,
                            },
                            &momentum,
                        )?
                    };
                    let positions =
                        point_flux_association::positions(a, &association, &safe.owners)?;
                    physical.merge(a, &positions, &safe.discharge, &momentum)?;
                    joint.merge(a, &positions, &safe.discharge, &momentum)
                })?;
            }
            if select_rows.numel() == 0 {
                continue;
            }
            let operand = checked::<R, _>(c, "point flux one candidate selection", |a| {
                point_flux_operands::select(a, &operand, &select_rows)
            })?;
            let point = super::point_flux_selection::evaluate(self.sources, operand)?;
            checked::<R, _>(c, "point flux selected evaluation envelope", |a| {
                let water = a.expansion_interval(&point.discharge)?;
                let water = hull(a, &water, &point.discharge_physical)?;
                let arithmetic = a.expansion_interval(&point.momentum_arithmetic)?;
                let numerical = a.expansion_interval(&point.momentum)?;
                let momentum = hull(a, &point.momentum_evaluation, &arithmetic)?;
                let momentum = hull(a, &momentum, &numerical)?;
                let positions =
                    point_flux_association::positions(a, &association, &point.invocation_owner)?;
                joint.merge(a, &positions, &water, &momentum)?;
                chosen = c.index_put(
                    &chosen,
                    0,
                    &positions,
                    &Tensor::ones(point.invocation_owner.shape(), a.dtype(), c.device())?,
                )?;
                Ok(())
            })?;
            selected.push(point);
        }
        checked::<R, _>(c, "point flux unchanged full-width acceptance", |a| {
            let owners = &candidates.invocation_owner;
            let mut physical_water = physical.water;
            let physical_momentum = physical.momentum;
            let mut joint_water = joint.water;
            let joint_momentum = joint.momentum;
            let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
            let physical_valid =
                c.mul(&complete, &c.mul(&physical.seen, &c.eq(&pending, &zero)?)?)?;
            let joint_valid = c.mul(&chosen, &physical_valid)?;
            let ready = &joint_valid;
            let unreflected = super::point_flux_reflection::apply(
                a,
                self.sources.velocity.input(),
                &self.invocation_rows,
                super::point_flux_reflection::ReflectedFaces {
                    owners,
                    association: &association,
                    ready,
                    physical_water: &mut physical_water,
                    joint_water: &mut joint_water,
                    joint_momentum: &joint_momentum,
                },
                &mut selected,
            )?;
            let accepted = crate::channel::flow::surveyed::acceptance::flux_acceptance_mask(
                a,
                &joint_water,
                &joint_momentum,
                self.sources.evaluation.config,
            )?;
            let status = Tensor::full_scalar(
                owners.shape(),
                DType::I64,
                PointFluxStatus::PendingCandidates as i64 as f64,
                c.device(),
            )?;
            let status = c.where_cond(
                &complete,
                &c.where_cond(
                    &c.gt(&pending, &zero)?,
                    &Tensor::full_scalar(
                        owners.shape(),
                        DType::I64,
                        PointFluxStatus::PendingPositiveArea as i64 as f64,
                        c.device(),
                    )?,
                    &c.where_cond(
                        ready,
                        &c.where_cond(
                            &accepted,
                            &Tensor::full_scalar(
                                owners.shape(),
                                DType::I64,
                                PointFluxStatus::Accepted as i64 as f64,
                                c.device(),
                            )?,
                            &Tensor::full_scalar(
                                owners.shape(),
                                DType::I64,
                                PointFluxStatus::FluxBudgetExceeded as i64 as f64,
                                c.device(),
                            )?,
                        )?,
                        &status,
                    )?,
                )?,
                &status,
            )?;
            // A mirror face the reflection identity cannot certify has an unresolved trace.
            let status = c.where_cond(
                &c.gt(&unreflected, &zero)?,
                &Tensor::full_scalar(
                    owners.shape(),
                    DType::I64,
                    PointFluxStatus::WaveEnvelopeUnresolved as i64 as f64,
                    c.device(),
                )?,
                &status,
            )?;
            Ok(PointFluxCertificates {
                candidates,
                status,
                physical_valid,
                joint_valid,
                physical_water,
                physical_momentum,
                joint_water,
                joint_momentum,
                selected,
                callback_points: self.callback_points(),
                point_limit: self.point_limit(),
            })
        })
    }
}

#[cfg(test)]
#[path = "point_flux_tests.rs"]
mod tests;
