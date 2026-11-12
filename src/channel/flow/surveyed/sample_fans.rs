use super::{
    curves::{fan_velocity, select_side},
    equation_ops::select_interval,
    error::{HydraulicResult, checked},
    roots::solve_sonic,
    sample_candidates::{exterior, from_state},
    sample_merge::Samples,
    types::{Oracle, WaveSide},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::Interval, integrate::CertifiedSimpsonAlgorithms,
    optimize::CertifiedBisectionAlgorithms,
};

pub(crate) fn fan<R>(
    oracle: &Oracle<'_, R>,
    side: WaveSide,
    faces: &Tensor<R>,
    normalized_depth_bounds: (&Tensor<R>, &Tensor<R>),
    head: &Interval<R>,
    tail: &Interval<R>,
    samples: &mut Samples<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    if faces.numel() == 0 {
        return Ok(());
    }
    let c = oracle.geometry.client();
    let zero = Tensor::zeros(&[], DType::F64, c.device())?;
    let active = checked(c, "surveyed fan classification", |a| {
        let state = select_side(
            a,
            match side {
                WaveSide::Left => oracle.left,
                WaveSide::Right => oracle.right,
            },
            faces,
        )?;
        let (outside, sonic) = match side {
            WaveSide::Left => (
                c.ge(head.upper(), &zero)?,
                c.mul(&c.le(head.lower(), &zero)?, &c.ge(tail.upper(), &zero)?)?,
            ),
            WaveSide::Right => (
                c.le(head.lower(), &zero)?,
                c.mul(&c.le(tail.lower(), &zero)?, &c.ge(head.upper(), &zero)?)?,
            ),
        };
        let candidate = exterior(a, &state, outside, oracle.config.gravity_m_per_s2)?;
        samples.merge(a, faces, candidate)?;
        samples.wave(a, faces, head)?;
        samples.wave(a, faces, tail)?;
        Ok(c.nonzero(&sonic)?.flatten()?)
    })?;
    if active.numel() == 0 {
        return Ok(());
    }
    let sonic_faces = c.index_select(faces, 0, &active)?;
    let lower = c.index_select(normalized_depth_bounds.0, 0, &active)?;
    let upper = c.index_select(normalized_depth_bounds.1, 0, &active)?;
    let roots = solve_sonic(oracle, side, &sonic_faces, &lower, &upper)?;
    checked(c, "surveyed sonic work status", |a| {
        samples.root_status(a, &sonic_faces, &roots.status)
    })?;
    if roots.face_owner.numel() != 0 {
        merge_fan_state(
            oracle,
            side,
            &roots.face_owner,
            &roots.normalized_depth,
            samples,
        )?;
    }
    // An uncertain bracket admits the whole supported fan segment, never a nominal root.
    let unresolved_code = Tensor::from_slice(&[2_i64], &[], c.device())?;
    let uncertain = c
        .nonzero(&c.eq(&roots.status, &unresolved_code)?)?
        .flatten()?;
    if uncertain.numel() != 0 {
        let faces = c.index_select(&sonic_faces, 0, &uncertain)?;
        let domain = checked(c, "surveyed unresolved sonic hull", |a| {
            Ok(a.bounds(
                &c.index_select(&lower, 0, &uncertain)?,
                &c.index_select(&upper, 0, &uncertain)?,
            )?)
        })?;
        merge_fan_state(oracle, side, &faces, &domain, samples)?;
    }
    Ok(())
}

fn merge_fan_state<R>(
    oracle: &Oracle<'_, R>,
    side: WaveSide,
    faces: &Tensor<R>,
    t: &Interval<R>,
    samples: &mut Samples<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = oracle.geometry.client();
    let state = checked(c, "surveyed sonic geometry", |a| {
        let bands = oracle.bands.select(a, faces)?;
        oracle.geometry.at_normalized_depth(a, &bands, t)
    })?;
    let velocity = fan_velocity(oracle, side, faces, &state)?;
    checked(c, "surveyed sonic candidate", |a| {
        let possible = Tensor::ones(faces.shape(), DType::F64, c.device())?;
        let candidate = from_state(
            a,
            &state,
            &velocity,
            None,
            possible,
            oracle.config.gravity_m_per_s2,
        )?;
        samples.merge(a, faces, candidate)
    })
}

pub(crate) fn normalized_segment<R>(
    oracle: &Oracle<'_, R>,
    faces: &Tensor<R>,
    near: &Interval<R>,
    far: &Interval<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = oracle.geometry.client();
    checked(c, "surveyed fan domain", |a| {
        let cap = select_interval(a, &oracle.bands.cap_depth, faces)?;
        let low = a.div_positive_interval(near, &cap)?;
        let high = a.div_positive_interval(far, &cap)?;
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        let one = Tensor::ones(&[], DType::F64, c.device())?;
        Ok((
            c.minimum(&one, &c.maximum(&zero, low.lower())?)?,
            c.maximum(&zero, &c.minimum(&one, high.upper())?)?,
        ))
    })
}
