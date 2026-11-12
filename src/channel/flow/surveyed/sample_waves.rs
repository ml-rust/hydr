use super::{
    analytic_bottom::{dry_front, potential_squared, vacuum as analytic_vacuum},
    curves::{select_side, sound, star_velocity},
    equation_ops::select_interval,
    error::{HydraulicError, HydraulicResult, PrismaticRiemannRejection, checked},
    pressure::pressure_secant,
    roots::{RootSet, solve_star},
    sample_candidates::{dry, exterior, from_state},
    sample_fans::{fan, normalized_segment},
    sample_merge::{SampledSolution, Samples},
    shock,
    types::{Oracle, WaveSide},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps, TypeConversionOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

pub(crate) fn sample<R>(oracle: &Oracle<'_, R>) -> HydraulicResult<SampledSolution<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = oracle.geometry.client();
    let count = oracle.left.original_area.numel();
    let all = c.arange(0.0, count as f64, 1.0, DType::I64)?;
    let mut samples = checked(c, "surveyed sample allocation", |a| Samples::new(a, count))?;
    if count == 0 {
        return checked(c, "surveyed empty sample", |a| samples.finish(a));
    }
    let uniform = checked(c, "surveyed uniform state sample", |a| {
        let equal = c.mul(
            &c.eq(&oracle.left.original_area, &oracle.right.original_area)?,
            &c.eq(
                &oracle.left.original_discharge,
                &oracle.right.original_discharge,
            )?,
        )?;
        let candidate = exterior(
            a,
            oracle.left,
            equal.clone(),
            oracle.config.gravity_m_per_s2,
        )?;
        samples.merge(a, &all, candidate)?;
        Ok(equal)
    })?;
    let (vacuum_certificate, left_front_certificate, right_front_certificate) =
        checked(c, "surveyed analytic bottom certificates", |a| {
            let left =
                potential_squared(a, oracle.bands, oracle.left, oracle.config.gravity_m_per_s2)?;
            let right = potential_squared(
                a,
                oracle.bands,
                oracle.right,
                oracle.config.gravity_m_per_s2,
            )?;
            Ok((
                analytic_vacuum(a, oracle.left, oracle.right, &left, &right)?,
                dry_front(a, oracle.left, WaveSide::Left, &left)?,
                dry_front(a, oracle.right, WaveSide::Right, &right)?,
            ))
        })?;
    let (star_faces, vacuum_faces) = checked(c, "surveyed vacuum classification", |a| {
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        let unequal = c.eq(&uniform, &zero)?;
        let wet = c.mul(&unequal, &c.mul(&oracle.left.wet, &oracle.right.wet)?)?;
        let difference = a.sub(&oracle.right.velocity, &oracle.left.velocity)?;
        let sum = a.add(&oracle.left.potential, &oracle.right.potential)?;
        let residual = a.sub(&difference, &sum)?;
        // Ordered dry fronts prove vacuum independently of the rounded velocity difference.
        let separated_fronts = c.mul(&left_front_certificate, &right_front_certificate)?;
        let vacuum = c.maximum(
            &c.maximum(&oracle.left.dry, &oracle.right.dry)?,
            &c.mul(
                &wet,
                &c.maximum(
                    &separated_fronts,
                    &c.maximum(&vacuum_certificate, &c.ge(residual.lower(), &zero)?)?,
                )?,
            )?,
        )?;
        let vacuum = c.mul(&unequal, &vacuum)?;
        let nonvacuum = c.mul(
            &wet,
            &c.mul(&c.eq(&vacuum, &zero)?, &c.lt(residual.upper(), &zero)?)?,
        )?;
        let ambiguous = c.mul(&wet, &c.eq(&c.maximum(&vacuum, &nonvacuum)?, &zero)?)?;
        if c.cast(&c.any(&ambiguous, &[], false)?, DType::F64)?
            .item::<f64>()?
            != 0.0
        {
            let rejected = c.nonzero(&ambiguous)?.flatten()?;
            let (query, section_owner) = rejection_identity(oracle, &all, &rejected)?;
            return Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::RootBracketUnresolved, query, section_owner, context: "Vacuum criterion overlaps zero through potential uncertainty. Refine quadrature before selecting a regime".into() });
        }
        // Signed envelopes include both exterior characteristic families and dry fronts.
        for state in [oracle.left, oracle.right] {
            let low = a.sub(&state.velocity, &state.sound)?;
            let high = a.add(&state.velocity, &state.sound)?;
            samples.wave(a, &all, &low)?;
            samples.wave(a, &all, &high)?;
        }
        Ok((
            c.nonzero(&nonvacuum)?.flatten()?,
            c.nonzero(&vacuum)?.flatten()?,
        ))
    })?;
    if vacuum_faces.numel() != 0 {
        sample_vacuum(
            oracle,
            &vacuum_faces,
            &left_front_certificate,
            &right_front_certificate,
            &mut samples,
        )?;
    }
    if star_faces.numel() != 0 {
        let roots = solve_star(oracle, &star_faces)?;
        checked(c, "surveyed star work status", |a| {
            samples.root_status(a, &star_faces, &roots.status)
        })?;
        if roots.face_owner.numel() != star_faces.numel() {
            let no_root = Tensor::from_slice(&[1_i64], &[], c.device())?;
            let absent = c.nonzero(&c.eq(&roots.status, &no_root)?)?.flatten()?;
            if absent.numel() != 0 {
                let (query, section_owner) = rejection_identity(oracle, &star_faces, &absent)?;
                return Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::CapExceeded, query, section_owner, context: "Wet star residual remains negative at the physical surveyed cap. Extend surveyed support or change the input state".into() });
            }
            let unresolved = Tensor::from_slice(&[2_i64], &[], c.device())?;
            let rejected = c.nonzero(&c.eq(&roots.status, &unresolved)?)?.flatten()?;
            let (query, section_owner) = rejection_identity(oracle, &star_faces, &rejected)?;
            return Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::RootBracketUnresolved, query, section_owner, context: "Wet star endpoint signs do not certify existence within t=[0,1]. Refine potential enclosures".into() });
        }
        sample_star(oracle, &roots, &mut samples)?;
    }
    checked(c, "surveyed sample coverage", |a| samples.finish(a))
}

fn rejection_identity<R>(
    oracle: &Oracle<'_, R>,
    invocation_faces: &Tensor<R>,
    rejected_rows: &Tensor<R>,
) -> HydraulicResult<(Option<usize>, Option<usize>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if rejected_rows.numel() == 0 {
        return Ok((None, None));
    }
    let c = oracle.geometry.client();
    // Status rows index the invocation, including rows without certified roots.
    let original_faces = c.index_select(invocation_faces, 0, rejected_rows)?;
    let first = c.min(&original_faces, &[0], true)?;
    let original_query = first.item::<i64>()?;
    let original_owner = c
        .index_select(&oracle.bands.section_owner, 0, &first)?
        .item::<i64>()?;
    let query = usize::try_from(original_query).map_err(|_| HydraulicError::Numerical {
        context: format!("Rejected query={original_query} cannot index the original input. Keep captured ownership immutable"),
    })?;
    let section_owner = usize::try_from(original_owner).map_err(|_| HydraulicError::Numerical {
        context: format!("Rejected section_owner={original_owner} cannot index the original sections. Keep captured ownership immutable"),
    })?;
    Ok((Some(query), Some(section_owner)))
}

fn sample_vacuum<R>(
    oracle: &Oracle<'_, R>,
    faces: &Tensor<R>,
    left_front_certificate: &Tensor<R>,
    right_front_certificate: &Tensor<R>,
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
    let (left_exhausted, right_exhausted) = checked(c, "surveyed vacuum dry sample", |a| {
        let left = select_side(a, oracle.left, faces)?;
        let right = select_side(a, oracle.right, faces)?;
        let left_front = a.add(&left.velocity, &left.potential)?;
        let right_front = a.sub(&right.velocity, &right.potential)?;
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        let left_proved = c.index_select(left_front_certificate, 0, faces)?;
        let right_proved = c.index_select(right_front_certificate, 0, faces)?;
        let left_exhausted = c.maximum(&left_proved, &c.le(left_front.upper(), &zero)?)?;
        let right_exhausted = c.maximum(&right_proved, &c.ge(right_front.lower(), &zero)?)?;
        let after_left = c.maximum(
            &left.dry,
            &c.maximum(&left_proved, &c.le(left_front.lower(), &zero)?)?,
        )?;
        let before_right = c.maximum(
            &right.dry,
            &c.maximum(&right_proved, &c.ge(right_front.upper(), &zero)?)?,
        )?;
        let candidate = dry(a, c.mul(&after_left, &before_right)?)?;
        samples.merge(a, faces, candidate)?;
        samples.wave(a, faces, &left_front)?;
        samples.wave(a, faces, &right_front)?;
        Ok((left_exhausted, right_exhausted))
    })?;
    for side in [WaveSide::Left, WaveSide::Right] {
        let state = match side {
            WaveSide::Left => oracle.left,
            WaveSide::Right => oracle.right,
        };
        let exhausted = match side {
            WaveSide::Left => &left_exhausted,
            WaveSide::Right => &right_exhausted,
        };
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        let wet = c.mul(
            &c.index_select(&state.wet, 0, faces)?,
            &c.eq(exhausted, &zero)?,
        )?;
        let rows = c.nonzero(&wet)?.flatten()?;
        let wet_faces = c.index_select(faces, 0, &rows)?;
        if wet_faces.numel() == 0 {
            continue;
        }
        let (head, tail, lower, upper) = checked(c, "surveyed dry expansion", |a| {
            let state = select_side(a, state, &wet_faces)?;
            let (head, tail) = match side {
                WaveSide::Left => (
                    a.sub(&state.velocity, &state.sound)?,
                    a.add(&state.velocity, &state.potential)?,
                ),
                WaveSide::Right => (
                    a.add(&state.velocity, &state.sound)?,
                    a.sub(&state.velocity, &state.potential)?,
                ),
            };
            let cap = select_interval(a, &oracle.bands.cap_depth, &wet_faces)?;
            let far = a.div_positive_interval(&state.geometry.depth, &cap)?;
            let zero = Tensor::zeros(wet_faces.shape(), DType::F64, c.device())?;
            let one = Tensor::ones(&[], DType::F64, c.device())?;
            Ok((head, tail, zero, c.minimum(&one, far.upper())?))
        })?;
        fan(
            oracle,
            side,
            &wet_faces,
            (&lower, &upper),
            &head,
            &tail,
            samples,
        )?;
    }
    Ok(())
}

fn sample_star<R>(
    oracle: &Oracle<'_, R>,
    roots: &RootSet<R>,
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
    let faces = &roots.face_owner;
    let star = checked(c, "surveyed star geometry", |a| {
        let bands = oracle.bands.select(a, faces)?;
        oracle
            .geometry
            .at_normalized_depth(a, &bands, &roots.normalized_depth)
    })?;
    let velocity = star_velocity(oracle, faces, &roots.normalized_depth)?;
    let mut inside = Tensor::ones(faces.shape(), DType::F64, c.device())?;
    for side in [WaveSide::Left, WaveSide::Right] {
        let (allowed, rare_rows, head, tail, side_depth) =
            checked(c, "surveyed star waves", |a| {
                let source = match side {
                    WaveSide::Left => oracle.left,
                    WaveSide::Right => oracle.right,
                };
                let source = select_side(a, source, faces)?;
                let bands = oracle.bands.select(a, faces)?;
                let secant = pressure_secant(
                    a,
                    oracle.geometry,
                    &bands,
                    &source.geometry,
                    &star,
                    oracle.config.gravity_m_per_s2,
                )?;
                let offset = shock::offset(a, &star.area, &source.original_area, &secant)?;
                let star_sound = sound(a, &star, oracle.config.gravity_m_per_s2)?;
                let zero = Tensor::zeros(&[], DType::F64, c.device())?;
                let shock_possible = c.ge(star.area.upper(), &source.original_area)?;
                let rare_possible = c.le(star.area.lower(), &source.original_area)?;
                let (shock, head, tail) = match side {
                    WaveSide::Left => (
                        a.sub(&source.velocity, &offset)?,
                        a.sub(&source.velocity, &source.sound)?,
                        a.sub(&velocity, &star_sound)?,
                    ),
                    WaveSide::Right => (
                        a.add(&source.velocity, &offset)?,
                        a.add(&source.velocity, &source.sound)?,
                        a.add(&velocity, &star_sound)?,
                    ),
                };
                let (outside, inside_shock, inside_fan) = match side {
                    WaveSide::Left => (
                        c.ge(shock.upper(), &zero)?,
                        c.le(shock.lower(), &zero)?,
                        c.le(tail.lower(), &zero)?,
                    ),
                    WaveSide::Right => (
                        c.le(shock.lower(), &zero)?,
                        c.ge(shock.upper(), &zero)?,
                        c.ge(tail.upper(), &zero)?,
                    ),
                };
                let candidate = exterior(
                    a,
                    &source,
                    c.mul(&shock_possible, &outside)?,
                    oracle.config.gravity_m_per_s2,
                )?;
                samples.merge(a, faces, candidate)?;
                // Equality retains both traces, with their separately calculated fluxes.
                samples.wave(a, faces, &shock)?;
                let allowed = c.maximum(
                    &c.mul(&shock_possible, &inside_shock)?,
                    &c.mul(&rare_possible, &inside_fan)?,
                )?;
                Ok((
                    allowed,
                    c.nonzero(&rare_possible)?.flatten()?,
                    head,
                    tail,
                    source.geometry.depth,
                ))
            })?;
        inside = c.mul(&inside, &allowed)?;
        if rare_rows.numel() != 0 {
            let rare_faces = c.index_select(faces, 0, &rare_rows)?;
            let (near, far, head, tail) = checked(c, "surveyed rarefaction segment", |a| {
                Ok((
                    select_interval(a, &star.depth, &rare_rows)?,
                    select_interval(a, &side_depth, &rare_rows)?,
                    select_interval(a, &head, &rare_rows)?,
                    select_interval(a, &tail, &rare_rows)?,
                ))
            })?;
            let (lower, upper) = normalized_segment(oracle, &rare_faces, &near, &far)?;
            fan(
                oracle,
                side,
                &rare_faces,
                (&lower, &upper),
                &head,
                &tail,
                samples,
            )?;
        }
    }
    checked(c, "surveyed star candidate", |a| {
        let candidate = from_state(
            a,
            &star,
            &velocity,
            None,
            inside,
            oracle.config.gravity_m_per_s2,
        )?;
        samples.merge(a, faces, candidate)
    })
}
