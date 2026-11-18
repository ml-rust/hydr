use super::{
    point_flux_types::{PointFluxKind, SelectedPointFlux},
    types::CapturedPointInput,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Face state of one selection, in the face order of its candidates.
pub(super) struct ReflectedFaces<'f, R: Runtime<DType = DType>> {
    /// Invocation owner of each face.
    pub(super) owners: &'f Tensor<R>,
    /// Face position of each invocation owner.
    pub(super) association: &'f Tensor<R>,
    /// One where the face holds a complete selected trace.
    pub(super) ready: &'f Tensor<R>,
    pub(super) physical_water: &'f mut Interval<R>,
    pub(super) joint_water: &'f mut Interval<R>,
    pub(super) joint_momentum: &'f Interval<R>,
}

/// Applies the exact reflection identity to the ready mirror faces of one selection.
///
/// A mirror face has both sides wet, equal captured areas in every component, and captured
/// discharges that are exact negatives in every component, with equal radii.
/// The reflected problem is the same problem, so the interface discharge is exactly 0.
/// A certified mirror face takes +0.0 water bounds, a +0.0 selected discharge and velocity,
/// and the pressure-only momentum of the same trace. The momentum bounds stay unchanged.
///
/// Returns the zero-one mask of mirror faces the identity cannot certify: water bounds that
/// exclude 0, a trace that is neither star nor dry, or a pressure enclosure outside the
/// momentum bound. Those faces and every other face keep their values bit for bit.
pub(super) fn apply<R>(
    a: &mut Arithmetic<'_, R>,
    input: &CapturedPointInput<R>,
    invocation_rows: &Tensor<R>,
    faces: ReflectedFaces<'_, R>,
    selected: &mut [SelectedPointFlux<R>],
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
    let zero = Tensor::zeros(faces.owners.shape(), a.dtype(), c.device())?;
    if faces.owners.numel() == 0 {
        return Ok(zero);
    }
    let mirror = c.mul(
        &mirror_mask(a, input, invocation_rows, faces.owners)?,
        faces.ready,
    )?;
    // Joint and physical water must already enclose the exact value 0.
    let mut excluded = zero.clone();
    for bound in [&*faces.physical_water, &*faces.joint_water] {
        let contains = c.mul(&c.le(bound.lower(), &zero)?, &c.ge(bound.upper(), &zero)?)?;
        excluded = c.maximum(&excluded, &c.eq(&contains, &zero)?)?;
    }
    let mut rejected = c.mul(&mirror, &excluded)?;
    // The selection loop already checked every selected owner's position.
    let mut positions = Vec::with_capacity(selected.len());
    for point in selected.iter() {
        let at = c.index_select(faces.association, 0, &point.invocation_owner)?;
        let rows = c.index_select(&mirror, 0, &at)?;
        let row_zero = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
        let enclosure = match (point.kind, &point.pressure) {
            (PointFluxKind::Star, Some(pressure)) => {
                let chosen = a.expansion_interval(&pressure.chosen)?;
                Some(a.bounds(
                    &c.minimum(chosen.lower(), pressure.evaluation.lower())?,
                    &c.maximum(chosen.upper(), pressure.evaluation.upper())?,
                )?)
            }
            (PointFluxKind::Dry, None) => Some(a.point(&row_zero)?),
            // A mirror face has velocity 0 at the interface, so a fan or exterior trace
            // contradicts the identity.
            _ => None,
        };
        let failed = match enclosure {
            Some(enclosure) => {
                let lower = c.index_select(faces.joint_momentum.lower(), 0, &at)?;
                let upper = c.index_select(faces.joint_momentum.upper(), 0, &at)?;
                let inside = c.mul(
                    &c.le(&lower, enclosure.lower())?,
                    &c.le(enclosure.upper(), &upper)?,
                )?;
                c.mul(&rows, &c.eq(&inside, &row_zero)?)?
            }
            None => rows,
        };
        let previous = c.index_select(&rejected, 0, &at)?;
        rejected = c.index_put(&rejected, 0, &at, &c.maximum(&previous, &failed)?)?;
        positions.push(at);
    }
    let certified = c.gt(&c.mul(&mirror, &c.eq(&rejected, &zero)?)?, &zero)?;
    for bound in [&mut *faces.physical_water, &mut *faces.joint_water] {
        *bound = a.bounds(
            &c.where_cond(&certified, &zero, bound.lower())?,
            &c.where_cond(&certified, &zero, bound.upper())?,
        )?;
    }
    for (point, at) in selected.iter_mut().zip(&positions) {
        let take = c.index_select(&certified, 0, at)?;
        let row_zero = Tensor::zeros(take.shape(), a.dtype(), c.device())?;
        let replace = |a: &mut Arithmetic<'_, R>,
                       value: &CertifiedExpansion<R>,
                       by: [&Tensor<R>; 3]|
         -> HydraulicResult<CertifiedExpansion<R>> {
            Ok(a.expansion_bounds(
                &c.where_cond(&take, by[0], value.high())?,
                &c.where_cond(&take, by[1], value.low())?,
                &c.where_cond(&take, by[2], value.abs_error())?,
            )?)
        };
        let zeros = [&row_zero, &row_zero, &row_zero];
        point.discharge = replace(a, &point.discharge, zeros)?;
        point.velocity = replace(a, &point.velocity, zeros)?;
        point.reflected = c.maximum(&point.reflected, &take)?;
        // A dry trace already holds zero momentum.
        if let (PointFluxKind::Star, Some(pressure)) = (point.kind, &point.pressure) {
            let chosen = &pressure.chosen;
            point.momentum = replace(
                a,
                &point.momentum,
                [chosen.high(), chosen.low(), chosen.abs_error()],
            )?;
        }
    }
    Ok(rejected)
}

/// Zero-one mask, in face order, of the faces whose captured states are exact mirrors.
/// Components compare by value, so +0.0 and -0.0 low parts count as equal.
fn mirror_mask<R>(
    a: &mut Arithmetic<'_, R>,
    input: &CapturedPointInput<R>,
    invocation_rows: &Tensor<R>,
    owners: &Tensor<R>,
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
    let source = c.index_select(invocation_rows, 0, owners)?;
    let take = |value: &Tensor<R>| c.index_select(value, 0, &source);
    let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
    let (left_area, right_area) = (&input.left_area, &input.right_area);
    let (left_q, right_q) = (&input.left_discharge, &input.right_discharge);
    let mut mask = c.gt(&take(left_area.high())?, &zero)?;
    for (left, right) in [
        (left_area.high(), right_area.high()),
        (left_area.low(), right_area.low()),
        (left_area.abs_error(), right_area.abs_error()),
        // A negated expansion keeps its radius.
        (left_q.abs_error(), right_q.abs_error()),
    ] {
        mask = c.mul(&mask, &c.eq(&take(left)?, &take(right)?)?)?;
    }
    for (left, right) in [
        (left_q.high(), right_q.high()),
        (left_q.low(), right_q.low()),
    ] {
        mask = c.mul(&mask, &c.eq(&take(left)?, &c.neg(&take(right)?)?)?)?;
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::super::corpus_coverage::Coverage;
    use super::super::point_flux::fixture::diagnostics;
    use super::super::{source_fixture_support as frozen, wet_depth::fixture as support};
    use frozen::{Result, hydraulic};

    mod reflection {
        use super::super::super::{
            corpus_coverage::Coverage,
            point_flux_association,
            point_flux_reflection::{self, ReflectedFaces},
            point_flux_types::{PointFluxCertificates, PointFluxKind, PointFluxStatus},
            point_root_session::PointRootSession,
            source_sonic_types::SonicSide,
            wet_star_levels::tests::{MIRROR, WALLS, integer_trapezoid, trapezoid},
        };
        use super::{diagnostics, frozen, hydraulic, support};
        use crate::channel::{
            flow::surveyed::{PrismaticRiemannConfig, error::checked},
            section::PrismaticSectionBatch,
        };
        use frozen::Result;
        use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

        /// The reach allowances: water and momentum atol and the relative allowance are all 1e-12.
        const ALLOWANCE: f64 = 1e-12;

        fn config() -> PrismaticRiemannConfig {
            PrismaticRiemannConfig {
                water_flux_atol_m3_per_s: ALLOWANCE,
                momentum_flux_atol_m4_per_s2: ALLOWANCE,
                flux_rtol: ALLOWANCE,
                ..PrismaticRiemannConfig::default()
            }
        }

        pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            near_mirror::<R>(client)?;
            fan::<R>(client)?;
            identity::<R>(client, &trapezoid::<R>(client)?, "mirror", MIRROR[0])?;
            if coverage.is_full() {
                let section = integer_trapezoid::<R>(client)?;
                for (label, state) in WALLS {
                    identity::<R>(client, &section, label, state)?;
                }
            }
            Ok(())
        }

        /// Each exact mirror face resolves to zero water and the pressure-only momentum.
        fn identity<R>(
            client: &R::Client,
            section: &PrismaticSectionBatch<R>,
            label: &str,
            state: [f64; 4],
        ) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = config();
            support::with_section::<R, _>(client, section, &[state], &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.resolve_point_fluxes(&owners))?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::Accepted as i64],
                    "{label}"
                );
                for selection in &result.selections {
                    diagnostics::selection(&session, selection)?;
                }
                let index = usize::try_from(result.selection.try_to_vec::<i64>()?[0])?;
                let owning = result
                    .selections
                    .get(index)
                    .ok_or("Face names an absent selection")?;
                let point = owning
                    .selected
                    .first()
                    .ok_or("Owning selection lacks its trace")?;
                assert!(
                    matches!(point.kind, PointFluxKind::Star | PointFluxKind::Dry),
                    "{label} kind {:?}",
                    point.kind
                );
                assert_eq!(point.reflected.try_to_vec::<f64>()?, [1.], "{label}");
                assert_eq!(frozen::components(&point.discharge)?, [[0; 3]], "{label}");
                for water in [&result.joint_water, &result.physical_water] {
                    for value in [water.lower(), water.upper()] {
                        assert_eq!(bits(value)?, [0], "{label}");
                    }
                }
                let momentum = frozen::endpoints(&result.joint_momentum)?[0];
                let width = momentum[1] - momentum[0];
                let allowance = ALLOWANCE + ALLOWANCE * momentum[0].abs().max(momentum[1].abs());
                eprintln!(
                    "POINT_FLUX_REFLECTION\t{label}\tkind={:?}\trounds={}\tmomentum={momentum:?}\twidth={width:e}\tallowance={allowance:e}",
                    point.kind, result.rounds
                );
                assert!(width <= allowance, "{label} momentum width {width:e}");
                Ok(())
            })
        }

        /// A 1-ulp area difference is not a mirror: the identity leaves every value bit for bit.
        fn near_mirror<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = config();
            let section = trapezoid::<R>(client)?;
            let [left_area, left_q, right_area, right_q] = MIRROR[0];
            let state = [
                left_area,
                left_q,
                f64::from_bits(right_area.to_bits() + 1),
                right_q,
            ];
            support::with_section::<R, _>(client, &section, &[state], &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let mut result = hydraulic(session.select_point_fluxes(&owners))?;
                diagnostics::selection(&session, &result)?;
                assert_ne!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::WaveEnvelopeUnresolved as i64]
                );
                for point in &result.selected {
                    assert_eq!(point.reflected.try_to_vec::<f64>()?, [0.]);
                }
                let before = snapshot(&result)?;
                let rejected = reflect(&session, &mut result)?;
                assert_eq!(rejected, [0.]);
                assert_eq!(snapshot(&result)?, before);
                Ok(())
            })
        }

        /// A mirror face whose selected trace is a fan cannot take the identity.
        fn fan<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = config();
            let section = trapezoid::<R>(client)?;
            support::with_section::<R, _>(client, &section, &MIRROR, &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let mut result = hydraulic(session.select_point_fluxes(&owners))?;
                assert_eq!(result.joint_valid.try_to_vec::<f64>()?, [1.]);
                for point in &mut result.selected {
                    point.kind = PointFluxKind::Fan(SonicSide::Left);
                }
                let before = snapshot(&result)?;
                let rejected = reflect(&session, &mut result)?;
                assert_eq!(rejected, [1.]);
                // A rejected face keeps its values.
                assert_eq!(snapshot(&result)?, before);
                Ok(())
            })
        }

        /// Applies the identity again to one selection and returns its rejection mask.
        fn reflect<R>(
            session: &PointRootSession<'_, '_, R>,
            result: &mut PointFluxCertificates<R>,
        ) -> Result<Vec<f64>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let client = session.sources.geometry.client();
            let count = session.invocation_rows.numel();
            let rejected = hydraulic(checked::<R, _>(client, "reflection identity rerun", |a| {
                let owners = &result.candidates.invocation_owner;
                let association = point_flux_association::inverse(a, count, owners)?;
                point_flux_reflection::apply(
                    a,
                    session.sources.velocity.input(),
                    &session.invocation_rows,
                    ReflectedFaces {
                        owners,
                        association: &association,
                        ready: &result.joint_valid,
                        physical_water: &mut result.physical_water,
                        joint_water: &mut result.joint_water,
                        joint_momentum: &result.joint_momentum,
                    },
                    &mut result.selected,
                )
            }))?;
            Ok(rejected.try_to_vec::<f64>()?)
        }

        /// Bits of the water bounds and of each selected discharge, velocity and momentum.
        fn snapshot<R>(result: &PointFluxCertificates<R>) -> Result<Vec<Vec<u64>>>
        where
            R: Runtime<DType = DType>,
        {
            let mut values = Vec::new();
            for water in [&result.physical_water, &result.joint_water] {
                values.push(bits(water.lower())?);
                values.push(bits(water.upper())?);
            }
            for point in &result.selected {
                for value in [&point.discharge, &point.velocity, &point.momentum] {
                    values.push(frozen::components(value)?.concat());
                }
            }
            Ok(values)
        }

        fn bits<R: Runtime<DType = DType>>(value: &Tensor<R>) -> Result<Vec<u64>> {
            Ok(value
                .try_to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect())
        }
    }

    #[test]
    fn configured_cpu_point_fluxes_reflect_exact_mirror_faces() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        reflection::run::<CpuRuntime>(&CpuClient::new(CpuDevice::new()), Coverage::Full)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_fluxes_reflect_exact_mirror_faces() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        reflection::run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
    }
}
