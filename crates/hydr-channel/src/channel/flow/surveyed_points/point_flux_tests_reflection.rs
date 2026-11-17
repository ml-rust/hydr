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
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
