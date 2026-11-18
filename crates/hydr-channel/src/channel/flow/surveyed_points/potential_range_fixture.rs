use super::potential_increment_fixtures as frozen;
use super::potential_range::PotentialValue;
use super::source_fixture_support::{self as support, Client, Result, components, endpoints, zero};
use crate::channel::{
    flow::surveyed::PrismaticRiemannConfig,
    section::{
        CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
    },
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn section<R>(
    client: &R::Client,
    profiles: &[frozen::Profile],
) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definitions = profiles
        .iter()
        .map(|profile| CrossSectionDefinition {
            id: profile.id.into(),
            vertical_datum: "Frozen potential datum".into(),
            points: profile
                .coordinates
                .iter()
                .map(|point| CrossSectionPoint {
                    station_m: f64::from_bits(point[0]),
                    elevation_m: f64::from_bits(point[1]),
                })
                .collect(),
            maximum_stage_m: f64::from_bits(profile.cap_stage),
            profile_provenance: "Independent original-coordinate potential reference".into(),
        })
        .collect();
    Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
}

pub(super) fn config(gravity: u64) -> PrismaticRiemannConfig {
    let mut config = PrismaticRiemannConfig {
        gravity_m_per_s2: f64::from_bits(gravity),
        water_flux_atol_m3_per_s: 1e-12,
        momentum_flux_atol_m4_per_s2: 1e-12,
        flux_rtol: 1e-12,
        ..PrismaticRiemannConfig::default()
    };
    config.root.atol = 1e-15;
    config.root.rtol = 1e-15;
    config.root.max_iterations = 256;
    config
}

pub(super) fn check(actual: [f64; 2], reference: frozen::Reference, label: &str) {
    let bits: frozen::Bounds = reference.bounds;
    let expected = bits.map(f64::from_bits);
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
    let excess = (actual[0] - expected[1])
        .abs()
        .max((actual[1] - expected[0]).abs());
    let allowance = f64::from_bits(reference.native_allowance);
    assert!(
        excess <= allowance,
        "{label}: error={excess:e}, allowance={allowance:e}, {actual:?}"
    );
    if reference.policy == frozen::ReferencePolicy::ExactZero {
        assert_eq!(actual, [0.0; 2], "{label}");
    }
}

pub(super) fn check_value<R>(
    client: &R::Client,
    value: &PotentialValue<R>,
    reference: frozen::Reference,
    label: &str,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    for interval in [
        &value.physical,
        &support::bounds::<R>(client, &value.retained)?,
    ] {
        for actual in endpoints(interval)? {
            check(actual, reference, label);
        }
    }
    if reference.policy == frozen::ReferencePolicy::ExactZero {
        for point in components(&value.retained)? {
            zero(point, label);
        }
    }
    Ok(())
}

pub(super) fn target_check<R>(
    client: &R::Client,
    value: &PotentialValue<R>,
    increment: &PotentialValue<R>,
    reference: &frozen::query_profiles::Endpoint,
    label: &str,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    check_value::<R>(client, value, reference.potential, label)?;
    check_value::<R>(client, increment, reference.increment, label)?;
    if reference.structural_equality {
        for point in components(&increment.retained)? {
            zero(point, label);
        }
    }
    Ok(())
}
