use super::fixture::{Result, parameters, single};
use hydr::runoff::{
    ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirInput, ReservoirParameters,
    ReservoirState,
};

#[test]
fn native_coefficient_retains_class_area_and_original_operation_order() -> Result {
    let p = parameters()?;
    let alpha_native = 1.49_f64 * 32.0 / 1024.0 * (1.0_f64 / 64.0).sqrt() / (1.0 / 32.0);
    assert_eq!(
        p.coefficient_native()
            .ok_or("missing finite coefficient")?
            .to_bits(),
        alpha_native.to_bits()
    );
    assert_eq!(alpha_native.to_bits(), 0.18625_f64.to_bits());
    let alpha_si = alpha_native * 0.3048_f64.powf(1.0 - 1.6666667);
    assert_eq!(
        p.coefficient_si()
            .ok_or("missing SI coefficient")?
            .to_bits(),
        alpha_si.to_bits()
    );
    let split =
        ReservoirParameters::from_native(256.0, 1024.0, 32.0, 1.0 / 64.0, 1.0 / 32.0, 0.125)?;
    assert_eq!(split.coefficient_native(), p.coefficient_native());
    assert_eq!(split.area_m2(), p.area_m2() / 4.0);
    assert_eq!(split.class_area_m2(), p.class_area_m2());
    assert!(!p.is_closed());
    assert!(!p.is_instantaneous());
    Ok(())
}

#[test]
fn si_constructor_preserves_supplied_geometry_bits() -> Result {
    let values = [
        95.0_f64.next_up(),
        100.0_f64.next_down(),
        9.5_f64.next_up(),
        0.02_f64.next_up(),
        0.03_f64.next_down(),
        0.04_f64.next_up(),
    ];
    let p = ReservoirParameters::from_si(
        values[0], values[1], values[2], values[3], values[4], values[5],
    )?;
    let actual = [
        p.area_m2(),
        p.class_area_m2(),
        p.width_m(),
        p.slope(),
        p.manning(),
        p.depression_depth_m(),
    ];
    for (actual, expected) in actual.into_iter().zip(values) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    Ok(())
}

#[test]
fn invalid_geometry_never_selects_a_different_physical_branch() -> Result {
    for values in [
        [0.0, 1024.0, 32.0, 0.01, 0.03, 0.1],
        [2048.0, 1024.0, 32.0, 0.01, 0.03, 0.1],
        [1024.0, f64::INFINITY, 32.0, 0.01, 0.03, 0.1],
        [1024.0, 1024.0, -1.0, 0.01, 0.03, 0.1],
        [1024.0, 1024.0, 32.0, -0.01, 0.03, 0.1],
        [1024.0, 1024.0, 32.0, 0.01, -0.03, 0.1],
        [1024.0, 1024.0, 32.0, 0.01, 0.03, -0.1],
        [1024.0, 1024.0, 0.0, 0.01, 0.0, 0.1],
        [1024.0, 1024.0, 32.0, 0.0, 0.0, 0.1],
        [
            1024.0,
            1024.0,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE,
            1.0,
            0.1,
        ],
        [1024.0, 1024.0, f64::MAX, 1.0, f64::MIN_POSITIVE, 0.1],
    ] {
        assert!(
            ReservoirParameters::from_native(
                values[0], values[1], values[2], values[3], values[4], values[5]
            )
            .is_err(),
            "geometry={values:?}"
        );
    }
    Ok(())
}

#[test]
fn identities_states_and_forcing_require_complete_explicit_support() -> Result {
    let p = parameters()?;
    let element = |id: &str| ReservoirElement {
        id: id.into(),
        parameters: p,
    };
    for elements in [
        vec![],
        vec![element("")],
        vec![element("bad\nidentity")],
        vec![element("duplicate"), element("duplicate")],
    ] {
        let states = vec![ReservoirState { depth_m: 0.0 }; elements.len()];
        assert!(ReservoirBatch::new(elements, states, 0.0, ReservoirConfig::default()).is_err());
    }
    for states in [
        vec![],
        vec![ReservoirState { depth_m: -1.0 }],
        vec![ReservoirState { depth_m: f64::NAN }],
    ] {
        assert!(
            ReservoirBatch::new(
                vec![element("one")],
                states,
                0.0,
                ReservoirConfig::default()
            )
            .is_err()
        );
    }
    for time in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            ReservoirBatch::new(
                vec![element("one")],
                vec![ReservoirState { depth_m: 0.0 }],
                time,
                ReservoirConfig::default()
            )
            .is_err()
        );
    }
    let batch = ReservoirBatch::new(
        vec![element("one")],
        vec![ReservoirState { depth_m: 0.0 }],
        0.0,
        ReservoirConfig::default(),
    )?;
    let before = batch.snapshot_data()?;
    for inputs in [
        vec![],
        vec![ReservoirInput {
            incoming_rate_m_per_s: -1.0,
        }],
        vec![ReservoirInput {
            incoming_rate_m_per_s: f64::NAN,
        }],
    ] {
        assert!(batch.trial_to(1.0, &inputs).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn numerical_controls_keep_root_and_volume_tolerances_independent() -> Result {
    let p = parameters()?;
    for config in [
        ReservoirConfig {
            root_absolute_depth: -1.0,
            ..ReservoirConfig::default()
        },
        ReservoirConfig {
            root_relative_depth: f64::NAN,
            ..ReservoirConfig::default()
        },
        ReservoirConfig {
            root_absolute_depth: 0.0,
            root_relative_depth: 0.0,
            ..ReservoirConfig::default()
        },
        ReservoirConfig {
            root_iterations: 0,
            ..ReservoirConfig::default()
        },
        ReservoirConfig {
            volume_absolute: f64::INFINITY,
            ..ReservoirConfig::default()
        },
        ReservoirConfig {
            volume_absolute: 0.0,
            volume_relative: 0.0,
            ..ReservoirConfig::default()
        },
    ] {
        assert!(single(p, 0.0, config).is_err());
    }
    let config = ReservoirConfig::default();
    assert_eq!(
        config.root_absolute_depth.to_bits(),
        f64::from_bits(1).to_bits()
    );
    assert_eq!(config.root_relative_depth, 64.0 * f64::EPSILON);
    assert_eq!(config.root_iterations, 512);
    assert_eq!(config.volume_absolute, 1e-9);
    assert_eq!(config.volume_relative, 1e-12);
    Ok(())
}
