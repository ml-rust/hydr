//! Public soil lifecycle, native branch, unit bridge, and batch parity checks.

use hydr::soil::{
    ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, NativeSoilInput, NativeSoilState,
    SoilInput, SoilOutput,
};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn parameters() -> Result<ModifiedGreenAmptParameters> {
    Ok(ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?)
}

fn state(saturated: bool) -> NativeSoilState {
    NativeSoilState {
        imd: 0.2,
        f_ft: 0.01,
        fu_ft: 0.02,
        saturated,
        t_s: 5.0,
    }
}

fn input(rate: f64, depth: f64) -> NativeSoilInput {
    NativeSoilInput {
        incoming_rate_ft_per_s: rate,
        ponded_depth_ft: depth,
    }
}

fn run(
    batch: &ModifiedGreenAmptBatch,
    dt: f64,
    inputs: &[NativeSoilInput],
) -> Result<(ModifiedGreenAmptBatch, Vec<SoilOutput>)> {
    let trial = batch.trial_native(dt, inputs)?;
    let outputs = trial.outputs()?;
    Ok((trial.into_next(), outputs))
}

fn close(actual: f64, expected: f64, absolute: f64) {
    assert!(
        (actual - expected).abs() <= absolute + 1e-11 * actual.abs().max(expected.abs()),
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}

#[test]
fn checked_parameters_retain_source_order_and_explicit_initialization() -> Result {
    let p = parameters()?;
    let expected = 4.0 * (1e-5_f64 * 12.0 * 3600.0).sqrt() / 12.0;
    assert_eq!(p.upper_zone_depth_ft().to_bits(), expected.to_bits());
    let batch = ModifiedGreenAmptBatch::initialized(&[p])?;
    assert_eq!(
        batch.native_states()?,
        vec![NativeSoilState {
            imd: 0.3,
            f_ft: 0.0,
            fu_ft: 0.0,
            saturated: false,
            t_s: 0.0
        }]
    );
    let si = ModifiedGreenAmptParameters::from_si(0.4 * 0.3048, 1e-5 * 0.3048, 0.3)?;
    close(si.suction_ft(), p.suction_ft(), 1e-15);
    close(si.conductivity_ft_per_s(), p.conductivity_ft_per_s(), 1e-20);
    for (s, k, d) in [
        (f64::NAN, 1e-5, 0.3),
        (-1.0, 1e-5, 0.3),
        (0.4, 0.0, 0.3),
        (0.4, 1e-5, 1.1),
        (0.4, f64::MAX, 0.3),
        (0.4, f64::from_bits(1), 0.3),
        (0.4, 1e-5, f64::from_bits(1)),
    ] {
        assert!(ModifiedGreenAmptParameters::from_native(s, k, d).is_err());
    }
    assert!(ModifiedGreenAmptBatch::initialized(&[]).is_err());
    Ok(())
}

#[test]
fn exact_recovery_ceiling_admits_endpoint_and_rejects_next_float() -> Result {
    let p = parameters()?;
    let ceiling = p
        .maximum_deficit()
        .max(p.capacity_ft() / p.upper_zone_depth_ft());
    assert_eq!(p.recovered_deficit_ceiling().to_bits(), ceiling.to_bits());
    let endpoint = NativeSoilState {
        imd: ceiling,
        f_ft: -0.01,
        fu_ft: p.capacity_ft(),
        saturated: true,
        t_s: -20.0,
    };
    ModifiedGreenAmptBatch::from_native_states(&[p], &[endpoint])?;
    let beyond = NativeSoilState {
        imd: ceiling.next_up(),
        ..endpoint
    };
    assert!(ModifiedGreenAmptBatch::from_native_states(&[p], &[beyond]).is_err());
    assert!(
        ModifiedGreenAmptBatch::from_native_states(
            &[p],
            &[NativeSoilState {
                fu_ft: p.capacity_ft().next_up(),
                ..endpoint
            }]
        )
        .is_err()
    );
    let tiny = ModifiedGreenAmptParameters::from_native(0.0, 1e-5, f64::MIN_POSITIVE)?;
    assert!(tiny.capacity_ft() > 0.0 && tiny.capacity_ft() < f64::MIN_POSITIVE);
    let initial = NativeSoilState {
        imd: tiny.maximum_deficit(),
        f_ft: -1.0,
        fu_ft: tiny.capacity_ft() / 2.0,
        saturated: false,
        t_s: 0.0,
    };
    let batch = ModifiedGreenAmptBatch::from_native_states(&[tiny], &[initial])?;
    let (next, _) = run(&batch, 1.0, &[input(0.0, 0.0)])?;
    assert!(next.native_states()?[0].imd <= tiny.recovered_deficit_ceiling());
    Ok(())
}

#[test]
fn saturated_dry_intervals_preserve_independent_memory() -> Result {
    let p = parameters()?;
    let initial = state(true);
    let batch = ModifiedGreenAmptBatch::from_native_states(&[p], &[initial])?;
    let (next, outputs) = run(&batch, 60.0, &[input(0.0, 0.0)])?;
    assert_eq!(outputs[0].native_rate_ft_per_s, 0.0);
    assert_eq!(
        next.native_states()?,
        vec![NativeSoilState {
            t_s: -55.0,
            ..initial
        }]
    );
    assert_eq!(batch.native_states()?, vec![initial]);
    Ok(())
}

#[test]
fn initialized_sequence_reaches_negative_event_memory() -> Result {
    let mut batch = ModifiedGreenAmptBatch::initialized(&[parameters()?])?;
    for (dt, rate) in [(60.0, 5e-6), (1.0, 0.0), (0.001, 1e-3), (60.0, 0.0)] {
        batch = run(&batch, dt, &[input(rate, 0.0)])?.0;
    }
    let final_state = batch.native_states()?[0];
    assert!(final_state.f_ft < 0.0 && final_state.fu_ft > 0.0 && final_state.t_s > 0.0);
    Ok(())
}

#[test]
fn legal_signed_event_logarithm_uses_native_minimum_fallback() -> Result {
    let p = ModifiedGreenAmptParameters::from_native(0.005, 1e-5, 0.3)?;
    let initial = NativeSoilState {
        f_ft: -0.01,
        ..state(true)
    };
    let batch = ModifiedGreenAmptBatch::from_native_states(&[p], &[initial])?;
    let (next, outputs) = run(&batch, 60.0, &[input(0.001, 0.0)])?;
    let minimum = initial.f_ft + 1e-5 * 60.0;
    let df = minimum - initial.f_ft;
    assert_eq!(
        outputs[0].native_rate_ft_per_s.to_bits(),
        (df / 60.0).to_bits()
    );
    assert_eq!(
        next.native_states()?[0].f_ft.to_bits(),
        (initial.f_ft + df).to_bits()
    );
    assert!(next.native_states()?[0].saturated);
    Ok(())
}

#[test]
fn strict_saturation_supply_and_recovery_timer_boundaries() -> Result {
    let p = ModifiedGreenAmptParameters::from_native(0.75, 1.0 / 1024.0, 0.25)?;
    let fs = 3.0 / 16.0;
    let initial = NativeSoilState {
        imd: 0.25,
        f_ft: fs,
        fu_ft: fs,
        saturated: true,
        t_s: 0.0,
    };
    let batch = ModifiedGreenAmptBatch::from_native_states(&[p], &[initial])?;
    let rate = 2.0_f64 / 1024.0;
    for (rain, saturated) in [
        (rate.next_down(), false),
        (rate, true),
        (rate.next_up(), true),
    ] {
        let (next, _) = run(&batch, 1.0, &[input(rain, 0.0)])?;
        assert_eq!(next.native_states()?[0].saturated, saturated);
    }
    let p = parameters()?;
    for timer in [60.0_f64.next_down(), 60.0, 60.0_f64.next_up()] {
        let batch = ModifiedGreenAmptBatch::from_native_states(
            &[p],
            &[NativeSoilState {
                t_s: timer,
                ..state(false)
            }],
        )?;
        let next = run(&batch, 60.0, &[input(0.0, 0.0)])?.0.native_states()?[0];
        assert_eq!(next.f_ft == 0.0, timer <= 60.0);
    }
    Ok(())
}

#[test]
fn availability_conductivity_and_positive_ponding_thresholds_preserve_branches() -> Result {
    let p = parameters()?;
    let initial = NativeSoilState {
        t_s: 1000.0,
        ..state(false)
    };
    let batch = ModifiedGreenAmptBatch::from_native_states(&[p], &[initial])?;
    let zero = 1e-10_f64;
    for (rain, dry) in [
        (zero.next_down(), true),
        (zero, false),
        (zero.next_up(), false),
    ] {
        let (next, outputs) = run(&batch, 60.0, &[input(rain, 0.0)])?;
        assert_eq!(outputs[0].native_rate_ft_per_s == 0.0, dry);
        assert_eq!(next.native_states()?[0].f_ft < initial.f_ft, dry);
    }
    for (factor, dry) in [(0.5, true), (2.0, false)] {
        let (next, outputs) = run(&batch, 60.0, &[input(0.0, factor * zero * 60.0)])?;
        assert_eq!(outputs[0].native_rate_ft_per_s == 0.0, dry);
        assert_eq!(next.native_states()?[0].f_ft < initial.f_ft, dry);
    }
    for (rain, renewed) in [
        (1e-5_f64.next_down(), false),
        (1e-5, false),
        (1e-5_f64.next_up(), true),
    ] {
        let next = run(&batch, 60.0, &[input(rain, 0.0)])?.0.native_states()?[0];
        assert_eq!(next.t_s > 1000.0, renewed);
    }
    Ok(())
}

#[test]
fn immutable_trials_restart_and_si_bridge_preserve_complete_state() -> Result {
    let p = parameters()?;
    let original = ModifiedGreenAmptBatch::initialized(&[p])?;
    let before = original.native_states()?;
    let (first, out) = run(&original, 60.0, &[input(1e-3, 0.0)])?;
    let (repeat, repeat_out) = run(&original, 60.0, &[input(1e-3, 0.0)])?;
    assert_eq!(out, repeat_out);
    assert_eq!(first.native_states()?, repeat.native_states()?);
    assert_eq!(original.native_states()?, before);
    let restored = ModifiedGreenAmptBatch::from_native_states(&[p], &first.native_states()?)?;
    let mut continued = first;
    let mut restarted = restored;
    for (dt, rate, depth) in [(600.0, 2e-4, 0.1), (3600.0, 0.0, 0.0), (60.0, 5e-6, 0.0)] {
        let a = run(&continued, dt, &[input(rate, depth)])?;
        let b = run(&restarted, dt, &[input(rate, depth)])?;
        assert_eq!(a.0.native_states()?, b.0.native_states()?);
        assert_eq!(a.1, b.1);
        continued = a.0;
        restarted = b.0;
    }
    let native = original.trial_native(60.0, &[input(5e-6, 0.01)])?;
    let si = original.trial(
        60.0,
        &[SoilInput {
            incoming_rate_m_per_s: 5e-6 * 0.3048,
            ponded_depth_m: 0.01 * 0.3048,
        }],
    )?;
    let n = native.outputs()?[0];
    let s = si.outputs()?[0];
    close(n.native_rate_ft_per_s, s.native_rate_ft_per_s, 1e-14);
    assert_eq!(
        n.rate_m_per_s.to_bits(),
        (n.native_rate_ft_per_s * 0.3048).to_bits()
    );
    assert_eq!(n.depth_m.to_bits(), (n.rate_m_per_s * 60.0).to_bits());
    Ok(())
}

#[test]
fn mixed_simd_and_tail_batches_match_independent_components() -> Result {
    let p = parameters()?;
    let signed = ModifiedGreenAmptParameters::from_native(0.005, 1e-5, 0.3)?;
    let seed = [
        (p, state(false), input(0.0, 0.0)),
        (p, state(true), input(0.0, 0.0)),
        (p, state(false), input(1e-5, 0.0)),
        (
            signed,
            NativeSoilState {
                f_ft: -0.01,
                ..state(true)
            },
            input(1e-3, 0.0),
        ),
        (p, state(true), input(1e-3, 0.1)),
        (p, state(false), input(1e-3, 0.0)),
    ];
    for length in [31, 32, 33, 64, 65] {
        for offset in 0..seed.len() {
            let lanes: Vec<_> = (0..length)
                .map(|i| seed[(i + offset) % seed.len()])
                .collect();
            let params: Vec<_> = lanes.iter().map(|x| x.0).collect();
            let states: Vec<_> = lanes.iter().map(|x| x.1).collect();
            let inputs: Vec<_> = lanes.iter().map(|x| x.2).collect();
            let batch = ModifiedGreenAmptBatch::from_native_states(&params, &states)?;
            let (next, outputs) = run(&batch, 60.0, &inputs)?;
            let next_states = next.native_states()?;
            for (i, (p, state, input)) in lanes.into_iter().enumerate() {
                let one = ModifiedGreenAmptBatch::from_native_states(&[p], &[state])?;
                let (single, out) = run(&one, 60.0, &[input])?;
                let expected = single.native_states()?[0];
                let actual = next_states[i];
                close(actual.imd, expected.imd, 1e-12);
                close(actual.f_ft, expected.f_ft, 1e-12);
                close(actual.fu_ft, expected.fu_ft, 1e-12);
                close(actual.t_s, expected.t_s, 1e-9);
                assert_eq!(actual.saturated, expected.saturated);
                close(
                    outputs[i].native_rate_ft_per_s,
                    out[0].native_rate_ft_per_s,
                    1e-14,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn rejected_trial_preserves_all_state_fields_and_all_lanes() -> Result {
    let p = parameters()?;
    let states = [state(false), state(true)];
    let batch = ModifiedGreenAmptBatch::from_native_states(&[p, p], &states)?;
    for inputs in [
        vec![input(1e-3, 0.0)],
        vec![input(1e-3, 0.0), input(f64::NAN, 0.0)],
        vec![input(1e-3, 0.0), input(0.0, -1.0)],
        vec![input(1e-3, 0.0), input(-1.0, 0.0)],
        vec![input(1e-3, 0.0), input(f64::MAX, f64::MAX)],
    ] {
        assert!(batch.trial_native(60.0, &inputs).is_err());
        assert_eq!(batch.native_states()?, states);
    }
    assert!(
        batch
            .trial(
                60.0,
                &[SoilInput {
                    incoming_rate_m_per_s: -1.0,
                    ponded_depth_m: 0.0
                }; 2]
            )
            .is_err()
    );
    assert_eq!(batch.native_states()?, states);
    assert!(
        batch
            .trial_native(f64::from_bits(1), &[input(1e-10, 0.0); 2])
            .is_err()
    );
    assert!(
        batch
            .trial(
                60.0,
                &[SoilInput {
                    incoming_rate_m_per_s: f64::MAX,
                    ponded_depth_m: 0.0,
                }; 2]
            )
            .is_err()
    );
    assert_eq!(batch.native_states()?, states);
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(batch.trial_native(dt, &[input(1e-3, 0.0); 2]).is_err());
        assert_eq!(batch.native_states()?, states);
    }
    Ok(())
}

#[test]
fn native_restart_preserves_signed_zero_memory_bits() -> Result {
    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.0)?;
    let initial = NativeSoilState {
        imd: -0.0,
        f_ft: -0.0,
        fu_ft: -0.0,
        saturated: false,
        t_s: -0.0,
    };
    let bits = |s: NativeSoilState| {
        [
            s.imd.to_bits(),
            s.f_ft.to_bits(),
            s.fu_ft.to_bits(),
            s.t_s.to_bits(),
        ]
    };
    let original = ModifiedGreenAmptBatch::from_native_states(&[p], &[initial])?;
    assert_eq!(bits(original.native_states()?[0]), bits(initial));
    let next = run(&original, 1.0, &[input(-0.0, -0.0)])?.0;
    let expected = NativeSoilState {
        t_s: -1.0,
        ..initial
    };
    assert_eq!(bits(next.native_states()?[0]), bits(expected));
    assert!(!next.native_states()?[0].saturated);
    let restarted = ModifiedGreenAmptBatch::from_native_states(&[p], &next.native_states()?)?;
    let continued = run(&next, 1.0, &[input(0.0, 0.0)])?;
    let restored = run(&restarted, 1.0, &[input(0.0, 0.0)])?;
    assert_eq!(
        bits(continued.0.native_states()?[0]),
        bits(restored.0.native_states()?[0])
    );
    assert_eq!(continued.1, restored.1);
    assert_eq!(bits(original.native_states()?[0]), bits(initial));
    Ok(())
}
