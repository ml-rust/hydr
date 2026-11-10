//! Complete channel continuation and checkpoint admission.

#[path = "support/runtime.rs"]
mod runtime;

use hydr::channel::{
    ChannelBoundary, ChannelBoundaryInterval, ChannelConfig, ChannelInitialState, ChannelReach,
    ChannelReachDefinition, RectangularChannelCell,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn reach() -> Result<ChannelReach<runtime::Runtime>, Box<dyn std::error::Error>> {
    Ok(ChannelReach::new(
        runtime::client(),
        ChannelReachDefinition {
            vertical_datum: "synthetic metres".into(),
            provenance: "rectangular restart fixture".into(),
            cells: (0..3)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 2.0,
                    width_m: 2.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.03,
                })
                .collect(),
            config: ChannelConfig::default(),
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![4.0, 1.0, 2.0],
            longitudinal_momentum_m4_per_s: vec![0.8, -0.1, 0.0],
        },
    )?)
}

fn advance(
    mut reach: ChannelReach<runtime::Runtime>,
    steps: usize,
) -> Result<ChannelReach<runtime::Runtime>, Box<dyn std::error::Error>> {
    for _ in 0..steps {
        let time = reach.time();
        reach = reach
            .step(
                time + 0.02,
                &ChannelBoundaryInterval {
                    start_s: time,
                    end_s: time + 1.0,
                    left: ChannelBoundary::Wall,
                    right: ChannelBoundary::Wall,
                },
            )?
            .next;
    }
    Ok(reach)
}

#[test]
fn trusted_and_cold_restart_reproduce_identical_continuation() -> TestResult {
    let origin = reach()?;
    let current = advance(origin.clone(), 5)?;
    let trusted = origin.restore(&current.snapshot())?;
    let cold = origin.import_snapshot(&current.snapshot_data()?)?;
    assert_eq!(trusted.snapshot_data()?, current.snapshot_data()?);
    assert_eq!(cold.snapshot_data()?, current.snapshot_data()?);
    let expected = advance(current, 7)?.snapshot_data()?;
    assert_eq!(advance(trusted, 7)?.snapshot_data()?, expected);
    assert_eq!(advance(cold, 7)?.snapshot_data()?, expected);
    assert_eq!(origin.accepted_steps(), 0);
    Ok(())
}

#[test]
fn zero_step_import_requires_original_state_and_zero_counters() -> TestResult {
    let origin = reach()?;
    let data = origin.snapshot_data()?;
    assert_eq!(origin.import_snapshot(&data)?.snapshot_data()?, data);
    let mut altered = data.clone();
    altered.volume_m3[0] += 0.5;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = data.clone();
    altered.ledger.boundary_in_m3 = 1.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = data.clone();
    altered.time_s = 1.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = data.clone();
    altered.ledger.numerical_transfer_underflow_bound_m3 = f64::from_bits(1);
    assert!(origin.import_snapshot(&altered).is_err());
    Ok(())
}

#[test]
fn cold_import_rejects_altered_bounded_continuation_fields() -> TestResult {
    let origin = reach()?;
    let current = advance(origin.clone(), 3)?;
    let data = current.snapshot_data()?;
    let unchanged = origin.snapshot_data()?;
    for mutation in 0..29 {
        let mut altered = data.clone();
        let last = altered
            .last_interval
            .as_mut()
            .ok_or("missing fixture interval")?;
        match mutation {
            0 => altered.volume_m3[0] = f64::from_bits(altered.volume_m3[0].to_bits() + 1),
            1 => altered.longitudinal_momentum_m4_per_s[0] += 1.0,
            2 => altered.time_s += 0.001,
            3 => altered.accepted_steps += 1,
            4 => last.previous_volume_m3[0] = f64::NAN,
            5 => last.previous_ledger.boundary_out_m3 = -1.0,
            6 => last.previous_time_s = altered.time_s,
            7 => last.forcing.start_s += 0.001,
            8 => last.requested_end_s = last.previous_time_s,
            9 => last.diagnostics.face_volume_m3[1] += 0.001,
            10 => last.diagnostics.cfl_limit_s *= 0.5,
            11 => {
                last.previous_accepted_steps = usize::MAX;
                altered.accepted_steps = usize::MAX;
            }
            12 => {
                last.diagnostics.face_volume_m3.pop();
            }
            13 => last.diagnostics.friction_impulse_m4_per_s[0] = f64::NAN,
            14 => altered.ledger.initial_storage_m3 += 1.0,
            15 => last.diagnostics.max_local_momentum_residual_m4_per_s = f64::NAN,
            16 => last.diagnostics.max_momentum_underflow_bound_m4_per_s = f64::NAN,
            17 => {
                last.diagnostics.momentum_underflow_bound_m4_per_s.pop();
            }
            18 => last.diagnostics.momentum_underflow_bound_m4_per_s[0] = f64::from_bits(1),
            19 => altered.ledger.numerical_transfer_underflow_bound_m3 = -1.0,
            20 => altered.ledger.numerical_transfer_underflow_bound_m3 = f64::NAN,
            21 => {
                last.previous_ledger.numerical_transfer_underflow_bound_m3 = 1.0;
            }
            22 => last.diagnostics.transfer_underflow_bound_m3 = f64::NAN,
            23 => {
                last.diagnostics.face_transfer_underflow_bound_m3.pop();
            }
            24 => last.diagnostics.face_transfer_underflow_bound_m3[0] = -1.0,
            25 => last.diagnostics.face_transfer_underflow_bound_m3[0] = f64::from_bits(1),
            26 => last.diagnostics.face_transfer_underflow_bound_m3[0] = f64::NAN,
            27 => {
                altered.ledger.numerical_transfer_underflow_bound_m3 = f64::from_bits(1);
            }
            28 => last.previous_ledger.numerical_transfer_underflow_bound_m3 = -1.0,
            _ => return Err("unrecognized fixture mutation".into()),
        }
        assert!(
            origin.import_snapshot(&altered).is_err(),
            "mutation {mutation}"
        );
        assert_eq!(origin.snapshot_data()?, unchanged);
    }
    Ok(())
}

#[test]
fn snapshot_identity_preserves_signed_zero_geometry_and_epoch() -> TestResult {
    let origin = reach()?;
    let mut altered = origin.snapshot_data()?;
    altered.definition.cells[0].bed_elevation_m = -0.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = origin.snapshot_data()?;
    altered.initial.time_s = -0.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = origin.snapshot_data()?;
    altered.initial.longitudinal_momentum_m4_per_s[2] = -0.0;
    assert!(origin.import_snapshot(&altered).is_err());
    Ok(())
}
