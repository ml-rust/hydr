use super::boundary_common::{Side, TestResult, definition, initial, interval, reach};
use super::fixtures::{FIXTURES, reservoir};

#[test]
fn unsupported_reservoir_states_and_operands_reject_atomically() -> TestResult {
    for fixture in FIXTURES.into_iter().take(2) {
        let critical = match fixture.shape {
            super::boundary_common::Shape::Rectangle { .. } => 4.0 / 9.0,
            super::boundary_common::Shape::Triangle { .. } => 16.0 / 25.0,
        };
        for side in [Side::Left, Side::Right] {
            for (depth, velocity, elevation, loss) in [
                (0.0, 0.0, 1.0, 0.0),
                (1.0, side.sign(), 1.0, 0.0),
                (1.0, 2.0 * side.sign(), 1.0, 0.0),
                (1.0, 0.0, critical, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 100.0, 0.0),
                (1.0, 0.0, f64::NAN, 0.0),
                (1.0, 0.0, f64::INFINITY, 0.0),
                (1.0, 0.0, 1.0, -1.0),
                (1.0, 0.0, 1.0, f64::NAN),
                (1.0, 0.0, 1.0, f64::INFINITY),
            ] {
                let origin = reach(
                    definition(fixture.shape, 1, 128.0, 4.0, 1.0),
                    initial(fixture.shape, &[128.0], &[depth], &[velocity]),
                )?;
                let before = origin.snapshot_data()?;
                let dt = 1.0 / 1024.0;
                assert!(
                    origin
                        .step(dt, &interval(side, reservoir(elevation, loss), 0.0, dt))
                        .is_err()
                );
                assert_eq!(origin.snapshot_data()?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn compression_at_or_beyond_the_original_cap_rejects_without_approximation() -> TestResult {
    for fixture in FIXTURES.into_iter().skip(2) {
        let cap = fixture.depth;
        for side in [Side::Left, Side::Right] {
            let origin = reach(
                definition(fixture.shape, 1, 128.0, cap, 1.0),
                initial(fixture.shape, &[128.0], &[1.0], &[0.0]),
            )?;
            let before = origin.snapshot_data()?;
            for elevation in [fixture.elevation, fixture.elevation + 0.01] {
                let dt = 1.0 / 1024.0;
                assert!(
                    origin
                        .step(
                            dt,
                            &interval(side, reservoir(elevation, fixture.loss), 0.0, dt)
                        )
                        .is_err(),
                    "{} cap admission",
                    fixture.name
                );
                assert_eq!(origin.snapshot_data()?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn reservoir_head_controls_reject_invalid_and_empty_budgets() -> TestResult {
    let fixture = FIXTURES[0];
    let defaults = hydr::channel::ChannelConfig::default().boundary;
    assert_eq!(
        defaults.reservoir_head_absolute_m.to_bits(),
        1e-12_f64.to_bits()
    );
    assert_eq!(
        defaults.reservoir_head_relative.to_bits(),
        1e-12_f64.to_bits()
    );
    for (absolute, relative) in [
        (-1.0, 1e-12),
        (f64::NAN, 1e-12),
        (f64::INFINITY, 1e-12),
        (1e-12, -1.0),
        (1e-12, f64::NAN),
        (1e-12, f64::INFINITY),
        (0.0, 0.0),
    ] {
        let mut layout = definition(fixture.shape, 1, 128.0, 4.0, 1.0);
        layout.config.boundary.reservoir_head_absolute_m = absolute;
        layout.config.boundary.reservoir_head_relative = relative;
        assert!(reach(layout, initial(fixture.shape, &[128.0], &[1.0], &[0.0])).is_err());
    }
    for (absolute, relative) in [(0.0, 1e-12), (1e-12, 0.0)] {
        let mut layout = definition(fixture.shape, 1, 128.0, 4.0, 1.0);
        layout.config.boundary.reservoir_head_absolute_m = absolute;
        layout.config.boundary.reservoir_head_relative = relative;
        let origin = reach(layout, initial(fixture.shape, &[128.0], &[1.0], &[0.0]))?;
        let dt = 1.0 / 1024.0;
        origin.step(dt, &interval(Side::Right, fixture.boundary(), 0.0, dt))?;
    }
    Ok(())
}
