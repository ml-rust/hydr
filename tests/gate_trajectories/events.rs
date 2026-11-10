use super::fixture::*;
use hydr::structures::*;

#[test]
fn original_events_keep_adjacent_times_common_support_and_right_segments() -> Result {
    let adjacent = 1.0_f64.next_up();
    let a = prescribed(&[(0.0, 0.5), (1.0, 0.5), (4.0, 0.5)], exact())?;
    let b = prescribed(&[(1.0, 0.5), (adjacent, 0.5), (3.0, 0.5)], exact())?;
    let batch = mixed()?.with_gate_trajectories(mixed_assignment(a, b))?;
    assert_eq!(
        batch.support(),
        GateTrajectorySupport {
            start_s: 1.0,
            end_s: 3.0
        }
    );
    let next = batch
        .next_event_after(1.0)?
        .ok_or("Missing adjacent original event")?;
    assert_eq!(next.time_s.to_bits(), adjacent.to_bits());
    assert_eq!(next.structure_ids, vec!["gate-1"]);
    assert_eq!(
        batch
            .next_event_after(adjacent)?
            .ok_or("Missing terminal event")?
            .time_s,
        3.0
    );
    assert!(batch.next_event_after(3.0)?.is_none());
    let interval = batch.check_interval(1.0, 3.0)?;
    assert_eq!(interval.interior_events, vec![next]);
    assert_eq!(
        interval.end_event,
        Some(GateTrajectoryEvent {
            time_s: 3.0,
            structure_ids: vec!["gate-1".into()]
        })
    );
    let start = batch.evaluate(1.0)?.outputs()?;
    assert_eq!(
        (
            start[0].element,
            start[0].segment_index,
            start[0].exact_knot_index
        ),
        (1, 1, Some(1))
    );
    assert_eq!(
        (
            start[1].element,
            start[1].segment_index,
            start[1].exact_knot_index
        ),
        (3, 0, Some(0))
    );
    let terminal = batch.evaluate(3.0)?.outputs()?;
    assert_eq!(terminal[0].exact_knot_index, None);
    assert_eq!(terminal[1].exact_knot_index, Some(2));
    assert_eq!(terminal[1].segment_index, 1);
    Ok(())
}

#[test]
fn point_and_interval_support_errors_remain_global_and_typed() -> Result {
    let batch = bind("binary_triangle", 1)?;
    for time in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        expect_error(batch.evaluate(time), "NonfiniteInput", None)?;
        expect_error(batch.next_event_after(time), "NonfiniteInput", None)?;
    }
    for time in [-1.0, 4.0_f64.next_up()] {
        expect_error(batch.evaluate(time), "UnsupportedRegime", None)?;
        expect_error(batch.next_event_after(time), "UnsupportedRegime", None)?;
    }
    for (start, end) in [
        (0.0, 0.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
    ] {
        expect_error(batch.check_interval(start, end), "Definition", None)?;
    }
    expect_error(batch.check_interval(-1.0, 1.0), "UnsupportedRegime", None)?;
    expect_error(batch.check_interval(0.0, 5.0), "UnsupportedRegime", None)?;
    Ok(())
}
