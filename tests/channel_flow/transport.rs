use super::common::{TestResult, advance, cell, check_water, definition, exterior, reach, walls};
use hydr::channel::ChannelBoundaryInterval;

#[test]
fn uniform_flow_preserves_depth_and_discharge_in_both_directions() -> TestResult {
    for velocity in [-1.0, 1.0] {
        let cells = (0..20).map(|i| cell(i, 5.0, 2.0, 0.0)).collect();
        let initial = reach(definition(cells, 1.0), &[1.0; 20], &[2.0 * velocity; 20])?;
        let result = advance(
            initial,
            5.0,
            [exterior(1.0, velocity), exterior(1.0, velocity)],
        )?;
        for cell in result.state()? {
            assert!((cell.depth_m - 1.0).abs() <= 1e-12);
            assert!((cell.discharge_m3_per_s - 2.0 * velocity).abs() <= 1e-11);
        }
        let ledger = result.ledger();
        assert!((ledger.boundary_in_m3 - 10.0).abs() <= 1e-11);
        assert!((ledger.boundary_out_m3 - 10.0).abs() <= 1e-11);
    }
    Ok(())
}

#[test]
fn partially_dry_lake_stays_balanced_across_width_and_bed_steps() -> TestResult {
    let cells = vec![
        cell(0, 1.0, 1.0, 0.0),
        cell(1, 1.0, 3.0, 2.0),
        cell(2, 1.0, 2.0, 0.0),
    ];
    let initial = reach(definition(cells, 1.0), &[1.5, 0.0, 1.5], &[0.0; 3])?;
    let result = advance(initial, 2.0, walls())?;
    for (cell, depth) in result.state()?.iter().zip([1.5, 0.0, 1.5]) {
        assert!((cell.depth_m - depth).abs() <= 1e-11);
        assert!(cell.discharge_m3_per_s.abs() <= 1e-11);
    }
    assert!(result.state()?[1].dry);
    let diagnostics = result.last_step()?.ok_or("missing accepted step")?;
    assert!(
        diagnostics
            .face_volume_m3
            .iter()
            .all(|&volume| volume == 0.0)
    );
    assert_eq!(result.ledger().boundary_in_m3, 0.0);
    assert_eq!(result.ledger().boundary_out_m3, 0.0);
    Ok(())
}

#[test]
fn wet_lake_balances_pressure_across_width_and_bed_steps() -> TestResult {
    let cells = vec![
        cell(0, 1.0, 1.0, 0.0),
        cell(1, 1.0, 3.0, 0.5),
        cell(2, 1.0, 2.0, 1.0),
    ];
    let result = advance(
        reach(definition(cells, 1.0), &[2.0, 1.5, 1.0], &[0.0; 3])?,
        2.0,
        walls(),
    )?;
    for (cell, depth) in result.state()?.iter().zip([2.0, 1.5, 1.0]) {
        assert!((cell.depth_m - depth).abs() <= 1e-11);
        assert!(cell.discharge_m3_per_s.abs() <= 1e-11);
    }
    let diagnostics = result.last_step()?.ok_or("missing accepted step")?;
    // Wet pressure cancellation permits roundoff, unlike exact dry-bank isolation.
    assert!(
        diagnostics
            .face_volume_m3
            .iter()
            .all(|volume| volume.abs() <= 1e-11)
    );
    Ok(())
}

#[test]
fn water_overtops_a_dry_raised_cell_without_losing_storage() -> TestResult {
    let cells = vec![
        cell(0, 1.0, 1.0, 0.0),
        cell(1, 1.0, 3.0, 2.0),
        cell(2, 1.0, 2.0, 0.0),
    ];
    let initial = reach(definition(cells, 1.0), &[2.5, 0.0, 0.0], &[0.0; 3])?;
    let result = advance(initial, 0.05, walls())?;
    assert!(result.state()?[1].volume_m3 > 0.0);
    assert!((result.ledger().storage_m3 - 2.5).abs() <= 1e-9 + 2.5e-12);
    Ok(())
}

#[test]
fn large_width_ratios_shorten_steps_before_depleting_narrow_cells() -> TestResult {
    let cells = vec![
        cell(0, 1.0, 1e-3, 0.0),
        cell(1, 1.0, 1e3, 0.0),
        cell(2, 1.0, 1e-3, 0.0),
    ];
    let initial = reach(definition(cells, 1.0), &[1.0, 0.5, 1.0], &[0.0; 3])?;
    let forcing = ChannelBoundaryInterval {
        start_s: 0.0,
        end_s: 0.1,
        left: walls()[0].clone(),
        right: walls()[1].clone(),
    };
    let next = initial.step(0.1, &forcing)?.next;
    assert!(next.time() < 0.1);
    check_water(&next)?;
    let step = next.last_step()?.ok_or("missing accepted step")?;
    assert!(step.retries > 0);
    assert!(next.time() <= step.cfl_limit_s);
    assert!(next.time() <= step.depletion_limit_s);
    assert!((next.ledger().storage_m3 - 500.002).abs() <= 1e-9 + 500.002e-12);
    Ok(())
}
