use super::fixture::{Result, orifice_case};
use hydr::coupling::CoupledLevelPoolBatch;

#[test]
fn temporal_refinement_matches_the_independent_midpoint_recurrence() -> Result {
    let continuous_transfer = 2.0_f64.sqrt() * 0.5 - 0.5 * 0.5_f64.powi(2);
    let references = [
        (0.5, 0.5930703308172536),
        (0.25, 0.5848635432250342),
        (0.125, 0.5827969783066328),
        (0.0625, 0.5822793936097672),
        (0.03125, 0.582149938242414),
    ];
    let mut previous_error = None;
    for (dt, discrete_reference) in references {
        let (mut setup, initial) = orifice_case([4.0, 2.0])?;
        setup.config.maximum_step_s = dt;
        let mut batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            setup,
            initial,
        )?;
        while batch.time() < 0.5 {
            let target = (batch.time() + dt).min(0.5);
            let next = batch.trial_to(target)?.into_next();
            assert_eq!(next.time(), target, "Unexpected shortened refinement step");
            batch = next;
        }
        let volumes = batch.pools().states()?;
        assert_eq!(volumes.len(), 2);
        let transferred = 4.0 - volumes[0].volume_m3;
        assert!(
            (transferred - discrete_reference).abs() <= 1e-8,
            "dt={dt}, transfer={transferred:.17e}, exact discrete={discrete_reference:.17e}"
        );
        let error = (transferred - continuous_transfer).abs();
        assert!(
            error <= 0.06 * dt * dt,
            "dt={dt}, error={error:.17e}, envelope={:.17e}",
            0.06 * dt * dt
        );
        if let Some(previous) = previous_error {
            let ratio = previous / error;
            assert!(
                (3.8..=4.2).contains(&ratio),
                "dt={dt}, refinement ratio={ratio}"
            );
        }
        previous_error = Some(error);
        assert!((volumes[0].volume_m3 + volumes[1].volume_m3 - 6.0).abs() <= 1e-9);
    }
    assert!(previous_error.ok_or("Missing finest refinement")? < 1e-4);
    Ok(())
}
