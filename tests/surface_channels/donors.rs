//! Donor limits: shared factors over native, land, and caller withdrawals.

use crate::fixture::Result;
use crate::scenario::{Batch, ReachSpec, Spec, depths, forcing, last, step, volumes, weir};
use hydr::channel::{ChannelLateralDirection, ChannelLateralInput};
use hydr::coupling::SurfaceChannelStepDiagnostics;
use hydr::forcing::WaterAmount;

const REACH: &str = "river";
/// Caller pump rate at channel cell 0, in m³/s. Over 0.05 s it reserves 2.9 of 3 m³.
const PUMP: f64 = 58.0;
const END: f64 = 0.05;

/// Fails unless every link carries the same donor factor, strictly between zero and one.
fn shared_factor(step: &SurfaceChannelStepDiagnostics) -> Result<f64> {
    let factor = step.links[0].donor_factor;
    for link in &step.links {
        assert_eq!(link.donor_factor.to_bits(), factor.to_bits(), "{step:?}");
        assert!(link.weir_volume_m3 != 0.0, "{link:?}");
    }
    if !(factor > 0.0 && factor < 1.0) {
        return Err(format!("the shared donor must limit the links: {step:?}").into());
    }
    Ok(factor)
}

/// A 1.5 m deep channel cell with a caller pump, spilling over 0.5 m crests onto dry cells.
fn pumped(surface_cells: usize) -> Spec {
    let mut spec = Spec::new(surface_cells, 1, 0.0, 0.0).reach(
        ReachSpec::new(REACH, 0.0, &[1.5, 1.5]).binding(
            "pump",
            0,
            ChannelLateralDirection::Withdrawal,
        ),
    );
    for cell in 0..surface_cells {
        spec = spec.link(weir(&format!("bank-{cell}"), cell, REACH, 0, 0.5, 20.0));
    }
    spec
}

fn pump_step(start: &Batch) -> Result<Batch> {
    let pump = ChannelLateralInput::Withdrawal {
        boundary_id: "pump".into(),
        amount: WaterAmount::Rate {
            rate_m3_per_s: PUMP,
        },
    };
    Ok(start
        .trial_to(END, &forcing(start, END, &[(REACH, pump)]))?
        .into_next())
}

/// Water withdrawn from channel cell 0 by every binding over the last interval.
fn withdrawn(batch: &Batch) -> Result<f64> {
    let reach = batch.reach(REACH).ok_or("missing reach")?;
    let receipts = reach
        .last_step()?
        .ok_or("missing reach step")?
        .lateral_receipts;
    Ok(receipts
        .iter()
        .filter(|r| r.cell_id == "cell-0")
        .map(|r| match r.input {
            ChannelLateralInput::Withdrawal { .. } => r.volume_m3,
            ChannelLateralInput::Inflow { .. } => 0.0,
        })
        .sum())
}

#[test]
fn competing_surface_cells_share_channel_donor() -> Result {
    let start = pumped(2).build()?;
    let donor = volumes(&start, REACH)?[0];
    let batch = pump_step(&start)?;
    let step = last(&batch)?;
    shared_factor(&step)?;
    let given: f64 = step.links.iter().map(|l| -l.signed_volume_m3).sum();
    let pumped = PUMP * (step.end_s - step.start_s);
    assert!(
        given > 0.0 && given + pumped <= donor,
        "given={given}, donor={donor}"
    );
    assert!((withdrawn(&batch)? - pumped - given).abs() <= 1e-12);
    assert!(volumes(&batch, REACH)?.iter().all(|v| *v >= 0.0));
    let received: f64 = depths(&batch)?.iter().map(|h| h * 4.0).sum();
    assert!((received - given).abs() <= 1e-12, "{received} vs {given}");
    Ok(())
}

#[test]
fn two_links_from_one_surface_cell_share_donor() -> Result {
    // Land withdrawal leaves the 0.5 m deep surface cell little water to spill into two cells.
    let mut spec = Spec::new(1, 1, 1.0, 0.5)
        .reach(ReachSpec::new(REACH, 0.0, &[0.5, 0.5]))
        .link(weir("east", 0, REACH, 0, 1.2, 20.0))
        .link(weir("west", 0, REACH, 1, 1.2, 20.0));
    spec.withdrawal = vec![4.9];
    let start = spec.build()?;
    let batch = step(&start, 0.1)?;
    let step = last(&batch)?;
    shared_factor(&step)?;
    let given: f64 = step.links.iter().map(|l| l.signed_volume_m3).sum();
    assert!(given > 0.0 && given <= 0.5 * 4.0, "{given}");
    assert!(depths(&batch)?[0] >= 0.0);
    let gained: f64 = volumes(&batch, REACH)?.iter().sum::<f64>() - 2.0;
    assert!((gained - given).abs() <= 1e-12, "{gained} vs {given}");
    Ok(())
}

#[test]
fn limited_withdrawals_pass_channel_combined_depletion() -> Result {
    let start = pumped(1).build()?;
    let donor = volumes(&start, REACH)?[0];
    let batch = pump_step(&start)?;
    let step = last(&batch)?;
    let link = &step.links[0];
    assert_eq!(
        step.end_s, END,
        "the channel accepts the full interval: {step:?}"
    );
    assert!(
        link.donor_factor < 1.0,
        "the reserved pump must limit the spill: {link:?}"
    );
    assert!(link.signed_volume_m3 < 0.0);
    let pumped = PUMP * END;
    let total = withdrawn(&batch)?;
    assert!(total <= donor, "withdrawn={total}, donor={donor}");
    assert!((total - pumped + link.signed_volume_m3).abs() <= 1e-12);
    let reach = batch.reach(REACH).ok_or("missing reach")?;
    let receipt = reach.last_step()?.ok_or("missing reach step")?;
    assert!(
        receipt.depletion_limit_s >= END,
        "{}",
        receipt.depletion_limit_s
    );
    assert!(volumes(&batch, REACH)?[0] >= 0.0);
    Ok(())
}
