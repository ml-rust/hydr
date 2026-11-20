//! A reach-end port and a bank weir drawing on one donor cell.
//!
//! Supercritical flow runs down `alpha` toward its left end and drains through the west port
//! onto a dry floodplain 1 m below. A long bank weir on the same end cell spills onto a dry
//! surface cell away from the port. The weir demand exceeds the end cell's water net of port
//! outflow, so the donor factor binds.

use crate::fixture::{Result, applicability};
use crate::ports_fixture::{
    ALPHA, Batch, GRAVITY, Reach, Scenario, balanced, last, reach_cells, west,
};
use hydr::channel::{ChannelBoundary, ChannelEnd};
use hydr::coupling::{SurfaceChannelLink, SurfaceChannelStepDiagnostics};

/// Depth of the inflowing reach water, in m.
const DEPTH: f64 = 0.5;

/// Bed of `alpha` and crest of the bank weir, in m. The surface bed lies 1 m below.
const ALPHA_BED: f64 = 1.0;

/// Accepted intervals of the run.
const INTERVALS: usize = 3;

/// Relative bound on the donor water the link and the port leave unused.
const NET: f64 = 1e-12;

/// The case and its batch: short `alpha` cells, supercritical inflow at the right end.
pub(crate) fn shared(surveyed: bool) -> Result<(Scenario, Batch)> {
    let velocity = -2.0 * (GRAVITY * DEPTH).sqrt();
    let mut case = Scenario::rest(0.0, 0.0, ALPHA_BED, vec![west("p")]).with_surveyed(surveyed);
    case.link = false;
    case.alpha = Reach::uniform(ALPHA_BED, DEPTH, DEPTH * velocity);
    case.alpha_size = [4.0, 0.125];
    case.beta = Reach::rest(0.25, 0.75);
    case.ends = vec![(
        ALPHA,
        ChannelEnd::Right,
        ChannelBoundary::Exterior {
            depth_m: DEPTH,
            velocity_m_per_s: velocity,
        },
    )];
    let (mut setup, initial) = case.parts()?;
    setup.links.push(SurfaceChannelLink {
        id: "bank".into(),
        surface_cell: 3,
        reach_id: ALPHA.into(),
        channel_cell: 0,
        crest_elevation_m: ALPHA_BED,
        crest_length_m: 100.0,
        weir_coefficient: 1.7,
        applicability: applicability(),
        reach_tangent: [1.0, 0.0],
        mapping_evidence: "Surface cell 3 borders the reach end cell".into(),
    });
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    Ok((case, batch))
}

/// Fails unless each side's net link exchange pairs with the summed link volumes.
fn links_pair(step: &SurfaceChannelStepDiagnostics) -> Result {
    let signed: f64 = step.links.iter().map(|link| link.signed_volume_m3).sum();
    let surface = step.surface_exchange_in_m3 - step.surface_exchange_out_m3;
    let channel = step.channel_exchange_in_m3 - step.channel_exchange_out_m3;
    if (surface + signed).abs() <= step.paired_bound_m3
        && (channel - signed).abs() <= step.paired_bound_m3
    {
        Ok(())
    } else {
        Err(format!("link pairing: {step:?}").into())
    }
}

#[test]
fn the_donor_factor_shares_the_end_cell_water_net_of_port_outflow() -> Result {
    donor_factor(false)
}

/// The shared donor cell on the rectangular reach, or on its surveyed rectangle.
pub(crate) fn donor_factor(surveyed: bool) -> Result {
    let (case, mut batch) = shared(surveyed)?;
    for _ in 0..INTERVALS {
        let held = reach_cells(&batch, ALPHA)?[0].volume_m3;
        batch = case.step(&batch, 1.0)?;
        let step = last(&batch)?;
        let port = &step.ports[0];
        let link = &step.links[0];
        // Both draw on the end cell: the port onto the floodplain, the weir onto cell 3.
        let port_out = -port.signed_volume_m3;
        let link_out = -link.signed_volume_m3;
        assert!(port_out > 0.0 && link_out > 0.0, "{step:?}");
        assert!(
            link.donor_factor > 0.0 && link.donor_factor < 1.0,
            "{link:?}"
        );
        // The weir takes the end cell's water net of the port outflow, and no more.
        assert!(link_out + port_out <= held, "{step:?}");
        assert!(held - port_out - link_out <= NET * held, "{step:?}");
        assert!(port.paired_gap_m3.abs() <= port.paired_bound_m3, "{port:?}");
        links_pair(&step)?;
        assert!(step.mass_residual_m3.abs() <= step.mass_bound_m3);
        assert!(
            reach_cells(&batch, ALPHA)?
                .iter()
                .all(|cell| cell.volume_m3 >= 0.0)
        );
        assert!(
            batch
                .surface()
                .state()?
                .cells()
                .iter()
                .all(|cell| cell.h >= 0.0)
        );
        balanced(&batch)?;
    }
    let ledgers = batch.port_ledgers()?;
    assert!(ledgers[0].channel_to_surface_m3 > 0.0 && ledgers[0].surface_to_channel_m3 == 0.0);
    let links = batch.link_ledgers()?;
    assert!(links[0].channel_to_surface_m3 > 0.0 && links[0].surface_to_channel_m3 == 0.0);
    Ok(())
}
