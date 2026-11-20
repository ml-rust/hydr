//! Port, face, link, and reach declaration order change no accepted bit.

use crate::fixture::Result;
use crate::ports_fixture::{
    ALPHA, BETA, Batch, Reach, Scenario, bank, east, forcing, reach_cells, west,
};
use hydr::coupling::{SurfaceChannelInitialState, SurfaceChannelLink, SurfaceChannelSetup};

/// Accepted intervals of each run.
const INTERVALS: usize = 12;

/// Bank weir between surface cell 5 and `beta` cell 1.
fn beta_bank() -> SurfaceChannelLink {
    SurfaceChannelLink {
        id: "bank-beta".into(),
        surface_cell: 5,
        reach_id: BETA.into(),
        channel_cell: 1,
        ..bank()
    }
}

/// Two ports on `alpha`, two bank links, and a reach above its floodplain.
fn parts(surveyed: bool) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let mut case =
        Scenario::rest(1.0, 1.6, 0.5, vec![west("p-west"), east("p-east")]).with_surveyed(surveyed);
    case.alpha = Reach::rest(0.5, 1.75);
    let (mut setup, initial) = case.parts()?;
    setup.links.push(beta_bank());
    Ok((setup, initial))
}

/// The same case with every declaration list reversed, port faces included.
fn reversed(surveyed: bool) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (mut setup, mut initial) = parts(surveyed)?;
    setup.ports.reverse();
    for port in &mut setup.ports {
        port.faces.reverse();
    }
    setup.links.reverse();
    setup.reaches.reverse();
    initial.reaches.reverse();
    Ok((setup, initial))
}

/// Every accepted bit of `batch` that its declaration order does not name, as text.
fn accepted(batch: &Batch) -> Result<String> {
    let mut text = format!(
        "{} {:?} {:?} {:?} {:?} {:?}",
        batch.time(),
        batch.last_step()?,
        batch.surface().state()?,
        batch.ledger()?,
        batch.port_ledgers()?,
        batch.link_ledgers()?,
    );
    for id in [ALPHA, BETA] {
        text.push_str(&format!(" {id}={:?}", reach_cells(batch, id)?));
    }
    Ok(text)
}

#[test]
fn declaration_order_changes_no_accepted_bit() -> Result {
    declaration_order(false)
}

/// Declaration order on the rectangular `alpha`, or on its surveyed rectangle.
pub(crate) fn declaration_order(surveyed: bool) -> Result {
    let build = |(setup, initial): (SurfaceChannelSetup, SurfaceChannelInitialState)| {
        Batch::new(crate::runtime::client(), setup, initial)
    };
    let mut ordered = build(parts(surveyed)?)?;
    let mut flipped = build(reversed(surveyed)?)?;
    let mut moved = 0.0;
    for index in 0..INTERVALS {
        let start = ordered.time();
        let end = start + 0.25;
        ordered = ordered
            .trial_to(end, &forcing(ordered.setup(), start, end))?
            .into_next();
        flipped = flipped
            .trial_to(end, &forcing(flipped.setup(), start, end))?
            .into_next();
        assert_eq!(accepted(&ordered)?, accepted(&flipped)?, "interval {index}");
        let step = ordered.last_step()?.ok_or("an accepted interval")?;
        assert_eq!(step.ports.len(), 2);
        assert_eq!(step.ports[0].port_id, "p-east");
        moved += step
            .ports
            .iter()
            .map(|p| p.signed_volume_m3.abs())
            .sum::<f64>();
    }
    assert!(moved > 0.0, "the ports must move water");
    Ok(())
}
