//! One physical bank resolved by 1, 2, 4, 8, and 16 links on refined surface cells.
//!
//! A `2 m` x `2 m` floodplain on a `1 m` bed starts `0.25 m` deep. Its `y = 0` edge is a
//! `2 m` bank at the bed, discharging freely into one channel cell below the crest. Level `n`
//! splits the floodplain into `n` x `n` cells and the bank into `n` links of `2/n m`. The total
//! exchanged volume over `2 s` converges at first order in the surface cell size.

use crate::fixture::Result;
use crate::scenario::{ReachSpec, Spec, advance, last, weir};
use hydr::coupling::SurfaceChannelWeirRegime;

const SIDE: f64 = 2.0;
const END: f64 = 2.0;

/// Total surface-to-channel volume over `END` with the bank split into `n` links.
fn exchanged(n: usize) -> Result<f64> {
    let dx = SIDE / n as f64;
    let mut spec = Spec::new(n, n, 1.0, 0.25).reach(ReachSpec::new("river", 0.0, &[0.1; 4]));
    spec.dx = dx;
    for cell in 0..n {
        spec = spec.link(weir(&format!("bank-{cell}"), cell, "river", 0, 1.0, dx));
    }
    let batch = advance(spec.build()?, END)?;
    for link in last(&batch)?.links {
        assert_eq!(link.regime, SurfaceChannelWeirRegime::Free, "{link:?}");
        assert!(link.midpoint_channel_stage_m < 1.0, "{link:?}");
    }
    let ledgers = batch.link_ledgers()?;
    assert!(ledgers.iter().all(|l| l.channel_to_surface_m3 == 0.0));
    Ok(ledgers.iter().map(|l| l.surface_to_channel_m3).sum())
}

#[test]
fn spatial_link_refinement_is_consistent() -> Result {
    // Frozen total exchanged volumes, in cubic metres, at each refinement level.
    let frozen = [
        (1, 5.080_982_123_815_221e-1),
        (2, 4.500_837_781_192_897e-1),
        (4, 4.097_174_344_728_737e-1),
        (8, 3.890_740_502_518_448e-1),
        (16, 3.790_584_235_755_662e-1),
    ];
    let mut volumes = Vec::with_capacity(frozen.len());
    for (n, frozen_volume) in frozen {
        let volume = exchanged(n)?;
        assert!(
            (volume - frozen_volume).abs() <= 1e-9,
            "n={n}: volume={volume:.17e}, frozen={frozen_volume:.17e}"
        );
        volumes.push(volume);
    }
    // Successive level differences shrink, reaching first order on the finer levels.
    let differences: Vec<f64> = volumes.windows(2).map(|w| (w[0] - w[1]).abs()).collect();
    for (index, pair) in differences.windows(2).enumerate() {
        let ratio = pair[0] / pair[1];
        let floor = if index == 0 { 1.4 } else { 1.9 };
        assert!(ratio >= floor, "level {index}: differences={differences:?}");
    }
    let finest = differences.last().ok_or("no refinement difference")?;
    // The 8-link and 16-link banks agree within 1.1e-2 m³, about 3% of the exchange.
    assert!(*finest <= 1.1e-2, "differences={differences:?}");
    Ok(())
}
