//! Dense sweeps across the inlet bands and across the outlet turning full: the discharge rises
//! with the headwater and stays continuous at each boundary.

use super::round_trip::invert;
use super::support::{Culvert, TestResult, client};
use hydr::structures::{CulvertBarrel, CulvertInlet, CulvertInletBatch, StructureLaw};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Frozen before the first run: the relative discharge change across a boundary between heads
/// `1e-7` apart. A continuous law moves by about `1e-7 / M`. A jump moves by far more.
const CONTINUITY: f64 = 1e-5;

/// Relative head offset on each side of a boundary.
const OFFSET: f64 = 1e-7;

fn short_box() -> TestResult<Culvert> {
    let mut culvert = Culvert::both(
        CulvertBarrel::rectangular_box(2.0, 1.5)?,
        [0.0, 0.0],
        CulvertInlet::hds5(9)?,
        0.0,
    );
    culvert.length_m = 2.0;
    culvert.manning_n = 0.009;
    culvert.end_b = None;
    Ok(culvert)
}

/// Inlet headwater midpoints at `X = 3.5` and `X = 4` of the A end of `culvert`.
fn anchors(culvert: &Culvert) -> TestResult<[f64; 2]> {
    let StructureLaw::Culvert(parameters) = culvert.law()? else {
        return Err("fixture law is not a culvert".into());
    };
    let (row, _) = parameters.forward_rows().ok_or("no A end")?;
    let barrel = row.barrel;
    let area = barrel.span_m() * barrel.rise_m();
    let scale = area * barrel.rise_m().sqrt() * 0.3048_f64.sqrt();
    let flows = [3.5 * scale, 4.0 * scale];
    let device = CpuDevice::new();
    let q = Tensor::<CpuRuntime>::from_slice(&flows, &[2], &device)?;
    let rows = CulvertInletBatch::<CpuRuntime>::new(client(), vec![row; 2])?
        .headwater(&q)?
        .project()?;
    Ok([0, 1].map(|i| 0.5 * (rows[i].headwater_m.0 + rows[i].headwater_m.1)))
}

#[test]
fn inlet_band_sweep_rises_and_stays_continuous_at_both_limits() -> TestResult {
    let culvert = short_box()?;
    let [low, high] = anchors(&culvert)?;
    let mut heads: Vec<f64> = (1..=16).map(|i| high * 1.4 * f64::from(i) / 16.0).collect();
    for anchor in [low, high] {
        heads.push(anchor * (1.0 - OFFSET));
        heads.push(anchor * (1.0 + OFFSET));
    }
    heads.sort_by(f64::total_cmp);
    let lanes = vec![culvert; heads.len()];
    let found = invert(&lanes, &heads, &vec![0.0; heads.len()])?;
    for (pair, window) in found.windows(2).zip(heads.windows(2)) {
        let [(below, ..), (above, ..)] = pair else {
            return Err("windows of two".into());
        };
        assert!(
            above > below,
            "Q falls from {below:e} to {above:e} at {window:?}"
        );
    }
    for anchor in [low, high] {
        let index = heads
            .iter()
            .position(|&h| h == anchor * (1.0 - OFFSET))
            .ok_or("anchor head missing")?;
        let (below, above) = (found[index].0, found[index + 1].0);
        assert!(
            (above - below).abs() <= CONTINUITY * below,
            "jump at the anchor {anchor}: {below:e} to {above:e}"
        );
    }
    Ok(())
}

#[test]
fn outlet_turning_full_stays_continuous_in_the_tailwater() -> TestResult {
    let rise = 1.5;
    let mut culvert = Culvert::both(
        CulvertBarrel::rectangular_box(2.0, rise)?,
        [0.0, 0.0],
        CulvertInlet::hds5(9)?,
        0.5,
    );
    culvert.length_m = 100.0;
    culvert.manning_n = 0.03;
    culvert.end_b = None;
    let tailwater = [
        1.2,
        1.4,
        rise * (1.0 - OFFSET),
        rise * (1.0 + OFFSET),
        1.6,
        1.8,
    ];
    let lanes = vec![culvert; tailwater.len()];
    let found = invert(&lanes, &[2.5; 6], &tailwater)?;
    for (pair, window) in found.windows(2).zip(tailwater.windows(2)) {
        let [(upstream, ..), (downstream, ..)] = pair else {
            return Err("windows of two".into());
        };
        assert!(
            downstream < upstream,
            "Q rises from {upstream:e} to {downstream:e} as the tailwater rises over {window:?}"
        );
    }
    let (below, above) = (found[2].0, found[3].0);
    assert!(
        (above - below).abs() <= CONTINUITY * below,
        "jump where the outlet turns full: {below:e} to {above:e}"
    );
    Ok(())
}
