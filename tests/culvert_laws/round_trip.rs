//! `Q(HW(Q))` returns `Q`: the law inverts the larger control headwater of the query batches.

use super::support::{Culvert, TestResult, client, close, dry, elements, propose, wet};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, CulvertInletBatch, CulvertOutletBatch, StructureHydraulicOperands,
    StructureLaw, StructureRegime,
};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Frozen before the first run. Headwater enclosures stay below `1e-9` of the headwater, and
/// the relative discharge error is that width over `Q dHW/dQ / HW`, at least `1/2.5` here.
pub const ROUND_TRIP_TOLERANCE: f64 = 1e-6;

/// Larger of the two control headwaters per lane at discharges `q` and tailwaters `tw`, as
/// midpoints of the query enclosures, with the inlet and outlet midpoints.
pub fn control_headwater(culverts: &[Culvert], q: &[f64], tw: &[f64]) -> TestResult<Vec<[f64; 3]>> {
    let mut inlet_rows = Vec::new();
    let mut outlet_rows = Vec::new();
    for culvert in culverts {
        let StructureLaw::Culvert(parameters) = culvert.law()? else {
            return Err("fixture law is not a culvert".into());
        };
        let (inlet, outlet) = parameters
            .forward_rows()
            .ok_or("fixture culvert has no A end")?;
        inlet_rows.push(inlet);
        outlet_rows.push(outlet);
    }
    let device = numr::runtime::cpu::CpuDevice::new();
    let column =
        |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device);
    let inlet = CulvertInletBatch::<CpuRuntime>::new(client(), inlet_rows)?
        .headwater(&column(q)?)?
        .project()?;
    let outlet = CulvertOutletBatch::<CpuRuntime>::new(client(), outlet_rows)?
        .headwater(&column(q)?, &column(tw)?)?
        .project()?;
    Ok(inlet
        .iter()
        .zip(&outlet)
        .map(|(i, o)| {
            let hi = 0.5 * (i.headwater_m.0 + i.headwater_m.1);
            let ho = 0.5 * (o.headwater_m.0 + o.headwater_m.1);
            [hi.max(ho), hi, ho]
        })
        .collect())
}

/// Per-barrel discharge, regime, and discharge enclosure of one lane.
pub type Inverted = (f64, StructureRegime, (f64, f64));

/// One proposal per lane at headwater `hw` above the A invert and tailwater `tw` above the B
/// invert, returning the per-barrel discharge and regime.
pub fn invert(culverts: &[Culvert], hw: &[f64], tw: &[f64]) -> TestResult<Vec<Inverted>> {
    let elements = elements(culverts)?;
    let supports: Vec<_> = culverts
        .iter()
        .zip(hw.iter().zip(tw))
        .map(|(culvert, (&h, &t))| {
            let receiver = if t > 0.0 {
                wet(culvert.invert_b_m + t)
            } else {
                dry(culvert.invert_b_m)
            };
            (wet(culvert.invert_a_m + h), receiver)
        })
        .collect();
    let outputs = propose(&elements, &supports)?;
    outputs
        .iter()
        .zip(culverts)
        .map(|(output, culvert)| {
            let bounds = match output.hydraulic_operands {
                Some(StructureHydraulicOperands::Culvert {
                    barrel_discharge_m3_per_s,
                    ..
                }) => barrel_discharge_m3_per_s,
                _ => (0.0, 0.0),
            };
            Ok((
                output.discharge_m3_per_s / f64::from(culvert.count),
                output.regime,
                bounds,
            ))
        })
        .collect()
}

fn mild(barrel: CulvertBarrel, code: u8) -> TestResult<Culvert> {
    Ok(Culvert::both(
        barrel,
        [0.03, 0.0],
        CulvertInlet::hds5(code)?,
        0.5,
    ))
}

/// Relative gap between the two control headwaters below which a lane may tie.
const SEPARATION: f64 = 1e-9;

#[test]
fn culvert_discharge_inverts_the_larger_control_headwater() -> TestResult {
    let circle = CulvertBarrel::circular(1.2)?;
    let rectangle = CulvertBarrel::rectangular_box(2.0, 1.5)?;
    // Mild, adverse within half the rise, and level barrels.
    let culverts = [
        mild(circle, 1)?,
        mild(rectangle, 26)?,
        Culvert::both(circle, [0.0, 0.3], CulvertInlet::hds5(3)?, 0.5),
        Culvert::both(rectangle, [0.0, 0.0], CulvertInlet::hds5(9)?, 0.2),
    ];
    let mut lanes = Vec::new();
    let mut q = Vec::new();
    let mut tw = Vec::new();
    for culvert in &culverts {
        for (flow, tail) in [(0.4, 0.0), (3.0, 1.0), (5.0, 2.0)] {
            lanes.push(*culvert);
            q.push(flow);
            tw.push(tail);
        }
    }
    // A short smooth level barrel with no entrance loss leaves the inlet in control.
    let mut short = Culvert::both(circle, [0.0, 0.0], CulvertInlet::hds5(1)?, 0.0);
    short.length_m = 2.0;
    short.manning_n = 0.009;
    for flow in [0.8, 1.5] {
        lanes.push(short);
        q.push(flow);
        tw.push(0.0);
    }
    let hw = control_headwater(&lanes, &q, &tw)?;
    let total: Vec<f64> = hw.iter().map(|h| h[0]).collect();
    let found = invert(&lanes, &total, &tw)?;
    let mut seen = [false; 2];
    for (index, ((discharge, regime, bounds), expected)) in found.iter().zip(&q).enumerate() {
        let [_, inlet, outlet] = hw[index];
        assert!(
            close(*discharge, *expected, ROUND_TRIP_TOLERANCE),
            "lane {index}: Q={discharge:e}, expected {expected:e}, {regime:?}, bounds {bounds:?}, hw {:?}",
            hw[index]
        );
        if !close(inlet, outlet, SEPARATION) {
            let governing = if inlet > outlet {
                StructureRegime::CulvertInletControl
            } else {
                StructureRegime::CulvertOutletControl
            };
            assert_eq!(*regime, governing, "lane {index}: hw {:?}", hw[index]);
            seen[usize::from(inlet < outlet)] = true;
        }
    }
    assert_eq!(seen, [true, true], "both controls govern some lane");
    Ok(())
}

/// Drowned full circle whose submerged inlet and full outlet headwaters cross at `X = 6`.
#[test]
fn controls_tie_where_the_two_headwaters_cross() -> TestResult {
    let diameter = 1.0;
    let mut culvert = Culvert::both(
        CulvertBarrel::circular(diameter)?,
        [0.0, 0.0],
        CulvertInlet::hds5(1)?,
        0.5,
    );
    culvert.length_m = 20.0;
    culvert.manning_n = 0.012;
    let g = super::support::GRAVITY;
    let area = std::f64::consts::PI * diameter * diameter / 4.0;
    let radius = diameter / 4.0;
    let ku = 0.3048_f64.powf(-0.5);
    let x = 6.0;
    let q = x * area * diameter.sqrt() / ku;
    // Code 1: c = 0.0398, Y = 0.67. The level barrel has no slope term.
    let inlet = diameter * (0.0398 * x * x + 0.67);
    let k = 1.0 + 0.5 + 2.0 * g * 0.012 * 0.012 * 20.0 / radius.powf(4.0 / 3.0);
    let velocity = q / area;
    let tie = inlet - k * velocity * velocity / (2.0 * g);
    assert!(tie > diameter, "the outlet stays drowned");
    let lanes = [culvert; 3];
    let tw = [tie - 0.05, tie, tie + 0.05];
    let found = invert(&lanes, &[inlet; 3], &tw)?;
    let regimes: Vec<StructureRegime> = found.iter().map(|lane| lane.1).collect();
    assert_eq!(
        regimes,
        [
            StructureRegime::CulvertInletControl,
            StructureRegime::CulvertControlTie,
            StructureRegime::CulvertOutletControl,
        ]
    );
    assert!(
        close(found[1].0, q, ROUND_TRIP_TOLERANCE),
        "{found:?}, q {q}"
    );
    assert!(close(found[0].0, q, ROUND_TRIP_TOLERANCE));
    assert!(found[2].0 < q);
    Ok(())
}

/// A 2 m by 1.5 m box at slope 0.02 above its critical capacity, free and drowned outlets.
#[test]
fn steep_box_above_its_critical_capacity_inverts_its_headwater() -> TestResult {
    let mut culvert = Culvert::both(
        CulvertBarrel::rectangular_box(2.0, 1.5)?,
        [0.8, 0.0],
        CulvertInlet::hds5(9)?,
        0.5,
    );
    culvert.length_m = 40.0;
    // The B end would fall more than half the rise, so the culvert runs A to B only.
    culvert.end_b = None;
    let mut q = Vec::new();
    let mut tw = Vec::new();
    for flow in [12.0, 15.0, 18.0, 21.0, 24.0] {
        for tail in [0.0, 1.0, 1.8] {
            q.push(flow);
            tw.push(tail);
        }
    }
    let lanes = vec![culvert; q.len()];
    let hw = control_headwater(&lanes, &q, &tw)?;
    if let Some(row) = hw
        .iter()
        .find(|row| !row.iter().all(|value| value.is_finite()))
    {
        return Err(format!("control headwater {row:?} is not finite").into());
    }
    let total: Vec<f64> = hw.iter().map(|h| h[0]).collect();
    let found = invert(&lanes, &total, &tw)?;
    for (index, ((discharge, regime, bounds), expected)) in found.iter().zip(&q).enumerate() {
        if !close(*discharge, *expected, ROUND_TRIP_TOLERANCE) {
            return Err(format!(
                "lane {index}: Q={discharge:e}, expected {expected:e}, {regime:?}, bounds {bounds:?}, hw {:?}",
                hw.get(index)
            )
            .into());
        }
    }
    Ok(())
}
