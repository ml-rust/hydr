//! Independent 100-digit corpus rows inverted: the law returns each row's discharge from its
//! reference headwater, under the control that row exercises.

use super::inlet_rows::{BARRELS, GRAVITY as INLET_GRAVITY, InletRow, ROWS, SLOPES};
use super::outlet_rows::{FULL_ROWS, GRAVITY as OUTLET_GRAVITY, OutletRow, PROFILE_ROWS};
use super::round_trip::{ROUND_TRIP_TOLERANCE, invert};
use super::support::{Culvert, GRAVITY, TestResult, client, close};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, CulvertInletForm, CulvertOutletBatch, OutletRegime, StructureLaw,
    StructureRegime,
};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// One inlet row per this many, spread over every code and slope.
const INLET_STRIDE: usize = 26;

/// Profile rows that cover each outlet class: mild, critical outlet, horizontal, adverse,
/// supercritical and jump circles, and box rows that reach the crown or drown a steep inlet.
const PROFILE_PICKS: [usize; 8] = [2, 3, 5, 6, 7, 8, 14, 19];

/// The items of `list` whose flag in `kept` is set.
fn keep<T: Copy>(list: &[T], kept: &[bool]) -> Vec<T> {
    list.iter()
        .zip(kept)
        .filter_map(|(item, &flag)| flag.then_some(*item))
        .collect()
}

fn lower(bits: [u64; 2]) -> f64 {
    f64::from_bits(bits[0])
}

fn barrel(circular: bool, rise: u64, span: u64) -> TestResult<CulvertBarrel> {
    Ok(if circular {
        CulvertBarrel::circular(f64::from_bits(rise))?
    } else {
        CulvertBarrel::rectangular_box(f64::from_bits(span), f64::from_bits(rise))?
    })
}

#[test]
fn corpus_gravity_is_the_fixture_gravity() {
    assert_eq!(f64::from_bits(INLET_GRAVITY), GRAVITY);
    assert_eq!(f64::from_bits(OUTLET_GRAVITY), GRAVITY);
}

#[test]
fn inlet_corpus_rows_return_their_discharge_where_the_inlet_governs() -> TestResult {
    let mut culverts = Vec::new();
    let mut picked = Vec::new();
    for row in ROWS.iter().step_by(INLET_STRIDE) {
        let (circular, rise, span) = BARRELS[row.barrel];
        let slope = f64::from_bits(SLOPES[row.slope]);
        // A one-metre smooth barrel with no entrance loss and a free outfall: a weak outlet.
        let mut culvert = Culvert::both(
            barrel(circular, rise, span)?,
            [slope, 0.0],
            CulvertInlet::hds5(row.code)?,
            0.0,
        );
        culvert.length_m = 1.0;
        culvert.manning_n = 0.009;
        culvert.end_b = None;
        culverts.push(culvert);
        picked.push(row);
    }
    // The outlet headwater at the reference discharge decides which rows the inlet governs.
    let mut outlet_rows = Vec::new();
    for culvert in &culverts {
        let StructureLaw::Culvert(parameters) = culvert.law()? else {
            return Err("fixture law is not a culvert".into());
        };
        outlet_rows.push(parameters.forward_rows().ok_or("no A end")?.1);
    }
    let device = CpuDevice::new();
    let flows: Vec<f64> = picked
        .iter()
        .map(|row| f64::from_bits(row.discharge))
        .collect();
    let q = Tensor::<CpuRuntime>::from_slice(&flows, &[flows.len()], &device)?;
    let tw = Tensor::<CpuRuntime>::zeros(&[flows.len()], numr::dtype::DType::F64, &device)?;
    let outlet = CulvertOutletBatch::<CpuRuntime>::new(client(), outlet_rows)?
        .headwater(&q, &tw)?
        .project()?;
    // Rows whose outlet headwater lies certainly below the reference headwater are
    // inlet-governed. The march leaves some steep near-full boxes unresolved; they stay out.
    let governs: Vec<bool> = outlet
        .iter()
        .zip(&picked)
        .map(|(o, row)| {
            o.regime != OutletRegime::Unresolved && o.headwater_m.1 < lower(row.headwater)
        })
        .collect();
    let culverts: Vec<Culvert> = keep(&culverts, &governs);
    let picked: Vec<&InletRow> = keep(&picked, &governs);
    let headwater: Vec<f64> = picked.iter().map(|row| lower(row.headwater)).collect();
    let found = invert(&culverts, &headwater, &vec![0.0; culverts.len()])?;
    let mut governed = 0;
    for (index, ((discharge, regime, _), row)) in found.iter().zip(&picked).enumerate() {
        governed += 1;
        let expected = f64::from_bits(row.discharge);
        assert_eq!(*regime, StructureRegime::CulvertInletControl, "row {index}");
        assert!(
            close(*discharge, expected, ROUND_TRIP_TOLERANCE),
            "code {} slope {} {}: Q={discharge:e}, reference {expected:e}",
            row.code,
            row.slope,
            row.label
        );
    }
    assert!(
        2 * governed > governs.len(),
        "{governed} of {} rows inlet-governed",
        governs.len()
    );
    Ok(())
}

fn outlet_culvert(row: &OutletRow) -> TestResult<Culvert> {
    let inputs = row.inputs;
    let length = f64::from_bits(inputs.length);
    let slope = f64::from_bits(inputs.slope);
    // A weak explicit inlet keeps the inlet headwater far below the outlet headwater.
    let inlet = CulvertInlet::explicit(CulvertInletForm::Two, [1e-3, 1.0, 1e-4, 0.05], false)?;
    let mut culvert = Culvert::both(
        barrel(inputs.circular, inputs.rise, inputs.span)?,
        [slope * length, 0.0],
        inlet,
        f64::from_bits(inputs.ke),
    );
    culvert.length_m = length;
    culvert.manning_n = f64::from_bits(inputs.manning);
    culvert.end_b = None;
    Ok(culvert)
}

#[test]
fn outlet_corpus_rows_return_their_discharge_under_outlet_control() -> TestResult {
    let rows: Vec<&OutletRow> = FULL_ROWS
        .iter()
        .chain(PROFILE_PICKS.iter().filter_map(|&i| PROFILE_ROWS.get(i)))
        .collect();
    let culverts = rows
        .iter()
        .map(|row| outlet_culvert(row))
        .collect::<TestResult<Vec<_>>>()?;
    let headwater: Vec<f64> = rows.iter().map(|row| lower(row.headwater)).collect();
    let tailwater: Vec<f64> = rows
        .iter()
        .map(|row| f64::from_bits(row.inputs.tailwater))
        .collect();
    let found = invert(&culverts, &headwater, &tailwater)?;
    for ((discharge, regime, bounds), row) in found.iter().zip(&rows) {
        let expected = f64::from_bits(row.inputs.discharge);
        assert_eq!(
            *regime,
            StructureRegime::CulvertOutletControl,
            "{}",
            row.name
        );
        assert!(
            close(*discharge, expected, ROUND_TRIP_TOLERANCE),
            "{}: Q={discharge:e} in {bounds:?}, reference {expected:e}",
            row.name
        );
    }
    Ok(())
}
