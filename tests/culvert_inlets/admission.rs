//! Inlet admission: falling transitions, crown-bound critical heads, rows and query inputs.

use super::support::{TestResult, batch, tensor};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, CulvertInletForm, CulvertInletRow, StructureError,
    StructureInputErrorKind as K,
};

fn row(barrel: CulvertBarrel, inlet: CulvertInlet, slope: f64) -> CulvertInletRow {
    CulvertInletRow {
        barrel,
        inlet,
        slope,
        gravity_m_per_s2: 9.81,
    }
}

/// Requires `result` to reject with `kind` at `element`, naming every fragment.
fn rejects<T>(
    result: Result<T, StructureError>,
    kind: K,
    element: Option<usize>,
    fragments: &[&str],
) -> TestResult {
    match result.map(|_| ()) {
        Err(StructureError::InvalidInput {
            kind: found,
            context,
            element: at,
        }) if found == kind
            && at == element
            && fragments.iter().all(|fragment| context.contains(fragment)) =>
        {
            Ok(())
        }
        other => Err(format!(
            "expected {kind:?} at {element:?} with {fragments:?}, found {other:?}"
        )
        .into()),
    }
}

#[test]
fn admission_rejects_a_falling_transition_naming_the_coefficients() -> TestResult {
    let box_barrel = CulvertBarrel::rectangular_box(2.0, 1.5)?;
    // K 3.5^M = 3.5 lies above 16 c + Y = 0.66, so the headwater falls across the transition.
    let falling = CulvertInlet::explicit(CulvertInletForm::Two, [1.0, 1.0, 0.01, 0.5], false)?;
    let good = CulvertInlet::hds5(12)?;
    rejects(
        batch(vec![
            row(box_barrel, good, 0.01),
            row(box_barrel, falling, 0.01),
        ]),
        K::Definition,
        Some(1),
        &[
            "explicit inlet form=Two,k=1.0,m=1.0,c=0.01,y=0.5,mitered=false",
            "the headwater falls across the transition",
            "Supply coefficients whose submerged headwater at X = 4 exceeds",
        ],
    )?;
    // Code 12 rises at small slopes, but a steep slope lowers its submerged headwater below the
    // unsubmerged one, so admission rejects the code there.
    rejects(
        batch(vec![row(box_barrel, good, 0.3)]),
        K::Definition,
        Some(0),
        &[
            "inlet code=12,form=Two,k=0.51",
            "slope=0.3",
            "falls across the transition",
        ],
    )?;
    Ok(())
}

#[test]
fn admission_rejects_a_form_one_row_whose_critical_depth_reaches_the_crown() -> TestResult {
    let mut weak = row(
        CulvertBarrel::rectangular_box(2.0, 1.5)?,
        CulvertInlet::hds5(9)?,
        0.0,
    );
    // Box critical depth at X = 3.5 reaches the crown below g = 12.25 x 0.3048 m/s2.
    weak.gravity_m_per_s2 = 1.0;
    rejects(
        batch(vec![weak]),
        K::Definition,
        Some(0),
        &[
            "inlet code=9",
            "gravity_m_per_s2=1.0",
            "is not below the crown",
        ],
    )
}

#[test]
fn admission_rejects_rows_naming_the_row_value_and_action() -> TestResult {
    let circle = CulvertBarrel::circular(1.2)?;
    let coded = CulvertInlet::hds5(1)?;
    rejects(
        batch(vec![row(circle, CulvertInlet::hds5(9)?, 0.01)]),
        K::Definition,
        Some(0),
        &[
            "inlet code=9",
            "barrel_shape=Circular",
            "the code is for Box barrels",
        ],
    )?;
    rejects(
        batch(vec![row(circle, coded, 0.0), row(circle, coded, f64::NAN)]),
        K::Definition,
        Some(1),
        &["slope=NaN: supply a finite barrel slope"],
    )?;
    let mut heavy = row(circle, coded, 0.0);
    heavy.gravity_m_per_s2 = -9.81;
    rejects(
        batch(vec![heavy]),
        K::Definition,
        Some(0),
        &["gravity_m_per_s2=-9.81: supply a finite positive gravity"],
    )?;
    rejects(
        batch(Vec::new()),
        K::Definition,
        None,
        &["supply at least one culvert inlet row"],
    )
}

#[test]
fn headwater_query_rejects_bad_discharges_naming_the_row() -> TestResult {
    let circle = CulvertBarrel::circular(1.2)?;
    let inlets = batch(vec![
        row(circle, CulvertInlet::hds5(1)?, 0.0),
        row(
            CulvertBarrel::rectangular_box(2.0, 1.5)?,
            CulvertInlet::hds5(20)?,
            0.0,
        ),
    ])?;
    rejects(
        inlets.headwater(&tensor(&[0.5, -0.5])?),
        K::OperatingPoint,
        Some(1),
        &["discharge_m3_per_s=-0.5: supply a nonnegative discharge"],
    )?;
    rejects(
        inlets.headwater(&tensor(&[f64::INFINITY, 0.5])?),
        K::NonfiniteInput,
        Some(0),
        &["supply a finite discharge"],
    )?;
    rejects(
        inlets.headwater(&tensor(&[0.5])?),
        K::OperatingPoint,
        None,
        &["supply one value per barrel"],
    )
}

#[test]
fn zero_discharge_returns_the_equation_value_exactly() -> TestResult {
    let circle = CulvertBarrel::circular(1.2)?;
    let rectangle = CulvertBarrel::rectangular_box(2.0, 1.5)?;
    let inlets = batch(vec![
        row(circle, CulvertInlet::hds5(1)?, 0.02),
        row(rectangle, CulvertInlet::hds5(20)?, 0.02),
        row(circle, CulvertInlet::hds5(5)?, 0.0),
    ])?;
    let rows = inlets.headwater(&tensor(&[0.0, 0.0, 0.0])?)?.project()?;
    let found: Vec<(f64, f64)> = rows.iter().map(|r| r.headwater_m).collect();
    // Form 1 gives D Ks S, which the slope bounds enclose. Form 2 and a level barrel give zero.
    let sloped = 1.2 * -0.5 * 0.02;
    let first = found.first().ok_or("no first row")?;
    if !(first.0 <= sloped && sloped <= first.1 && first.1 - first.0 < 1e-17) {
        return Err(format!("form 1 at Q = 0: {first:?}, expected about {sloped:e}").into());
    }
    if found.get(1..) != Some(&[(0.0, 0.0), (0.0, 0.0)][..]) {
        return Err(format!("zero-flow rows {found:?}").into());
    }
    Ok(())
}
