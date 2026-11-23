//! Critical and normal depth above a floor against the frozen independent certificates and roots.

use super::depth_rows::{CERTIFICATES, CRITICAL, CRITICAL_ROWS, Class, GRAVITY, NORMAL_ROWS};
use super::runtime::{Runtime, client};
use super::section_rows::{SECTIONS, Section, Shape};
use super::sections::{Outcome, floored, tensor};
use hydr::structures::{
    CriticalDepthStatus, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelDepthRow, CulvertFloor,
    NormalDepthStatus,
};

fn section(index: usize) -> Outcome<&'static Section> {
    Ok(SECTIONS.get(index).ok_or("row names no section")?)
}

fn reduced(index: usize) -> f64 {
    SECTIONS
        .get(index)
        .map_or(f64::NAN, |s| f64::from_bits(s.reduced_rise))
}

/// The floored barrel of `section` with the bed ratio `ratio`.
fn with_ratio(section: &Section, ratio: f64) -> Outcome<CulvertBarrel> {
    let barrel = floored(section)?;
    let floor = barrel.floor().ok_or("the section lost its floor")?;
    Ok(barrel.with_floor(CulvertFloor {
        roughness_ratio: ratio,
        ..floor
    })?)
}

/// `true` when the enclosure holds the reference root, or the row has no reference root.
fn holds(depth: Option<(f64, f64)>, root: Option<[u64; 2]>) -> bool {
    match (depth, root) {
        (_, None) => true,
        (Some((lower, upper)), Some([down, up])) => {
            lower <= f64::from_bits(down) && f64::from_bits(up) <= upper
        }
        (None, Some(_)) => false,
    }
}

/// `true` when a solved enclosure is no wider than `relative_width` times its root.
fn narrow(depth: Option<(f64, f64)>, root: Option<[u64; 2]>, relative_width: f64) -> bool {
    let scale = root.map_or(0.0, |[_, up]| f64::from_bits(up));
    depth.is_some_and(|(lower, upper)| upper - lower <= relative_width * scale)
}

/// `2^e` that takes the rise `rise` into `[1, 2)`, the frame scale of its barrel.
fn frame_scale(rise: f64) -> f64 {
    2f64.powi(rise.log2().floor() as i32)
}

#[test]
fn floor_certificates_hold_the_independent_minima_and_peaks() -> Outcome<()> {
    let barrels = CERTIFICATES
        .iter()
        .map(|c| with_ratio(section(c.section)?, f64::from_bits(c.roughness_ratio)))
        .collect::<Outcome<Vec<_>>>()?;
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let found = batch.depth_certificates()?;
    let mut failures = Vec::new();
    for (entry, row) in CERTIFICATES.iter().zip(&found) {
        let index = entry.section;
        let name = section(index)?.name;
        let ratio = f64::from_bits(entry.roughness_ratio);
        let critical = CRITICAL.get(index).ok_or("section has no critical row")?;
        let kc = row.critical_slope;
        if !(0.0 < kc && kc <= f64::from_bits(critical.minimum)) {
            failures.push(format!("{name} rho={ratio}: kappa_c={kc:e}"));
        }
        if let Some(minimum) = entry.box_minimum {
            let constants = row.critical_top == 1.0
                && kc == 3.0
                && row.normal_top == 1.0
                && row.normal_slope == 3.0
                && row.peak == (1.0, 1.0);
            if !(constants && row.normal_slope <= f64::from_bits(minimum)) {
                failures.push(format!("{name} rho={ratio}: box constants {row:?}"));
            }
            continue;
        }
        let s_n = row.normal_top;
        // The true minimum over [2^-30, s_i] bounds the minimum over [2^-30, s] above for s_i <= s.
        let below = |s: f64| {
            entry
                .rising
                .iter()
                .filter(|(at, _)| f64::from_bits(*at) <= s)
                .map(|(_, m)| f64::from_bits(*m))
                .next_back()
        };
        let kn = row.normal_slope;
        if !(0.0 < kn && below(s_n).is_some_and(|m| kn <= m)) {
            failures.push(format!(
                "{name} rho={ratio}: kappa_n={kn:e} against {:?}",
                below(s_n)
            ));
        }
        // Knots rise to s_N and their floors fall to kappa_n, each under the reference minimum.
        let knots_hold = row.limb.len() == 8
            && row.limb.windows(2).all(|w| match w {
                [(s, k), (t, l)] => s <= t && k >= l,
                _ => false,
            })
            && row
                .limb
                .iter()
                .all(|&(s, k)| kn <= k && s <= s_n && below(s).is_none_or(|m| k <= m));
        if !knots_hold {
            failures.push(format!("{name} rho={ratio}: knots {:?}", row.limb));
        }
        let [down, up] = entry
            .peak
            .ok_or("arc section without a peak")?
            .map(f64::from_bits);
        let (lower, upper) = row.peak;
        if !(lower <= down && up <= upper && s_n < down && row.critical_top == 1.0 - f64::EPSILON) {
            failures.push(format!(
                "{name} rho={ratio}: peak [{lower:e}, {upper:e}] misses [{down:e}, {up:e}]"
            ));
        }
        // The context holds the log conveyance in the frame. The reference holds it in units of
        // D', which differ by (8/3) ln D'_f.
        let [log_down, log_up] = entry
            .log_peak
            .ok_or("arc section without a log peak")?
            .map(f64::from_bits);
        let open = reduced(index) / frame_scale(rise(section(index)?));
        let shift = 8.0 / 3.0 * open.ln();
        let slack = 1e-14 * (1.0 + shift.abs());
        let (log_lower, log_upper) = row.log_peak;
        if !(log_lower - shift - slack <= log_down && log_up <= log_upper - shift + slack) {
            failures.push(format!(
                "{name} rho={ratio}: log peak [{log_lower:e}, {log_upper:e}] - {shift:e} misses [{log_down:e}, {log_up:e}]"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// Unreduced rise of a section: the only input of a circle, the second of every other shape.
fn rise(section: &Section) -> f64 {
    let index = usize::from(section.shape != Shape::Circular);
    section
        .inputs
        .get(index)
        .map_or(f64::NAN, |bits| f64::from_bits(*bits))
}

#[test]
fn peaks_rise_with_the_bed_roughness() -> Outcome<()> {
    for (index, entry) in SECTIONS.iter().enumerate() {
        if entry.shape == Shape::Box {
            continue;
        }
        let ratios = [0.5, 1.0, 2.5];
        let barrels = ratios
            .iter()
            .map(|&ratio| with_ratio(entry, ratio))
            .collect::<Outcome<Vec<_>>>()?;
        let found = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?.depth_certificates()?;
        let peaks: Vec<(f64, f64)> = found.iter().map(|row| row.peak).collect();
        assert!(
            peaks.windows(2).all(|w| w[0].1 < w[1].0),
            "{} (section {index}): {peaks:?}",
            entry.name
        );
    }
    Ok(())
}

#[test]
fn critical_roots_above_a_floor_lie_inside_enclosures_with_matching_outcomes() -> Outcome<()> {
    use CriticalDepthStatus::{AboveCrown, AtCrown, BelowFloor, Dry, Solved};
    let barrels = CRITICAL_ROWS
        .iter()
        .map(|row| floored(section(row.section)?))
        .collect::<Outcome<Vec<_>>>()?;
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let flows: Vec<f64> = CRITICAL_ROWS
        .iter()
        .map(|row| f64::from_bits(row.discharge))
        .collect();
    let rows: Vec<CulvertBarrelDepthRow<CriticalDepthStatus>> = batch
        .critical_depth(&tensor(&flows)?, f64::from_bits(GRAVITY))?
        .project()?;
    let mut failures = Vec::new();
    for (reference, row) in CRITICAL_ROWS.iter().zip(&rows) {
        let name = section(reference.section)?.name;
        let d = reduced(reference.section);
        let admitted: &[CriticalDepthStatus] = match reference.class {
            Class::Dry => &[Dry],
            Class::BelowFloor => &[BelowFloor],
            Class::NearFloor => &[BelowFloor, Solved],
            Class::Free => &[Solved],
            Class::NearCrown => &[Solved, AtCrown, AboveCrown],
            _ => &[AboveCrown],
        };
        let shaped = match (row.status, row.depth_m) {
            (Dry, depth) => depth == Some((0.0, 0.0)),
            (BelowFloor, Some((lower, upper))) => lower == 0.0 && upper <= 2.0 * d * 2f64.powi(-30),
            (Solved, depth) if reference.class == Class::Free => {
                narrow(depth, reference.root, reference.relative_width)
            }
            (AtCrown | AboveCrown, Some((_, upper))) => upper == d,
            (AboveCrown, None) => section(reference.section)?.shape == Shape::Box,
            (_, depth) => depth.is_some(),
        };
        if !(admitted.contains(&row.status) && holds(row.depth_m, reference.root) && shaped) {
            failures.push(format!(
                "{name} {}: {row:?} against {:?}",
                reference.label, reference.root
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn normal_roots_above_a_floor_lie_inside_enclosures_with_matching_outcomes() -> Outcome<()> {
    use NormalDepthStatus::{
        AboveCapacity, BelowFloor, Dry, NearCapacity, NonPositiveSlope, Solved,
    };
    let barrels = NORMAL_ROWS
        .iter()
        .map(|row| floored(section(row.section)?))
        .collect::<Outcome<Vec<_>>>()?;
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let flows: Vec<f64> = NORMAL_ROWS
        .iter()
        .map(|row| f64::from_bits(row.discharge))
        .collect();
    let slopes: Vec<f64> = NORMAL_ROWS
        .iter()
        .map(|row| f64::from_bits(row.slope))
        .collect();
    let roughness = NORMAL_ROWS
        .iter()
        .map(|row| Ok(f64::from_bits(section(row.section)?.wall_manning)))
        .collect::<Outcome<Vec<f64>>>()?;
    let rows = batch
        .normal_depth(&tensor(&flows)?, &tensor(&roughness)?, &tensor(&slopes)?)?
        .project()?;
    let mut failures = Vec::new();
    for (reference, row) in NORMAL_ROWS.iter().zip(&rows) {
        let name = section(reference.section)?.name;
        let d = reduced(reference.section);
        let admitted: &[NormalDepthStatus] = match reference.class {
            Class::Dry => &[Dry],
            Class::BelowFloor => &[BelowFloor],
            Class::NearFloor => &[BelowFloor, Solved],
            Class::Free => &[Solved],
            Class::NearPeak => &[Solved, NearCapacity],
            Class::AboveCapacity => &[AboveCapacity],
            _ => &[NonPositiveSlope],
        };
        let shaped = match (row.status, row.depth_m) {
            (Dry, depth) => depth == Some((0.0, 0.0)),
            (BelowFloor, Some((lower, upper))) => lower == 0.0 && upper <= 2.0 * d * 2f64.powi(-30),
            (Solved, depth) if reference.class == Class::Free => {
                narrow(depth, reference.root, reference.relative_width)
            }
            (NearCapacity, Some((lower, upper))) => 0.0 < lower && upper <= d,
            (AboveCapacity | NonPositiveSlope, depth) => depth.is_none(),
            (_, depth) => depth.is_some(),
        };
        if !(admitted.contains(&row.status) && holds(row.depth_m, reference.root) && shaped) {
            failures.push(format!(
                "{name} {} S={:e}: {row:?} against {:?}",
                reference.label,
                f64::from_bits(reference.slope),
                reference.root
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
