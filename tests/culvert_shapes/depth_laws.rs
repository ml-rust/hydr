//! Arc barrel depth laws: monotonicity, scale invariance, tiny barrels and unchanged circle and
//! box rows.

use super::depth_rows::CERTIFIED;
use super::geometry::{Outcome, tensor};
use super::runtime::{Runtime, client};
use super::{TestResult, barrel};
use hydr::structures::{
    CriticalDepthStatus, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelDepthRow,
    NormalDepthStatus,
};

type Critical = Vec<CulvertBarrelDepthRow<CriticalDepthStatus>>;
type Normal = Vec<CulvertBarrelDepthRow<NormalDepthStatus>>;

/// The evidence barrels of the plan, one per shape family and source.
fn evidence() -> Outcome<Vec<CulvertBarrel>> {
    CERTIFIED
        .iter()
        .take(12)
        .map(|entry| barrel(&entry.barrel))
        .collect()
}

fn critical(barrels: Vec<CulvertBarrel>, flows: &[f64]) -> Outcome<Critical> {
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    Ok(batch.critical_depth(&tensor(flows)?, 9.81)?.project()?)
}

fn normal(barrels: Vec<CulvertBarrel>, [flows, roughness, slopes]: [&[f64]; 3]) -> Outcome<Normal> {
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let rows = batch.normal_depth(&tensor(flows)?, &tensor(roughness)?, &tensor(slopes)?)?;
    Ok(rows.project()?)
}

/// No certified reversal: each enclosure reaches at least the previous lower end.
fn rising(depths: &[Option<(f64, f64)>]) -> bool {
    depths.windows(2).all(|pair| match pair {
        [Some((lower, _)), Some((_, upper))] => upper >= lower,
        _ => true,
    })
}

#[test]
fn depths_rise_with_discharge_roughness_and_falling_slope() -> TestResult {
    let flows: Vec<f64> = (0..65).map(|i| 1e-4 * 1.25f64.powi(i)).collect();
    for culvert in evidence()? {
        let rows = critical(vec![culvert; flows.len()], &flows)?;
        let depths: Vec<_> = rows.iter().map(|row| row.depth_m).collect();
        assert!(rising(&depths), "{culvert:?} critical");
        let n = vec![0.024; flows.len()];
        let s = vec![1e-3; flows.len()];
        let rows = normal(vec![culvert; flows.len()], [&flows, &n, &s])?;
        let depths: Vec<_> = rows.iter().map(|row| row.depth_m).collect();
        assert!(rising(&depths), "{culvert:?} normal in Q");
        let roughness: Vec<f64> = (0..17).map(|i| 0.01 + 0.002 * f64::from(i)).collect();
        let q = vec![0.3; roughness.len()];
        let s = vec![1e-3; roughness.len()];
        let rows = normal(vec![culvert; roughness.len()], [&q, &roughness, &s])?;
        let depths: Vec<_> = rows.iter().map(|row| row.depth_m).collect();
        assert!(rising(&depths), "{culvert:?} normal in n");
        let slopes: Vec<f64> = (0..17).map(|i| 0.05 * 0.65f64.powi(i)).collect();
        let n = vec![0.024; slopes.len()];
        let rows = normal(vec![culvert; slopes.len()], [&q, &n, &slopes])?;
        let depths: Vec<_> = rows.iter().map(|row| row.depth_m).collect();
        assert!(rising(&depths), "{culvert:?} normal in S");
    }
    Ok(())
}

/// The barrel with every length times `scale`.
fn scaled(culvert: &CulvertBarrel, scale: f64) -> Outcome<CulvertBarrel> {
    use hydr::structures::CulvertBarrelShape::{
        Arch, HorizontalEllipse, PipeArch, VerticalEllipse,
    };
    let [b, d] = [culvert.span_m() * scale, culvert.rise_m() * scale];
    let radii: Vec<f64> = culvert
        .given_radius_m()
        .unwrap_or(&[])
        .iter()
        .map(|r| r * scale)
        .collect();
    Ok(match (culvert.shape(), radii.as_slice()) {
        (HorizontalEllipse, &[r]) => CulvertBarrel::horizontal_ellipse(b, d, r)?,
        (VerticalEllipse, &[r]) => CulvertBarrel::vertical_ellipse(b, d, r)?,
        (PipeArch, &[c, r]) => CulvertBarrel::pipe_arch(b, d, c, r)?,
        (Arch, &[]) => CulvertBarrel::arch(b, d)?,
        _ => return Err("not an arc barrel".into()),
    })
}

fn fractions(rows: &[Option<(f64, f64)>], rise: f64) -> Vec<Option<(f64, f64)>> {
    rows.iter()
        .map(|d| d.map(|(l, u)| (l / rise, u / rise)))
        .collect()
}

fn meet(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> bool {
    matches!((a, b), (Some(a), Some(b)) if a.0 <= b.1 && b.0 <= a.1)
}

#[test]
fn scaled_and_tiny_barrels_solve_at_the_same_depth_fraction() -> TestResult {
    // Lengths scale by 2^-300: critical Q by 2^-750 and normal Q by 2^-800 keep the fraction.
    let scale = 2f64.powi(-300);
    for culvert in evidence()? {
        let small = scaled(&culvert, scale)?;
        let rows = critical(vec![culvert, small], &[0.4, 0.4 * 2f64.powi(-750)])?;
        let [plain, tiny] = [&rows[0], &rows[1]];
        assert!(
            plain.status == CriticalDepthStatus::Solved && tiny.status == plain.status,
            "{rows:?}"
        );
        let a = fractions(&[plain.depth_m], culvert.rise_m());
        let b = fractions(&[tiny.depth_m], small.rise_m());
        assert!(meet(a[0], b[0]), "{culvert:?}: {a:?} and {b:?}");

        let q = [0.1, 0.1 * 2f64.powi(-800)];
        let rows = normal(vec![culvert, small], [&q, &[0.024; 2], &[1e-3; 2]])?;
        let [plain, tiny] = [&rows[0], &rows[1]];
        assert!(
            plain.status == NormalDepthStatus::Solved && tiny.status == plain.status,
            "{rows:?}"
        );
        let a = fractions(&[plain.depth_m], culvert.rise_m());
        let b = fractions(&[tiny.depth_m], small.rise_m());
        assert!(meet(a[0], b[0]), "{culvert:?}: {a:?} and {b:?}");

        // A represented discharge far above the free-surface range of a tiny barrel.
        let rows = critical(vec![small], &[1e-100])?;
        assert_eq!(
            rows[0].status,
            CriticalDepthStatus::AboveCrown,
            "{culvert:?}"
        );
        assert!(
            rows[0]
                .depth_m
                .is_some_and(|(lower, upper)| 0.0 < lower && upper == small.rise_m()),
            "{rows:?}"
        );
    }
    Ok(())
}

#[test]
fn circle_and_box_rows_keep_their_bits_beside_arc_rows() -> TestResult {
    let plain = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.5, 0.8)?,
        CulvertBarrel::circular(0.3)?,
    ];
    let mut mixed = Vec::new();
    for (culvert, arc) in plain.iter().zip(evidence()?) {
        mixed.push(*culvert);
        mixed.push(arc);
    }
    let flows = [0.5, 1.5, 0.05];
    let mixed_flows: Vec<f64> = flows.iter().flat_map(|q| [*q, *q]).collect();
    let alone = critical(plain.clone(), &flows)?;
    let beside = critical(mixed.clone(), &mixed_flows)?;
    for (a, b) in alone.iter().zip(beside.iter().step_by(2)) {
        assert_eq!(a.status, b.status);
        assert_eq!(
            a.depth_m.map(|(l, u)| (l.to_bits(), u.to_bits())),
            b.depth_m.map(|(l, u)| (l.to_bits(), u.to_bits()))
        );
    }
    let n = [0.013; 6];
    let s = [0.01; 6];
    let alone = normal(plain, [&flows, &n[..3], &s[..3]])?;
    let beside = normal(mixed, [&mixed_flows, &n, &s])?;
    for (a, b) in alone.iter().zip(beside.iter().step_by(2)) {
        assert_eq!(a.status, b.status);
        assert_eq!(
            a.depth_m.map(|(l, u)| (l.to_bits(), u.to_bits())),
            b.depth_m.map(|(l, u)| (l.to_bits(), u.to_bits()))
        );
    }
    Ok(())
}
