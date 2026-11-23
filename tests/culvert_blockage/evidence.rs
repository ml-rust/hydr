//! Frozen fixture identity and the invariants every blockage reference must hold.

use super::composite_rows::{
    COMPOSITE_SHA256, EXAMPLES, FRICTION, HEAD_LOSS_RELATIVE_TOLERANCE,
    IDENTITY_RELATIVE_TOLERANCE, PRINTED_RESULT, PUBLISHED_TOLERANCE,
};
use super::depth_rows::{CERTIFICATES, CRITICAL, CRITICAL_ROWS, Class, DEPTHS_SHA256, NORMAL_ROWS};
use super::inlet_rows::{self, FACE_FRACTIONS, INLET_SHA256};
use super::outlet_rows::{self, FACE_LOSSES, Flow, OUTLET_SHA256};
use super::section_rows::{LAYOUT, ROWS, Row, SECTIONS, SECTIONS_SHA256, Section, Shape, State};

/// Quantity indices in row order.
const AREA: usize = 0;
const TOP_WIDTH: usize = 1;
const WALL: usize = 2;
const PERIMETER: usize = 3;
const WEIGHTED: usize = 4;
const RADIUS: usize = 5;
const MOMENT: usize = 6;

/// Outward bounds as floats.
type Bounds = (f64, f64);

fn bounds(pair: [u64; 2]) -> Bounds {
    (f64::from_bits(pair[0]), f64::from_bits(pair[1]))
}

fn midpoint(pair: [u64; 2]) -> f64 {
    let (down, up) = bounds(pair);
    0.5 * (down + up)
}

fn quantity(row: &Row, index: usize) -> Bounds {
    row.values
        .get(index)
        .copied()
        .map_or((f64::NAN, f64::NAN), bounds)
}

/// Unreduced rise: the only input of a circle, the second input of every other shape.
fn rise(section: &Section) -> f64 {
    let index = usize::from(section.shape != Shape::Circular);
    section
        .inputs
        .get(index)
        .map_or(f64::NAN, |bits| f64::from_bits(*bits))
}

/// Bits of one named section value.
fn layout(section: usize, name: &str) -> Option<[u64; 2]> {
    LAYOUT
        .iter()
        .find(|value| value.section == section && value.name == name)
        .map(|value| value.bounds)
}

fn rows_of(section: usize) -> Vec<&'static Row> {
    ROWS.iter().filter(|row| row.section == section).collect()
}

/// `true` when `[a.0 + b.0, a.1 + b.1]`, widened one ulp each way, meets `c`.
fn sum_meets(a: Bounds, b: Bounds, c: Bounds) -> bool {
    (a.0 + b.0).next_down() <= c.1 && c.0 <= (a.1 + b.1).next_up()
}

#[test]
fn fixtures_are_the_frozen_files() {
    assert_eq!(
        SECTIONS_SHA256,
        "7e63173d02e2501eb7b1adf9ffeb99129724818e0da93d088b57c1862df7a0b2"
    );
    assert_eq!(
        DEPTHS_SHA256,
        "5937323aa561476fd254667e5adfa2cd84857818ed80b2f1364f0b0938dfb1c0"
    );
    assert_eq!(
        INLET_SHA256,
        "4c55773266053a0bc7266825721a323ac3275c2b207e9f0efb2d86aa43b49c9f"
    );
    assert_eq!(
        OUTLET_SHA256,
        "ecf9a3ca1b393be347640db2351643ec3c149df6862c58691e3ff9ad908d3034"
    );
    assert_eq!(
        COMPOSITE_SHA256,
        "51247e0d05a3fdd1bd0c69a3d171f03ee41e20b8fad2691e02289170d1bbb42a"
    );
}

#[test]
fn row_counts_match_the_evidence() {
    assert_eq!(SECTIONS.len(), 18);
    assert_eq!(ROWS.len(), 1383);
    assert_eq!(LAYOUT.len(), 227);
    assert_eq!(CRITICAL.len(), SECTIONS.len());
    assert_eq!(CERTIFICATES.len(), 3 * SECTIONS.len());
    assert_eq!(CRITICAL_ROWS.len(), 110);
    assert_eq!(NORMAL_ROWS.len(), 324);
    assert_eq!(inlet_rows::ROWS.len(), 520);
    assert_eq!(inlet_rows::ANCHORS.len(), 26);
    assert!(inlet_rows::ANCHORS.iter().all(|a| a.transition_rising));
    assert_eq!(inlet_rows::CODES.len(), 13);
    assert_eq!(outlet_rows::ROWS.len(), 162);
    assert_eq!(FACE_LOSSES.len(), 16);
    assert_eq!(super::composite_rows::ROWS.len(), 4 * SECTIONS.len());
    assert_eq!(FRICTION.len(), SECTIONS.len());
    assert_eq!(EXAMPLES.len(), 4);
}

#[test]
fn bed_and_reduced_rise_are_exact_inputs() {
    for section in SECTIONS {
        let name = section.name;
        let rise = rise(section);
        let bed = f64::from_bits(section.sediment_depth);
        let reduced = f64::from_bits(section.reduced_rise);
        assert!(
            bed + reduced == rise && rise - bed == reduced && rise - reduced == bed,
            "{name}"
        );
        assert!((0.0..rise).contains(&bed), "{name}");
        let floored = matches!(section.shape, Shape::Box | Shape::Arch);
        assert!(
            floored || bed > 0.0,
            "{name}: a barrel without a floor needs a bed"
        );
        let ratio = f64::from_bits(section.bed_manning) / f64::from_bits(section.wall_manning);
        assert_eq!(f64::from_bits(section.roughness_ratio), ratio, "{name}");
    }
}

#[test]
fn every_section_runs_from_the_bed_to_above_the_crown() {
    for (index, section) in SECTIONS.iter().enumerate() {
        let name = section.name;
        let rows = rows_of(index);
        assert!(
            rows.first()
                .is_some_and(|row| row.state == State::Dry && row.depth == 0),
            "{name}"
        );
        assert!(rows.iter().any(|row| row.state == State::Free), "{name}");
        assert_eq!(
            rows.iter().filter(|row| row.state == State::Full).count(),
            2,
            "{name}"
        );
        let reduced = f64::from_bits(section.reduced_rise);
        for row in &rows {
            let full = f64::from_bits(row.depth) >= reduced;
            assert_eq!(full, row.state == State::Full, "{name} at {}", row.fraction);
        }
        assert!(
            rows.windows(2)
                .all(|w| f64::from_bits(w[0].depth) < f64::from_bits(w[1].depth)),
            "{name}"
        );
    }
}

#[test]
fn area_wall_and_moment_never_fall_with_depth() {
    for (index, section) in SECTIONS.iter().enumerate() {
        let rows = rows_of(index);
        for pair in rows.windows(2) {
            let (low, high) = (pair[0], pair[1]);
            for q in [AREA, WALL, MOMENT] {
                let (a, b) = (quantity(low, q), quantity(high, q));
                assert!(
                    a.0 <= b.0 && a.1 <= b.1,
                    "{} {} -> {}: quantity {q}",
                    section.name,
                    low.fraction,
                    high.fraction
                );
            }
        }
    }
}

#[test]
fn perimeter_is_wall_plus_bed_and_radius_is_area_over_perimeter() {
    for row in ROWS {
        let name = SECTIONS.get(row.section).map_or("?", |s| s.name);
        let bed = layout(row.section, "bed_width").map_or((f64::NAN, f64::NAN), bounds);
        let power = layout(row.section, "ratio_power").map_or((f64::NAN, f64::NAN), bounds);
        let wall = quantity(row, WALL);
        let weighted_bed = ((power.0 * bed.0).next_down(), (power.1 * bed.1).next_up());
        assert!(
            sum_meets(wall, bed, quantity(row, PERIMETER)),
            "{name} at {}: P",
            row.fraction
        );
        assert!(
            sum_meets(wall, weighted_bed, quantity(row, WEIGHTED)),
            "{name} at {}: P_n",
            row.fraction
        );
        let (area, perimeter, radius) = (
            quantity(row, AREA),
            quantity(row, PERIMETER),
            quantity(row, RADIUS),
        );
        let lower = (area.0 / perimeter.1).next_down();
        let upper = (area.1 / perimeter.0).next_up();
        assert!(
            lower <= radius.1 && radius.0 <= upper,
            "{name} at {}: R",
            row.fraction
        );
    }
}

#[test]
fn dry_and_full_rows_hold_the_bed_and_crown_values() {
    for row in ROWS {
        let name = SECTIONS.get(row.section).map_or("?", |s| s.name);
        let value = |q: usize| row.values.get(q).copied();
        match row.state {
            State::Dry => {
                for q in [AREA, WALL, RADIUS, MOMENT] {
                    assert_eq!(value(q), Some([0, 0]), "{name}: dry quantity {q}");
                }
                assert_eq!(value(TOP_WIDTH), layout(row.section, "bed_width"), "{name}");
                assert_eq!(value(PERIMETER), layout(row.section, "bed_width"), "{name}");
                assert_eq!(
                    value(WEIGHTED),
                    layout(row.section, "weighted_bed_width"),
                    "{name}"
                );
            }
            State::Full => {
                assert_eq!(value(TOP_WIDTH), Some([0, 0]), "{name}");
                let names = [
                    (AREA, "full_area"),
                    (WALL, "full_wall_perimeter"),
                    (PERIMETER, "full_perimeter"),
                    (WEIGHTED, "full_weighted_perimeter"),
                    (RADIUS, "full_radius"),
                ];
                for (q, key) in names {
                    assert_eq!(value(q), layout(row.section, key), "{name}: {key}");
                }
                if row.fraction == "1" {
                    assert_eq!(value(MOMENT), layout(row.section, "full_moment"), "{name}");
                }
            }
            State::Free => {
                assert!(
                    row.relative_width.iter().all(|w| w.is_finite() && *w > 0.0),
                    "{name} at {}",
                    row.fraction
                );
            }
        }
    }
}

#[test]
fn depth_certificates_cover_every_section() {
    for (index, (section, critical)) in SECTIONS.iter().zip(CRITICAL).enumerate() {
        let name = section.name;
        assert_eq!(critical.name, name);
        assert!(f64::from_bits(critical.minimum) > 0.0, "{name}");
        let certificates: Vec<_> = CERTIFICATES.iter().filter(|c| c.section == index).collect();
        assert_eq!(certificates.len(), 3, "{name}");
        if section.shape == Shape::Box {
            assert!(
                certificates
                    .iter()
                    .all(|c| c.sign_changes == 0 && c.box_minimum.is_some() && c.peak.is_none()),
                "{name}"
            );
            continue;
        }
        assert!(
            certificates
                .iter()
                .all(|c| c.sign_changes == 1 && c.rising.len() == 41),
            "{name}"
        );
        // The ratios run smooth bed, equal, rough bed. A rougher bed raises the peak.
        let peaks: Vec<f64> = certificates
            .iter()
            .filter_map(|c| c.peak)
            .map(|p| f64::from_bits(p[0]))
            .collect();
        assert_eq!(peaks.len(), 3, "{name}");
        assert!(peaks.windows(2).all(|w| w[0] < w[1]), "{name}: {peaks:?}");
        assert!(peaks.iter().all(|s| (0.5..1.0).contains(s)), "{name}");
    }
}

#[test]
fn depth_rows_hold_every_class() {
    for (index, section) in SECTIONS.iter().enumerate() {
        let name = section.name;
        let reduced = f64::from_bits(section.reduced_rise);
        let critical: Vec<_> = CRITICAL_ROWS
            .iter()
            .filter(|r| r.section == index)
            .collect();
        let normal: Vec<_> = NORMAL_ROWS.iter().filter(|r| r.section == index).collect();
        assert!(critical.iter().any(|r| r.class == Class::Dry), "{name}");
        assert!(
            normal.iter().any(|r| r.class == Class::NonPositiveSlope),
            "{name}"
        );
        assert!(
            normal.iter().any(|r| r.class == Class::AboveCapacity),
            "{name}"
        );
        if section.shape == Shape::Box {
            assert!(
                critical.iter().any(|r| r.class == Class::AboveCrown),
                "{name}"
            );
        } else {
            assert!(normal.iter().any(|r| r.class == Class::NearPeak), "{name}");
        }
        let free = critical
            .iter()
            .filter(|r| r.class == Class::Free)
            .map(|r| (r.root, r.relative_width))
            .chain(
                normal
                    .iter()
                    .filter(|r| r.class == Class::Free)
                    .map(|r| (r.root, r.relative_width)),
            );
        for (root, width) in free {
            let (down, up) = root.map_or((f64::NAN, f64::NAN), bounds);
            assert!(0.0 < down && down <= up && up <= reduced, "{name}");
            assert!(width.is_finite() && width > 0.0, "{name}");
        }
    }
}

#[test]
fn inlet_rows_scale_discharge_by_the_open_share() {
    for row in inlet_rows::ROWS {
        let discharge = f64::from_bits(row.discharge);
        for (blocked, fraction) in row.blocked_discharge.iter().zip(FACE_FRACTIONS) {
            let open = 1.0 - f64::from_bits(fraction);
            let (down, up) = bounds(*blocked);
            assert!(
                (down..=up).contains(&(open * discharge)),
                "code {} at {}",
                row.code,
                row.label
            );
        }
        assert_eq!(
            row.blocked_discharge.first(),
            Some(&[row.discharge, row.discharge])
        );
        let (down, up) = bounds(row.headwater);
        assert!(
            0.0 < down && down <= up,
            "code {} at {}",
            row.code,
            row.label
        );
    }
    for section in inlet_rows::SECTIONS {
        assert!(
            SECTIONS.iter().any(|s| s.name == section.name
                && s.sediment_depth == section.sediment_depth
                && s.reduced_rise == section.reduced_rise),
            "{}",
            section.name
        );
    }
}

#[test]
fn outlet_face_loss_keeps_ke_and_raises_headwater() {
    for loss in FACE_LOSSES {
        if loss.face_fraction == 0 {
            assert_eq!(loss.face_ke, [loss.ke, loss.ke]);
        } else {
            assert!(f64::from_bits(loss.face_ke[0]) > f64::from_bits(loss.ke));
        }
    }
    for row in outlet_rows::ROWS {
        if row.face_fraction == 0 {
            assert_eq!(row.face_ke, [row.ke, row.ke], "{}", row.name);
        }
        if row.name == "full" {
            assert_eq!(row.flow, Flow::Full);
        }
        let blocked = outlet_rows::ROWS.iter().find(|other| {
            other.section == row.section
                && other.name == row.name
                && other.discharge == row.discharge
                && other.tailwater == row.tailwater
                && f64::from_bits(other.face_fraction) > f64::from_bits(row.face_fraction)
        });
        if let Some(other) = blocked {
            assert!(
                f64::from_bits(other.headwater[0]) > f64::from_bits(row.headwater[1]),
                "{} on section {}",
                row.name,
                row.section
            );
        }
    }
    for section in outlet_rows::SECTIONS {
        assert!(
            SECTIONS.iter().any(|s| s.name == section.name
                && s.sediment_depth == section.sediment_depth
                && s.roughness_ratio == section.roughness_ratio),
            "{}",
            section.name
        );
    }
}

#[test]
fn composite_example_rounds_to_the_printed_value() {
    for example in EXAMPLES {
        let (down, up) = bounds(example.composite_n);
        assert!(
            (down - PRINTED_RESULT).abs() <= PUBLISHED_TOLERANCE
                && (up - PRINTED_RESULT).abs() <= PUBLISHED_TOLERANCE,
            "{} {}",
            example.perimeters,
            example.exponent
        );
        // Exponent 2/3 gives 0.0213 and the printed 0.67 gives 0.0209.
        match example.exponent {
            "2/3" => assert!(down > 0.0212),
            _ => assert!(up < 0.0210),
        }
    }
}

#[test]
fn weighted_perimeter_reproduces_equation_3_8() {
    for row in super::composite_rows::ROWS {
        let section = SECTIONS.get(row.section);
        let wall_n = section.map_or(f64::NAN, |s| f64::from_bits(s.wall_manning));
        let ratio = section.map_or(f64::NAN, |s| f64::from_bits(s.roughness_ratio));
        let composite = midpoint(row.composite_n);
        let perimeter = midpoint(row.perimeter);
        let weighted = midpoint(row.weighted_perimeter);
        let identity = wall_n.powi(2) * weighted.powf(4.0 / 3.0);
        let gap = (composite.powi(2) * perimeter.powf(4.0 / 3.0) - identity).abs();
        assert!(
            gap <= IDENTITY_RELATIVE_TOLERANCE * identity,
            "section {} at {}",
            row.section,
            row.fraction
        );
        let bed_n = ratio * wall_n;
        let parts = midpoint(row.wall_perimeter) * wall_n.powf(1.5)
            + midpoint(row.bed_width) * bed_n.powf(1.5);
        let direct = (parts / perimeter).powf(2.0 / 3.0);
        assert!(
            (direct - composite).abs() <= IDENTITY_RELATIVE_TOLERANCE * composite,
            "section {} at {}",
            row.section,
            row.fraction
        );
    }
}

#[test]
fn full_head_loss_is_composite_manning_friction() {
    for friction in FRICTION {
        let discharge = f64::from_bits(friction.discharge);
        let length = f64::from_bits(friction.length);
        let velocity = discharge / midpoint(friction.full_area);
        let n = midpoint(friction.composite_n);
        let expected =
            n * n * velocity * velocity * length / midpoint(friction.full_radius).powf(4.0 / 3.0);
        let found = midpoint(friction.head_loss);
        assert!(
            (found - expected).abs() <= HEAD_LOSS_RELATIVE_TOLERANCE * found,
            "section {}",
            friction.section
        );
    }
}
