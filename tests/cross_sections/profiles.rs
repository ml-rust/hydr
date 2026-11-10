use hydr::channel::{CrossSectionDefinition, CrossSectionPoint, CrossSectionProperties};

pub fn definition(id: &str, points: &[(f64, f64)], cap: f64) -> CrossSectionDefinition {
    CrossSectionDefinition {
        id: id.into(),
        vertical_datum: "Synthetic analytical datum".into(),
        points: points
            .iter()
            .map(|&(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: cap,
        profile_provenance: "Explicit analytical station/elevation profile".into(),
    }
}

pub fn rectangle() -> CrossSectionDefinition {
    definition(
        "rectangle",
        &[(0.0, 4.0), (0.0, 0.0), (2.0, 0.0), (2.0, 4.0)],
        4.0,
    )
}

pub fn trapezoid() -> CrossSectionDefinition {
    definition(
        "trapezoid",
        &[(0.0, 3.0), (3.0, 0.0), (5.0, 0.0), (8.0, 3.0)],
        3.0,
    )
}

pub fn triangle() -> CrossSectionDefinition {
    definition("triangle", &[(-3.0, 3.0), (0.0, 0.0), (3.0, 3.0)], 3.0)
}

pub fn bench() -> CrossSectionDefinition {
    definition(
        "bench",
        &[
            (-3.0, 3.0),
            (-3.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (3.0, 1.0),
            (3.0, 3.0),
        ],
        2.5,
    )
}

pub fn all() -> Vec<CrossSectionDefinition> {
    vec![rectangle(), trapezoid(), triangle(), bench()]
}

#[derive(Clone, Copy)]
pub struct Expected {
    pub area: f64,
    pub width: f64,
    pub perimeter: f64,
    pub integral: f64,
}

pub fn rectangle_expected(h: f64) -> Expected {
    Expected {
        area: 2.0 * h,
        width: 2.0,
        perimeter: 2.0 + 2.0 * h,
        integral: h * h,
    }
}

pub fn trapezoid_expected(h: f64) -> Expected {
    Expected {
        area: 2.0 * h + h * h,
        width: 2.0 + 2.0 * h,
        perimeter: 2.0 + 2.0 * h * 2.0_f64.sqrt(),
        integral: h * h + h * h * h / 3.0,
    }
}

pub fn triangle_expected(h: f64) -> Expected {
    Expected {
        area: h * h,
        width: 2.0 * h,
        perimeter: 2.0 * h * 2.0_f64.sqrt(),
        integral: h * h * h / 3.0,
    }
}

pub fn bench_expected(h: f64) -> Expected {
    if h <= 1.0 {
        rectangle_expected(h)
    } else {
        Expected {
            area: 2.0 * h + 4.0 * (h - 1.0),
            width: 6.0,
            perimeter: 8.0 + 2.0 * (h - 1.0),
            integral: h * h + 2.0 * (h - 1.0) * (h - 1.0),
        }
    }
}

pub fn near(actual: f64, expected: f64) {
    let allowance = 1e-12 + 1e-12 * expected.abs();
    assert!(
        actual.is_finite() && (actual - expected).abs() <= allowance,
        "actual={actual:.17e}, expected={expected:.17e}, allowance={allowance:.17e}"
    );
    if expected > 0.0 {
        assert!(actual > 0.0, "Positive geometry became zero");
    }
}

pub fn assert_wet(
    actual: &CrossSectionProperties,
    expected: Expected,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_properties(actual, expected, 1.0, false)
}

pub fn assert_scaled_wet(
    actual: &CrossSectionProperties,
    expected: Expected,
    scale: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_properties(actual, expected, scale, true)
}

fn normalized_near(actual: f64, expected: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 1e-12,
        "normalized actual={actual:.17e}, expected={expected:.17e}, allowance=1e-12"
    );
    if expected > 0.0 {
        assert!(actual > 0.0, "Positive normalized geometry became zero");
    }
}

fn assert_properties(
    actual: &CrossSectionProperties,
    expected: Expected,
    scale: f64,
    normalized: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let check: fn(f64, f64) = if normalized { normalized_near } else { near };
    assert!(!actual.dry);
    check(actual.area_m2 / (scale * scale), expected.area);
    check(
        actual.top_width_m.ok_or("Wet section lacks top width")? / scale,
        expected.width,
    );
    check(actual.wetted_perimeter_m / scale, expected.perimeter);
    check(
        actual.hydrostatic_integral_m3 / (scale * scale * scale),
        expected.integral,
    );
    check(
        actual
            .hydraulic_radius_m
            .ok_or("Wet section lacks hydraulic radius")?
            / scale,
        expected.area / expected.perimeter,
    );
    check(
        actual
            .hydraulic_depth_m
            .ok_or("Wet section lacks hydraulic depth")?
            / scale,
        expected.area / expected.width,
    );
    Ok(())
}

pub fn assert_dry(actual: &CrossSectionProperties, stage: f64) {
    assert!(actual.dry);
    assert_eq!(actual.query_stage_m.to_bits(), stage.to_bits());
    assert_eq!(actual.area_m2, 0.0);
    assert_eq!(actual.wetted_perimeter_m, 0.0);
    assert_eq!(actual.hydrostatic_integral_m3, 0.0);
    assert_eq!(actual.top_width_m, None);
    assert_eq!(actual.hydraulic_radius_m, None);
    assert_eq!(actual.hydraulic_depth_m, None);
}

pub fn assert_same(a: &CrossSectionProperties, b: &CrossSectionProperties) {
    assert_eq!(a.section_id, b.section_id);
    assert_eq!(a.query_stage_m.to_bits(), b.query_stage_m.to_bits());
    assert_eq!(a.dry, b.dry);
    assert_geometry_bits(a, b);
}

pub fn assert_geometry_bits(a: &CrossSectionProperties, b: &CrossSectionProperties) {
    assert_eq!(a.area_m2.to_bits(), b.area_m2.to_bits());
    assert_eq!(
        a.wetted_perimeter_m.to_bits(),
        b.wetted_perimeter_m.to_bits()
    );
    assert_eq!(
        a.hydrostatic_integral_m3.to_bits(),
        b.hydrostatic_integral_m3.to_bits()
    );
    assert_eq!(
        a.top_width_m.map(f64::to_bits),
        b.top_width_m.map(f64::to_bits)
    );
    assert_eq!(
        a.hydraulic_radius_m.map(f64::to_bits),
        b.hydraulic_radius_m.map(f64::to_bits)
    );
    assert_eq!(
        a.hydraulic_depth_m.map(f64::to_bits),
        b.hydraulic_depth_m.map(f64::to_bits)
    );
}
