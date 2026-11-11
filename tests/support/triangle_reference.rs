//! Independent cell averages for prismatic triangular-channel acceptance tests.
//!
//! Pressure is p = g A h / 3, with A = k h² and c² = g h / 2.
//! Inputs require finite x_left < x_right, finite t >= 0, and finite positive g and k.
//! Calculations require representable wave positions and integrals.
//! Each return value contains average area [m²] and average discharge [m³/s].
//! No production geometry, flux, wave bounds, or roots enter these references.
//! Three-point Gauss quadrature integrates each polynomial segment exactly through degree five.
//! Wave splitting prevents quadrature across discontinuities or vacuum boundaries.

/// Returns conserved cell averages for depth-one water released onto a dry bed.
pub fn dry_release(x_left: f64, x_right: f64, t: f64, g: f64, k: f64) -> (f64, f64) {
    if t == 0.0 {
        return step_average(x_left, x_right, 0.0, (k, 0.0), (0.0, 0.0));
    }
    let c = (g / 2.0).sqrt();
    let head = -c * t;
    let front = 4.0 * c * t;
    let width = x_right - x_left;
    let constant_length = (x_right.min(head) - x_left).max(0.0);
    let mut area = k * constant_length;
    let mut discharge = 0.0;
    let a = x_left.max(head);
    let b = x_right.min(front);
    if b > a {
        let integral = fan_integral(a, b, front, 5.0 * c * t, |y| {
            let y4 = y * y * y * y;
            (k * y4, 4.0 * c * k * y4 * (1.0 - y))
        });
        area += integral.0;
        discharge += integral.1;
    }
    (area / width, discharge / width)
}

/// Returns conserved cell averages for separating depth-one states at velocities ±5c.
pub fn twofan_vacuum(x_left: f64, x_right: f64, t: f64, g: f64, k: f64) -> (f64, f64) {
    let c = (g / 2.0).sqrt();
    if t == 0.0 {
        return step_average(x_left, x_right, 0.0, (k, -5.0 * c * k), (k, 5.0 * c * k));
    }
    let left = vacuum_left_integral(x_left, x_right, t, c, k);
    let reflected = vacuum_left_integral(-x_right, -x_left, t, c, k);
    let width = x_right - x_left;
    (
        (left.0 + reflected.0) / width,
        (left.1 - reflected.1) / width,
    )
}

/// Returns conserved cell averages across the exact shock between areas four and one.
/// The speed and left velocity assume k = 1, as frozen in the acceptance case.
pub fn exact_wetshock(x_left: f64, x_right: f64, t: f64, g: f64) -> (f64, f64) {
    let u_left = (7.0 * g / 4.0).sqrt();
    let shock = (4.0 * u_left / 3.0) * t;
    step_average(x_left, x_right, shock, (4.0, 4.0 * u_left), (1.0, 0.0))
}

/// Converts average conserved area into the depth represented by a finite-volume cell.
/// This differs from averaging point depths inside a fan or across a shock.
pub fn represented_depth(average_area: f64, k: f64) -> f64 {
    (average_area / k).sqrt()
}

/// Returns length-weighted mean absolute errors in depth [m] and discharge [m³/s].
/// Each entry contains (cell length, numerical area, numerical discharge, reference area, reference discharge).
/// Entries require finite positive lengths, nonnegative areas, and finite discharges.
/// Empty input returns None.
pub fn mean_errors(
    cells: impl IntoIterator<Item = (f64, f64, f64, f64, f64)>,
    k: f64,
) -> Option<(f64, f64)> {
    let mut length = 0.0;
    let mut depth_error = 0.0;
    let mut discharge_error = 0.0;
    for (dx, numerical_area, numerical_q, reference_area, reference_q) in cells {
        length += dx;
        depth_error += dx
            * (represented_depth(numerical_area, k) - represented_depth(reference_area, k)).abs();
        discharge_error += dx * (numerical_q - reference_q).abs();
    }
    (length > 0.0).then(|| (depth_error / length, discharge_error / length))
}

fn step_average(
    x_left: f64,
    x_right: f64,
    transition: f64,
    left: (f64, f64),
    right: (f64, f64),
) -> (f64, f64) {
    let width = x_right - x_left;
    let left_length = (x_right.min(transition) - x_left).clamp(0.0, width);
    let right_length = width - left_length;
    (
        (left_length * left.0 + right_length * right.0) / width,
        (left_length * left.1 + right_length * right.1) / width,
    )
}

fn vacuum_left_integral(x_left: f64, x_right: f64, t: f64, c: f64, k: f64) -> (f64, f64) {
    let head = -6.0 * c * t;
    let front = -c * t;
    let constant_length = (x_right.min(head) - x_left).max(0.0);
    let mut area = k * constant_length;
    let mut discharge = -5.0 * c * k * constant_length;
    let a = x_left.max(head);
    let b = x_right.min(front);
    if b > a {
        let integral = fan_integral(a, b, front, 5.0 * c * t, |y| {
            let y4 = y * y * y * y;
            (k * y4, -c * k * y4 * (1.0 + 4.0 * y))
        });
        area += integral.0;
        discharge += integral.1;
    }
    (area, discharge)
}

// Integrate in normalized fan coordinates without subtracting close polynomial primitives.
fn fan_integral(
    a: f64,
    b: f64,
    front: f64,
    fan_width: f64,
    state: impl Fn(f64) -> (f64, f64),
) -> (f64, f64) {
    let half_width = 0.5 * (b - a);
    let y_a = (front - a) / fan_width;
    let y_b = (front - b) / fan_width;
    let y_mid = y_b + 0.5 * (y_a - y_b);
    let y_offset = 0.5 * (y_a - y_b) * (3.0_f64 / 5.0).sqrt();
    let left = state(y_mid + y_offset);
    let middle = state(y_mid);
    let right = state(y_mid - y_offset);
    (
        half_width * ((5.0 / 9.0) * (left.0 + right.0) + (8.0 / 9.0) * middle.0),
        half_width * ((5.0 / 9.0) * (left.1 + right.1) + (8.0 / 9.0) * middle.1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 9.81;
    const K: f64 = 1.0;

    // Tolerances carry the units of the checked quantity through its supplied scale.
    fn close(actual: f64, expected: f64, dimensional_scale: f64) {
        let tolerance = 256.0 * f64::EPSILON * dimensional_scale;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:e} != {expected:e}, tolerance {tolerance:e}"
        );
    }

    #[test]
    fn initial_cells_integrate_both_sides_of_the_discontinuity() {
        let c = (G / 2.0).sqrt();
        assert_eq!(dry_release(-3.0, 1.0, 0.0, G, K), (0.75, 0.0));
        let vacuum = twofan_vacuum(-3.0, 1.0, 0.0, G, K);
        close(vacuum.0, 1.0, 1.0);
        close(vacuum.1, -2.5 * c, 5.0 * c);
        let shock = exact_wetshock(-3.0, 1.0, 0.0, G);
        close(shock.0, 3.25, 4.0);
        close(
            shock.1,
            3.0 * (7.0 * G / 4.0).sqrt(),
            4.0 * (7.0 * G / 4.0).sqrt(),
        );
    }

    #[test]
    fn dry_release_preserves_volume_and_accumulates_pressure_momentum() {
        let t = 4.0;
        let average = dry_release(-100.0, 100.0, t, G, K);
        close(average.0 * 200.0, 100.0 * K, 100.0 * K);
        close(average.1 * 200.0, G * K * t / 3.0, G * K * t);
        let c = (G / 2.0).sqrt();
        let fan = dry_release(-c * t, 4.0 * c * t, t, G, K);
        close(fan.0 * 5.0 * c * t, K * c * t, K * c * t);
        assert_eq!(dry_release(4.0 * c * t, 100.0, t, G, K), (0.0, 0.0));
    }

    #[test]
    fn vacuum_preserves_reflection_and_front_support() {
        let t = 1.0;
        let c = (G / 2.0).sqrt();
        let symmetric = twofan_vacuum(-6.0 * c * t, 6.0 * c * t, t, G, K);
        close(
            symmetric.0 * 12.0 * c * t,
            2.0 * K * c * t,
            12.0 * K * c * t,
        );
        close(symmetric.1, 0.0, 5.0 * K * c);
        assert_eq!(twofan_vacuum(-c * t, c * t, t, G, K), (0.0, 0.0));
        let left = twofan_vacuum(-6.0 * c * t, -c * t, t, G, K);
        close(
            left.1 * 5.0 * c * t,
            -(13.0 / 3.0) * K * c * c * t,
            5.0 * K * c * c * t,
        );
    }

    #[test]
    fn split_cells_equal_whole_interval_integrals() {
        type Reference = fn(f64, f64, f64, f64, f64) -> (f64, f64);
        let references: [Reference; 2] = [dry_release, twofan_vacuum];
        for reference in references {
            let whole = reference(-30.0, 30.0, 1.0, G, K);
            let left = reference(-30.0, 0.7, 1.0, G, K);
            let right = reference(0.7, 30.0, 1.0, G, K);
            close(whole.0 * 60.0, left.0 * 30.7 + right.0 * 29.3, 60.0 * K);
            close(
                whole.1 * 60.0,
                left.1 * 30.7 + right.1 * 29.3,
                300.0 * K * (G / 2.0).sqrt(),
            );
        }
    }

    #[test]
    fn shock_integrates_partial_cells_and_rankine_hugoniot_balances() {
        let u = (7.0 * G / 4.0).sqrt();
        let t = 4.0;
        let s = 4.0 * u / 3.0;
        let average = exact_wetshock(s * t - 3.0, s * t + 1.0, t, G);
        close(average.0, 3.25, 4.0);
        close(average.1, 3.0 * u, 4.0 * u);
        close(s * (4.0 - 1.0), 4.0 * u, 4.0 * u);
        let left_flux = 4.0 * u * u + G * 4.0 * 2.0 / 3.0;
        let right_flux = G / 3.0;
        close(s * 4.0 * u, left_flux - right_flux, left_flux);
        let whole = exact_wetshock(-100.0, 100.0, t, G);
        close(whole.0 * 200.0, 500.0 + 4.0 * u * t, 800.0);
        close(
            whole.1 * 200.0,
            400.0 * u + (left_flux - right_flux) * t,
            800.0 * u,
        );
        let left = exact_wetshock(-100.0, 0.7, t, G);
        let right = exact_wetshock(0.7, 100.0, t, G);
        close(whole.0 * 200.0, left.0 * 100.7 + right.0 * 99.3, 800.0);
        close(whole.1 * 200.0, left.1 * 100.7 + right.1 * 99.3, 800.0 * u);
    }

    #[test]
    fn error_norm_uses_depth_of_average_area_and_length_weights() {
        close(represented_depth(2.5, K), 2.5_f64.sqrt(), 2.0);
        let errors = mean_errors([(1.0, 4.0, 3.0, 1.0, 1.0), (3.0, 1.0, 1.0, 1.0, 1.0)], K);
        assert_eq!(errors, Some((0.25, 0.5)));
        assert_eq!(mean_errors([], K), None);
    }
}
