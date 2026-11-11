//! Independent references for the proposed wet-subcritical channel boundary fixtures.
//!
//! Right fixture: x in [-20, 0], initial depth one and velocity zero, boundary depth H.
//! Rectangle uses g=1 and width=3. Triangle uses g=2 and k=1.
//! Cell averages integrate area and discharge before recovering represented depth.
//! Inputs require positive finite H and finite x_left < x_right <= 0 with finite t >= 0.
//! No Hydr, Numr, Solvr, production trace function, or numerical root enters this reference.

#[derive(Clone, Copy, Debug)]
pub enum Geometry {
    Rectangular,
    Triangular,
}

impl Geometry {
    pub fn area(self, depth: f64) -> f64 {
        match self {
            Self::Rectangular => 3.0 * depth,
            Self::Triangular => depth * depth,
        }
    }

    pub fn pressure(self, depth: f64) -> f64 {
        match self {
            Self::Rectangular => 1.5 * depth * depth,
            Self::Triangular => (2.0 / 3.0) * depth * depth * depth,
        }
    }

    pub fn represented_depth(self, area: f64) -> f64 {
        match self {
            Self::Rectangular => area / 3.0,
            Self::Triangular => area.sqrt(),
        }
    }

    pub fn invariant_factor(self) -> f64 {
        match self {
            Self::Rectangular => 2.0,
            Self::Triangular => 4.0,
        }
    }

    pub fn critical_depth(self) -> f64 {
        match self {
            Self::Rectangular => 4.0 / 9.0,
            Self::Triangular => 16.0 / 25.0,
        }
    }

    pub fn critical_outward_discharge(self) -> f64 {
        match self {
            Self::Rectangular => 8.0 / 9.0,
            Self::Triangular => 1024.0 / 3125.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Wave {
    Constant,
    Rarefaction { head: f64, tail: f64 },
    Shock { speed: f64 },
}

#[derive(Clone, Copy, Debug)]
pub struct Trace {
    pub depth: f64,
    pub area: f64,
    pub outward_velocity: f64,
    pub outward_discharge: f64,
    pub momentum_flux: f64,
    pub wave: Wave,
}

/// Returns (velocity, discharge, momentum flux, shock speed) from conserved jump data.
/// Inputs describe a deeper boundary state connected to initial rest.
pub fn shock_from_conserved_jump(
    initial_area: f64,
    initial_pressure: f64,
    boundary_area: f64,
    boundary_pressure: f64,
) -> (f64, f64, f64, f64) {
    let area_jump = boundary_area - initial_area;
    let pressure_jump = boundary_pressure - initial_pressure;
    let velocity = -(pressure_jump * area_jump / (boundary_area * initial_area)).sqrt();
    let discharge = boundary_area * velocity;
    let momentum_flux = discharge * velocity + boundary_pressure;
    let speed = discharge / area_jump;
    (velocity, discharge, momentum_flux, speed)
}

pub fn trace(geometry: Geometry, boundary_depth: f64) -> Trace {
    let area = geometry.area(boundary_depth);
    let initial_area = geometry.area(1.0);
    let sound = boundary_depth.sqrt();
    let (outward_velocity, wave) = if boundary_depth < 1.0 {
        let velocity = geometry.invariant_factor() * (1.0 - sound);
        (
            velocity,
            Wave::Rarefaction {
                head: -1.0,
                tail: velocity - sound,
            },
        )
    } else if boundary_depth > 1.0 {
        let (velocity, _, _, speed) = shock_from_conserved_jump(
            initial_area,
            geometry.pressure(1.0),
            area,
            geometry.pressure(boundary_depth),
        );
        (velocity, Wave::Shock { speed })
    } else {
        (0.0, Wave::Constant)
    };
    let outward_discharge = area * outward_velocity;
    Trace {
        depth: boundary_depth,
        area,
        outward_velocity,
        outward_discharge,
        momentum_flux: outward_discharge * outward_velocity + geometry.pressure(boundary_depth),
        wave,
    }
}

/// Returns (average area, average longitudinal discharge) for the right-end fixture.
pub fn right_cell_average(
    geometry: Geometry,
    boundary_depth: f64,
    x_left: f64,
    x_right: f64,
    time: f64,
) -> (f64, f64) {
    let initial_area = geometry.area(1.0);
    if time == 0.0 {
        return (initial_area, 0.0);
    }
    let boundary = trace(geometry, boundary_depth);
    let width = x_right - x_left;
    match boundary.wave {
        Wave::Constant => (initial_area, 0.0),
        Wave::Shock { speed } => {
            let initial_length = (x_right.min(speed * time) - x_left).clamp(0.0, width);
            let boundary_length = width - initial_length;
            (
                (initial_length * initial_area + boundary_length * boundary.area) / width,
                boundary_length * boundary.outward_discharge / width,
            )
        }
        Wave::Rarefaction { head, tail } => {
            let head_position = head * time;
            let tail_position = tail * time;
            let initial_length = (x_right.min(head_position) - x_left).clamp(0.0, width);
            let boundary_length = (x_right - x_left.max(tail_position)).clamp(0.0, width);
            let mut integral = (
                initial_length * initial_area + boundary_length * boundary.area,
                boundary_length * boundary.outward_discharge,
            );
            let a = x_left.max(head_position);
            let b = x_right.min(tail_position);
            if b > a {
                let r = geometry.invariant_factor();
                let fan = |x: f64| {
                    let sound = (r - x / time) / (r + 1.0);
                    let depth = sound * sound;
                    let area = geometry.area(depth);
                    (area, area * r * (1.0 - sound))
                };
                let half_width = 0.5 * (b - a);
                let middle = a + half_width;
                let offset = half_width * (3.0_f64 / 5.0).sqrt();
                let left = fan(middle - offset);
                let centre = fan(middle);
                let right = fan(middle + offset);
                // Area and discharge are polynomials of degree at most four and five.
                integral.0 +=
                    half_width * ((5.0 / 9.0) * (left.0 + right.0) + (8.0 / 9.0) * centre.0);
                integral.1 +=
                    half_width * ((5.0 / 9.0) * (left.1 + right.1) + (8.0 / 9.0) * centre.1);
            }
            (integral.0 / width, integral.1 / width)
        }
    }
}

/// Returns reflected averages for a left-end fixture on [0, 20].
pub fn left_cell_average(
    geometry: Geometry,
    boundary_depth: f64,
    x_left: f64,
    x_right: f64,
    time: f64,
) -> (f64, f64) {
    let (area, discharge) = right_cell_average(geometry, boundary_depth, -x_right, -x_left, time);
    (area, -discharge)
}

/// Entries contain (length, numerical area, numerical discharge, exact area, exact discharge).
pub fn mean_errors(
    geometry: Geometry,
    cells: impl IntoIterator<Item = (f64, f64, f64, f64, f64)>,
) -> Option<(f64, f64)> {
    let mut length = 0.0;
    let mut depth_error = 0.0;
    let mut discharge_error = 0.0;
    for (dx, area, discharge, exact_area, exact_discharge) in cells {
        length += dx;
        depth_error +=
            dx * (geometry.represented_depth(area) - geometry.represented_depth(exact_area)).abs();
        discharge_error += dx * (discharge - exact_discharge).abs();
    }
    (length > 0.0).then(|| (depth_error / length, discharge_error / length))
}

/// Returns the finest-grid physical signal against unchanged uniform initial water.
pub fn uniform_initial_signal(
    geometry: Geometry,
    boundary_depth: f64,
    count: usize,
    time: f64,
) -> Option<(f64, f64)> {
    if count == 0 {
        return None;
    }
    let dx = 20.0 / count as f64;
    mean_errors(
        geometry,
        (0..count).map(|index| {
            let left = -20.0 + index as f64 * dx;
            let (exact_area, exact_discharge) =
                right_cell_average(geometry, boundary_depth, left, left + dx, time);
            (dx, geometry.area(1.0), 0.0, exact_area, exact_discharge)
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, dimensional_scale: f64) {
        let tolerance = 256.0 * f64::EPSILON * dimensional_scale;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
        );
    }

    #[test]
    fn rarefaction_traces_match_independent_decimal_literals() {
        for (geometry, velocity, discharge, tail) in [
            (Geometry::Rectangular, 0.2, 0.486, -0.7),
            (Geometry::Triangular, 0.4, 0.26244, -0.5),
        ] {
            let boundary = trace(geometry, 0.81);
            close(boundary.outward_velocity, velocity, 1.0);
            close(boundary.outward_discharge, discharge, 1.0);
            match boundary.wave {
                Wave::Rarefaction {
                    head,
                    tail: actual_tail,
                } => {
                    assert_eq!(head, -1.0);
                    close(actual_tail, tail, 1.0);
                }
                _ => unreachable!("The literal fixture is a rarefaction"),
            }
        }
    }

    #[test]
    fn shock_trace_satisfies_both_jump_balances_and_strict_lax_order() {
        for geometry in [Geometry::Rectangular, Geometry::Triangular] {
            let boundary = trace(geometry, 1.21);
            let Wave::Shock { speed } = boundary.wave else {
                unreachable!("The deeper boundary fixture is a shock");
            };
            close(
                speed * (boundary.area - geometry.area(1.0)),
                boundary.outward_discharge,
                3.0,
            );
            close(
                speed * boundary.outward_discharge,
                boundary.momentum_flux - geometry.pressure(1.0),
                10.0,
            );
            assert!(boundary.outward_velocity - boundary.depth.sqrt() < speed);
            assert!(speed < -1.0);
            assert!(boundary.outward_velocity.abs() < boundary.depth.sqrt());
        }
    }

    #[test]
    fn whole_domain_integrals_match_independent_boundary_balances() {
        for geometry in [Geometry::Rectangular, Geometry::Triangular] {
            for depth in [0.81, 1.21] {
                let boundary = trace(geometry, depth);
                let average = right_cell_average(geometry, depth, -20.0, 0.0, 2.0);
                close(
                    average.0 * 20.0,
                    geometry.area(1.0) * 20.0 - boundary.outward_discharge * 2.0,
                    60.0,
                );
                close(
                    average.1 * 20.0,
                    (geometry.pressure(1.0) - boundary.momentum_flux) * 2.0,
                    10.0,
                );
            }
        }
    }

    #[test]
    fn split_cells_and_reflection_preserve_conserved_integrals() {
        for geometry in [Geometry::Rectangular, Geometry::Triangular] {
            for depth in [0.81, 1.21] {
                let whole = right_cell_average(geometry, depth, -3.0, 0.0, 2.0);
                let left = right_cell_average(geometry, depth, -3.0, -1.3, 2.0);
                let right = right_cell_average(geometry, depth, -1.3, 0.0, 2.0);
                close(whole.0 * 3.0, left.0 * 1.7 + right.0 * 1.3, 10.0);
                close(whole.1 * 3.0, left.1 * 1.7 + right.1 * 1.3, 3.0);
                let reflected = left_cell_average(geometry, depth, 0.0, 3.0, 2.0);
                assert_eq!(reflected.0, whole.0);
                assert_eq!(reflected.1, -whole.1);
            }
        }
    }

    #[test]
    fn independent_shock_literals_include_exact_geometry_caps() {
        // Rectangle: g=1, width=1, initial depth=1, boundary depth=cap=2.
        let rectangle = shock_from_conserved_jump(1.0, 0.5, 2.0, 2.0);
        close(rectangle.0, -3.0_f64.sqrt() / 2.0, 1.0);
        close(rectangle.1, -3.0_f64.sqrt(), 2.0);
        close(rectangle.2, 7.0 / 2.0, 4.0);
        assert!(rectangle.0 - 2.0_f64.sqrt() < rectangle.3 && rectangle.3 < -1.0);
        assert!(rectangle.0.abs() < 2.0_f64.sqrt());
        // Triangle: g=8, k=1, initial depth=1, boundary depth=cap=1.5.
        let triangle = shock_from_conserved_jump(1.0, 8.0 / 3.0, 9.0 / 4.0, 9.0);
        let expected_velocity = -(95.0_f64 / 27.0).sqrt();
        close(triangle.0, expected_velocity, 2.0);
        close(triangle.1, (9.0 / 4.0) * expected_velocity, 5.0);
        close(triangle.2, 203.0 / 12.0, 20.0);
        assert!(triangle.0 - 6.0_f64.sqrt() < triangle.3 && triangle.3 < -2.0);
        assert!(triangle.0.abs() < 6.0_f64.sqrt());
    }

    #[test]
    fn critical_trace_has_equal_outward_velocity_and_sound_speed() {
        for geometry in [Geometry::Rectangular, Geometry::Triangular] {
            let boundary = trace(geometry, geometry.critical_depth());
            close(boundary.outward_velocity, boundary.depth.sqrt(), 1.0);
            close(
                boundary.outward_discharge,
                geometry.critical_outward_discharge(),
                1.0,
            );
        }
    }
}
