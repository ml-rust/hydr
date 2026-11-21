//! The crown `y = D` and every depth above it are full. A circle is continuous there, a box jumps.

use super::reference_rows::BARRELS;
use super::support::{TestResult, barrel, batch, geometry};
use hydr::structures::{BarrelState, CulvertBarrelShape};

#[test]
fn crown_and_above_are_full_with_identical_constants() -> TestResult {
    for entry in BARRELS {
        let culvert = barrel(entry)?;
        let rise = culvert.rise_m();
        let below = rise.next_down();
        let barrels = batch(vec![culvert; 3])?;
        let rows = geometry(&barrels, &[rise, 1.5 * rise, below])?;
        let full = barrels.full_geometry()?.project()?;
        let constants = &full[0];
        assert_eq!(constants.state, BarrelState::Full);
        for row in &rows[..2] {
            assert_eq!(row.state, BarrelState::Full, "{}", entry.name);
            assert_eq!(row.area_m2, constants.area_m2);
            assert_eq!(row.top_width_m, (0.0, 0.0));
            assert_eq!(row.wetted_perimeter_m, constants.wetted_perimeter_m);
            assert_eq!(row.hydraulic_radius_m, constants.hydraulic_radius_m);
        }
        assert_eq!(
            rows[0].hydrostatic_moment_m3,
            constants.hydrostatic_moment_m3
        );
        assert!(rows[1].hydrostatic_moment_m3.0 > rows[0].hydrostatic_moment_m3.1);

        let free = &rows[2];
        let gap = rise - below;
        assert_eq!(free.state, BarrelState::Free, "{}", entry.name);
        match culvert.shape() {
            CulvertBarrelShape::Circular => {
                // A(D) - A(y) <= D (D - y), because T <= D.
                assert!(free.area_m2.0 <= constants.area_m2.1);
                assert!(free.area_m2.1 + 2.0 * rise * gap >= constants.area_m2.0);
                // P(D) - P(y) = D acos(1 - 2 gap/D) <= pi sqrt(D gap).
                let reach = 1.01 * std::f64::consts::PI * (rise * gap).sqrt();
                assert!(free.wetted_perimeter_m.0 <= constants.wetted_perimeter_m.1);
                assert!(free.wetted_perimeter_m.1 + reach >= constants.wetted_perimeter_m.0);
                assert!(free.top_width_m.1 <= 2.01 * (rise * gap).sqrt());
            }
            _ => {
                let span = culvert.span_m();
                assert_eq!(free.top_width_m, (span, span));
                // P(prev D) = B + 2 prev(D), then the full perimeter 2 (B + D) lies above it.
                let open = span + 2.0 * rise;
                assert!(free.wetted_perimeter_m.0 <= open.next_up());
                assert!(free.wetted_perimeter_m.1 >= (open - 4.0 * gap).next_down());
                assert!(free.wetted_perimeter_m.1 < constants.wetted_perimeter_m.0);
            }
        }
    }
    Ok(())
}
