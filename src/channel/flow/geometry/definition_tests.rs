use super::{ChannelGeometry, TriangularChannelCell};

fn geometry(left: f64, right: f64, bed: f64) -> ChannelGeometry {
    ChannelGeometry::PrismaticTriangular {
        left_bank_slope_h_per_v: left,
        right_bank_slope_h_per_v: right,
        bed_elevation_m: bed,
        cells: vec![TriangularChannelCell {
            id: "cell".into(),
            length_m: 2.0,
            maximum_depth_m: 3.0,
            manning_n: 0.03,
        }],
    }
}

#[test]
fn triangular_identity_retains_original_section_bits() {
    let section = geometry(1.0, 1.0, 0.0);
    assert!(section.same_bits(&section.clone()));
    assert!(!section.same_bits(&geometry(0.0, 2.0, 0.0)));
    assert!(!section.same_bits(&geometry(1.0, 1.0, -0.0)));
    assert!(!geometry(0.0, 2.0, 0.0).same_bits(&geometry(-0.0, 2.0, 0.0)));
    let mut changed = section.clone();
    if let ChannelGeometry::PrismaticTriangular { cells, .. } = &mut changed {
        cells[0].maximum_depth_m = f64::from_bits(3.0_f64.to_bits() + 1);
    }
    assert!(!section.same_bits(&changed));
}

#[test]
fn triangular_banks_require_a_finite_positive_section() {
    assert!(geometry(0.0, 2.0, 0.0).check().is_ok());
    for (left, right) in [
        (0.0, 0.0),
        (-1.0, 2.0),
        (f64::INFINITY, 1.0),
        (1.0, f64::NAN),
    ] {
        assert!(geometry(left, right, 0.0).check().is_err());
    }
}

#[test]
fn reach_identity_retains_flux_tolerance_bits() {
    use crate::channel::{ChannelConfig, ChannelReachDefinition};
    let original = ChannelReachDefinition {
        vertical_datum: "datum".into(),
        provenance: "flux tolerance identity".into(),
        geometry: geometry(1.0, 1.0, 0.0),
        lateral_bindings: vec![],
        config: ChannelConfig::default(),
    };
    for index in 0..3 {
        let mut changed = original.clone();
        let field = match index {
            0 => &mut changed.config.riemann_water_flux_absolute_m3_per_s,
            1 => &mut changed.config.riemann_momentum_flux_absolute_m4_per_s2,
            _ => &mut changed.config.riemann_flux_relative,
        };
        *field = f64::from_bits(field.to_bits() + 1);
        assert!(!original.same_bits(&changed));
    }
}
