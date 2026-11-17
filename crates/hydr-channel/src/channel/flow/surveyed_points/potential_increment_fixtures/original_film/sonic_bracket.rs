use super::super::model::{Reference, ReferencePolicy};
use super::model::{SonicBracket, Target};

pub(crate) const SONIC_BRACKET: SonicBracket = SonicBracket {
    zero_anchor: true,
    add: true,
    scale_components: [0x35f0000000000000, 0x0000000000000000],
    parameter_bounds: [0x3fbc71c71c71c71c, 0x3fbc71c71c71c71d],
    exact_physical_parameter_rational: "1/9",
    endpoints: [
        Target {
            id: "film_sonic_parameter_lower",
            area_rational: "2001599834386887/26328072917139296674479506920917608079723773850137277813577744384",
            area_bounds: [0x35bc71c71c71c71c, 0x35bc71c71c71c71c],
            declared_point_input: None,
            potential: Reference {
                bounds: [0x3ae5555555555555, 0x3ae5555555555556],
                native_allowance: 0x38d2533fe68fd3d1,
                policy: ReferencePolicy::PhysicalLocal,
            },
            sound: Reference {
                bounds: [0x3ad5555555555555, 0x3ad5555555555556],
                native_allowance: 0x38c2533fe68fd3d1,
                policy: ReferencePolicy::PhysicalLocal,
            },
            left_increment: Reference {
                bounds: [0xc000000000000000, 0xbfffffffffffffff],
                native_allowance: 0x3deb7cdfd9d7bdba,
                policy: ReferencePolicy::PhysicalLocal,
            },
            right_increment: Reference {
                bounds: [0xc000000000000000, 0xbfffffffffffffff],
                native_allowance: 0x3deb7cdfd9d7bdba,
                policy: ReferencePolicy::PhysicalLocal,
            },
            residual: Reference {
                bounds: [0xb780000000000001, 0xb780000000000000],
                native_allowance: 0x38db7cdfd9d7bdba,
                policy: ReferencePolicy::FilmResidual,
            },
            physical_residual_zero: false,
        },
        Target {
            id: "film_sonic_parameter_upper",
            area_rational: "8006399337547549/105312291668557186697918027683670432318895095400549111254310977536",
            area_bounds: [0x35bc71c71c71c71d, 0x35bc71c71c71c71d],
            declared_point_input: None,
            potential: Reference {
                bounds: [0x3ae5555555555555, 0x3ae5555555555556],
                native_allowance: 0x38d2533fe68fd3d1,
                policy: ReferencePolicy::PhysicalLocal,
            },
            sound: Reference {
                bounds: [0x3ad5555555555555, 0x3ad5555555555556],
                native_allowance: 0x38c2533fe68fd3d1,
                policy: ReferencePolicy::PhysicalLocal,
            },
            left_increment: Reference {
                bounds: [0xc000000000000000, 0xbfffffffffffffff],
                native_allowance: 0x3deb7cdfd9d7bdba,
                policy: ReferencePolicy::PhysicalLocal,
            },
            right_increment: Reference {
                bounds: [0xc000000000000000, 0xbfffffffffffffff],
                native_allowance: 0x3deb7cdfd9d7bdba,
                policy: ReferencePolicy::PhysicalLocal,
            },
            residual: Reference {
                bounds: [0x3783ffffffffffff, 0x3784000000000000],
                native_allowance: 0x38db7cdfd9d7bdba,
                policy: ReferencePolicy::FilmResidual,
            },
            physical_residual_zero: false,
        },
    ],
    strict_opposite_residual_signs: true,
};
