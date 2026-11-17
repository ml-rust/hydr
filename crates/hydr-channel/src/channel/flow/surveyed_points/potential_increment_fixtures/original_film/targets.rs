use super::super::model::{Reference, ReferencePolicy};
use super::model::Target;

pub(crate) const STAR: Target = Target {
    id: "film_star",
    area_rational: "1/23384026197294446691258957323460528314494920687616",
    area_bounds: [0x35b0000000000000, 0x35b0000000000000],
    declared_point_input: Some([0x35b0000000000000, 0x0000000000000000]),
    potential: Reference {
        bounds: [0x3ae0000000000000, 0x3ae0000000000000],
        native_allowance: 0x38cb7cdfd9d7bdba,
        policy: ReferencePolicy::PhysicalLocal,
    },
    sound: Reference {
        bounds: [0x3ad0000000000000, 0x3ad0000000000000],
        native_allowance: 0x38bb7cdfd9d7bdba,
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
        bounds: [0x0000000000000000, 0x0000000000000000],
        native_allowance: 0x38db7cdfd9d7bdba,
        policy: ReferencePolicy::FilmResidual,
    },
    physical_residual_zero: true,
};

pub(crate) const SONIC_PHYSICAL: Target = Target {
    id: "film_right_sonic_physical",
    area_rational: "1/13153514735978126263833163494446547176903392886784",
    area_bounds: [0x35bc71c71c71c71c, 0x35bc71c71c71c71d],
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
        bounds: [0x8000000000000001, 0x0000000000000001],
        native_allowance: 0x38db7cdfd9d7bdba,
        policy: ReferencePolicy::FilmResidual,
    },
    physical_residual_zero: true,
};
