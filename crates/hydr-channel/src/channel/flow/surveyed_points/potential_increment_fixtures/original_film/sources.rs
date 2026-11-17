use super::super::model::{Reference, ReferencePolicy};
use super::model::Sources;

pub(crate) const SOURCES: Sources = Sources {
    case_id: "retained_vacuum_inside_film",
    gravity_bits: 0x3ff0000000000000,
    areas: [
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000],
    ],
    discharges: [
        [0xc000000000000000, 0x0000000000000000],
        [0x4000000000000000, 0xbaf0000000000000],
    ],
    velocities: [
        [0xc000000000000000, 0x0000000000000000],
        [0x4000000000000000, 0xbaf0000000000000],
    ],
    velocity_jump: [0x4010000000000000, 0xbaf0000000000000],
    epsilon: [0x3af0000000000000, 0x0000000000000000],
    potential: [
        Reference {
            bounds: [0x4000000000000000, 0x4000000000000000],
            native_allowance: 0x3deb7cdfd9d7bdba,
            policy: ReferencePolicy::PhysicalLocal,
        },
        Reference {
            bounds: [0x4000000000000000, 0x4000000000000000],
            native_allowance: 0x3deb7cdfd9d7bdba,
            policy: ReferencePolicy::PhysicalLocal,
        },
    ],
    sound: [
        Reference {
            bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            native_allowance: 0x3ddb7cdfd9d7bdba,
            policy: ReferencePolicy::PhysicalLocal,
        },
        Reference {
            bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            native_allowance: 0x3ddb7cdfd9d7bdba,
            policy: ReferencePolicy::PhysicalLocal,
        },
    ],
    exact_potential_identity_bits: 0x4000000000000000,
    potential_identity_requires_zero_radius: true,
};
