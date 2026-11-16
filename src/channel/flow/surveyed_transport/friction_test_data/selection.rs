pub(crate) struct Selection {
    pub id: &'static str,
    pub radii_momentum: [u64; 4],
    pub bound: [u64; 2],
    pub expected: &'static str,
}
pub(crate) const SELECTION: &[Selection] = &[
    Selection {
        id: "exact_radius",
        radii_momentum: [
            0x3fe0000000000000,
            0x3fe0000000000000,
            0x3fe0000000000000,
            0x4010000000000000,
        ],
        bound: [0x0000000000000000, 0x0000000000000000],
        expected: "accept",
    },
    Selection {
        id: "narrow_radius",
        radii_momentum: [
            0x3fe0000000000000,
            0x3fe0000000002000,
            0x3fe0000000000000,
            0x4010000000000000,
        ],
        bound: [0x3d85555555555555, 0x3d85555555555556],
        expected: "accept",
    },
    Selection {
        id: "broad_radius",
        radii_momentum: [
            0x3fd0000000000000,
            0x3ff0000000000000,
            0x3fe0000000000000,
            0x4010000000000000,
        ],
        bound: [0x4005555555555555, 0x4005555555555556],
        expected: "reject_geometry_budget",
    },
    Selection {
        id: "broad_radius_negative",
        radii_momentum: [
            0x3fd0000000000000,
            0x3ff0000000000000,
            0x3fe0000000000000,
            0xc010000000000000,
        ],
        bound: [0x4005555555555555, 0x4005555555555556],
        expected: "reject_geometry_budget",
    },
    Selection {
        id: "tiny_broad_radius",
        radii_momentum: [
            0x3d70000000000000,
            0x3d80000000000000,
            0x3d70000000000000,
            0x4010000000000000,
        ],
        bound: [0x3ff5555555555555, 0x3ff5555555555556],
        expected: "reject_geometry_budget",
    },
    Selection {
        id: "selection_outside_radius",
        radii_momentum: [
            0x3fd0000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x4010000000000000,
        ],
        bound: [0x0000000000000000, 0x0000000000000000],
        expected: "reject_selection",
    },
];
