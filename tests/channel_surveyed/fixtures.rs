#[derive(Clone, Copy)]
pub struct Front {
    pub direction: i32,
    pub position: [u64; 2],
}

#[derive(Clone, Copy)]
pub struct Case {
    pub id: &'static str,
    pub classification: &'static str,
    pub points: &'static [[u64; 2]],
    pub cap: u64,
    pub gravity: u64,
    pub initial: [[u64; 2]; 2],
    pub exterior: [[u64; 2]; 2],
    // scales[1] is A_max * max(|Q / A| + sqrt(g h(A))) over the two initial states.
    // A dry state adds 0. Gravity speed alone understates a supercritical release.
    pub scales: [u64; 2],
    pub grids: [&'static [[u64; 4]]; 3],
    pub fronts: &'static [Front],
}
#[path = "fixtures/both_knots_dry_fan_100.rs"]
mod both_knots_dry_fan_100;
#[path = "fixtures/both_knots_dry_fan_200.rs"]
mod both_knots_dry_fan_200;
#[path = "fixtures/both_knots_dry_fan_400.rs"]
mod both_knots_dry_fan_400;
#[path = "fixtures/both_knots_dry_fan_reverse_100.rs"]
mod both_knots_dry_fan_reverse_100;
#[path = "fixtures/both_knots_dry_fan_reverse_200.rs"]
mod both_knots_dry_fan_reverse_200;
#[path = "fixtures/both_knots_dry_fan_reverse_400.rs"]
mod both_knots_dry_fan_reverse_400;
#[path = "fixtures/generated_vacuum_100.rs"]
mod generated_vacuum_100;
#[path = "fixtures/generated_vacuum_200.rs"]
mod generated_vacuum_200;
#[path = "fixtures/generated_vacuum_400.rs"]
mod generated_vacuum_400;
#[path = "fixtures/legacy_multiband_100.rs"]
mod legacy_multiband_100;
#[path = "fixtures/legacy_multiband_200.rs"]
mod legacy_multiband_200;
#[path = "fixtures/legacy_multiband_400.rs"]
mod legacy_multiband_400;
#[path = "fixtures/original_rounded_wet_fan_100.rs"]
mod original_rounded_wet_fan_100;
#[path = "fixtures/original_rounded_wet_fan_200.rs"]
mod original_rounded_wet_fan_200;
#[path = "fixtures/original_rounded_wet_fan_400.rs"]
mod original_rounded_wet_fan_400;
#[path = "fixtures/trapezoid_wet_100.rs"]
mod trapezoid_wet_100;
#[path = "fixtures/trapezoid_wet_200.rs"]
mod trapezoid_wet_200;
#[path = "fixtures/trapezoid_wet_400.rs"]
mod trapezoid_wet_400;
#[path = "fixtures/trapezoid_wet_reverse_100.rs"]
mod trapezoid_wet_reverse_100;
#[path = "fixtures/trapezoid_wet_reverse_200.rs"]
mod trapezoid_wet_reverse_200;
#[path = "fixtures/trapezoid_wet_reverse_400.rs"]
mod trapezoid_wet_reverse_400;
pub const CASES: &[Case] = &[
    Case {
        id: "trapezoid_wet",
        classification: "wet_wet",
        points: &[
            [0xc010000000000000, 0x4010000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000],
            [0x4024000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
        gravity: 0x40239eb851eb851f,
        initial: [
            [0x4024000000000000, 0x0000000000000000],
            [0x3ff6000000000000, 0x0000000000000000],
        ],
        exterior: [
            [0x4000000000000000, 0x0000000000000000],
            [0x3fe0000000000000, 0x0000000000000000],
        ],
        scales: [0x4024000000000000, 0x404625b12a8664c6],
        grids: [
            trapezoid_wet_100::CELLS,
            trapezoid_wet_200::CELLS,
            trapezoid_wet_400::CELLS,
        ],
        fronts: &[],
    },
    Case {
        id: "trapezoid_wet_reverse",
        classification: "wet_wet",
        points: &[
            [0xc010000000000000, 0x4010000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000],
            [0x4024000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
        gravity: 0x40239eb851eb851f,
        initial: [
            [0x3ff6000000000000, 0x0000000000000000],
            [0x4024000000000000, 0x0000000000000000],
        ],
        exterior: [
            [0x3fe0000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000],
        ],
        scales: [0x4024000000000000, 0x404625b12a8664c6],
        grids: [
            trapezoid_wet_reverse_100::CELLS,
            trapezoid_wet_reverse_200::CELLS,
            trapezoid_wet_reverse_400::CELLS,
        ],
        fronts: &[],
    },
    Case {
        id: "legacy_multiband",
        classification: "wet_wet",
        points: &[
            [0x0000000000000000, 0x4010000000000000],
            [0x3ff0000000000000, 0x4000000000000000],
            [0x4008000000000000, 0x3ff0000000000000],
            [0x4010000000000000, 0x0000000000000000],
            [0x4014000000000000, 0x0000000000000000],
            [0x4018000000000000, 0x3ff0000000000000],
            [0x4020000000000000, 0x4000000000000000],
            [0x4022000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
        gravity: 0x40239eb851eb851f,
        initial: [
            [0x4028000000000000, 0x0000000000000000],
            [0x3fd0000000000000, 0x0000000000000000],
        ],
        exterior: [
            [0x400572fc8ab73db7, 0x0000000000000000],
            [0x3fca827999fcef32, 0x0000000000000000],
        ],
        scales: [0x4028000000000000, 0x404ec5747acf848e],
        grids: [
            legacy_multiband_100::CELLS,
            legacy_multiband_200::CELLS,
            legacy_multiband_400::CELLS,
        ],
        fronts: &[],
    },
    Case {
        id: "original_rounded_wet_fan",
        classification: "wet_wet",
        points: &[
            [0xc010000000000000, 0x4008000000000000],
            [0xc004000000000000, 0x4000000000000000],
            [0xc000000000000000, 0x3ff0000000000000],
            [0xbff0000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x3ff0000000000000],
            [0x4004000000000000, 0x4000000000000000],
            [0x4010000000000000, 0x4008000000000000],
        ],
        cap: 0x4008000000000000,
        gravity: 0x3ff0000000000000,
        initial: [
            [0x402c000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x400451bec8c475f1],
        ],
        exterior: [
            [0x4008000000000000, 0x0000000000000000],
            [0x3fda827999fcef32, 0x400451bec8c475f1],
        ],
        scales: [0x402c000000000000, 0x404648d91640c3ab],
        grids: [
            original_rounded_wet_fan_100::CELLS,
            original_rounded_wet_fan_200::CELLS,
            original_rounded_wet_fan_400::CELLS,
        ],
        fronts: &[],
    },
    Case {
        id: "both_knots_dry_fan",
        classification: "right_dry",
        points: &[
            [0x0000000000000000, 0x4010000000000000],
            [0x3ff0000000000000, 0x4000000000000000],
            [0x4008000000000000, 0x3ff0000000000000],
            [0x4010000000000000, 0x0000000000000000],
            [0x4014000000000000, 0x0000000000000000],
            [0x4018000000000000, 0x3ff0000000000000],
            [0x4020000000000000, 0x4000000000000000],
            [0x4022000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
        gravity: 0x40239eb851eb851f,
        initial: [
            [0x4028000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000],
        ],
        exterior: [
            [0x400572fc8ab73db7, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000],
        ],
        scales: [0x4028000000000000, 0x404ec5747acf848e],
        grids: [
            both_knots_dry_fan_100::CELLS,
            both_knots_dry_fan_200::CELLS,
            both_knots_dry_fan_400::CELLS,
        ],
        fronts: &[Front {
            direction: -1,
            position: [0x40389031f248697f, 0x40389031f2486980],
        }],
    },
    Case {
        id: "both_knots_dry_fan_reverse",
        classification: "left_dry",
        points: &[
            [0x0000000000000000, 0x4010000000000000],
            [0x3ff0000000000000, 0x4000000000000000],
            [0x4008000000000000, 0x3ff0000000000000],
            [0x4010000000000000, 0x0000000000000000],
            [0x4014000000000000, 0x0000000000000000],
            [0x4018000000000000, 0x3ff0000000000000],
            [0x4020000000000000, 0x4000000000000000],
            [0x4022000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
        gravity: 0x40239eb851eb851f,
        initial: [
            [0x0000000000000000, 0x0000000000000000],
            [0x4028000000000000, 0x0000000000000000],
        ],
        exterior: [
            [0x0000000000000000, 0x0000000000000000],
            [0x400572fc8ab73db7, 0x0000000000000000],
        ],
        scales: [0x4028000000000000, 0x404ec5747acf848e],
        grids: [
            both_knots_dry_fan_reverse_100::CELLS,
            both_knots_dry_fan_reverse_200::CELLS,
            both_knots_dry_fan_reverse_400::CELLS,
        ],
        fronts: &[Front {
            direction: 1,
            position: [0xc0389031f2486980, 0xc0389031f248697f],
        }],
    },
    Case {
        id: "generated_vacuum",
        classification: "vacuum",
        points: &[
            [0x0000000000000000, 0x4010000000000000],
            [0x3ff0000000000000, 0x4000000000000000],
            [0x4008000000000000, 0x3ff0000000000000],
            [0x4010000000000000, 0x0000000000000000],
            [0x4014000000000000, 0x0000000000000000],
            [0x4018000000000000, 0x3ff0000000000000],
            [0x4020000000000000, 0x4000000000000000],
            [0x4022000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
        gravity: 0x40239eb851eb851f,
        initial: [
            [0x4028000000000000, 0xc078000000000000],
            [0x4028000000000000, 0x4078000000000000],
        ],
        exterior: [
            [0x400572fc8ab73db7, 0xc040000000000000],
            [0x400572fc8ab73db7, 0x4040000000000000],
        ],
        scales: [0x4028000000000000, 0x407bd8ae8f59f092],
        grids: [
            generated_vacuum_100::CELLS,
            generated_vacuum_200::CELLS,
            generated_vacuum_400::CELLS,
        ],
        fronts: &[
            Front {
                direction: -1,
                position: [0xc043b7e706dbcb41, 0xc043b7e706dbcb40],
            },
            Front {
                direction: 1,
                position: [0x4043b7e706dbcb40, 0x4043b7e706dbcb41],
            },
        ],
    },
];
