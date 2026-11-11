#[derive(Clone, Copy)]
pub(super) struct Profile {
    pub id: &'static str,
    pub points: &'static [[u64; 2]],
    pub cap: u64,
}

#[derive(Clone, Copy)]
pub(super) struct Candidate {
    pub area: [u64; 2],
    pub discharge: [u64; 2],
    pub depth: [u64; 2],
}

#[derive(Clone, Copy)]
pub(super) struct Case {
    pub id: &'static str,
    pub profile: usize,
    pub input: [u64; 4],
    pub gravity: u64,
    pub loose_recovery: bool,
    pub rejection: Option<hydr::channel::PrismaticRiemannRejection>,
    pub allowed: &'static [hydr::channel::PrismaticRiemannRejection],
    pub zero_work: bool,
    pub flux: [[u64; 2]; 2],
    pub candidates: &'static [Candidate],
    pub edges: &'static [[u64; 2]],
    pub dry: bool,
    pub wet: bool,
}

#[path = "data_planned_three_bands.rs"]
mod data_planned_three_bands;
#[path = "data_rectangle.rs"]
mod data_rectangle;
#[path = "data_three_bands.rs"]
mod data_three_bands;
#[path = "data_trapezoid.rs"]
mod data_trapezoid;
#[path = "data_triangle.rs"]
mod data_triangle;

pub(super) const PROFILES: &[Profile] = &[
    Profile {
        id: "rectangle",
        points: &[
            [0x0000000000000000, 0x4010000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
    },
    Profile {
        id: "triangle",
        points: &[
            [0x0000000000000000, 0x4010000000000000],
            [0x4010000000000000, 0x0000000000000000],
            [0x4020000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
    },
    Profile {
        id: "trapezoid",
        points: &[
            [0x0000000000000000, 0x4010000000000000],
            [0x4010000000000000, 0x0000000000000000],
            [0x4014000000000000, 0x0000000000000000],
            [0x4022000000000000, 0x4010000000000000],
        ],
        cap: 0x4010000000000000,
    },
    Profile {
        id: "three_bands",
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
    },
    Profile {
        id: "planned_three_bands",
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
    },
];

pub(super) const CASE_GROUPS: &[&[Case]] = &[
    data_rectangle::CASES,
    data_triangle::CASES,
    data_trapezoid::CASES,
    data_three_bands::CASES,
    data_planned_three_bands::CASES,
];
