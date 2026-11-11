use super::{Candidate, Case};

pub(super) const CASES: &[Case] = &[
    Case {
        id: "planned_three_bands_both_knots_dry_fan",
        profile: 4,
        input: [
            0x402c000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        gravity: 0x3ff0000000000000,
        loose_recovery: false,
        rejection: None,
        allowed: &[],
        zero_work: false,
        flux: [
            [0x40194f3b8ee6c299, 0x40194f3b8ee6c29a],
            [0x40261ea4d72778db, 0x40261ea4d72778dc],
        ],
        candidates: &[Candidate {
            area: [0x4016ccf1ffa9e4d9, 0x4016ccf1ffa9e4da],
            discharge: [0x40194f3b8ee6c299, 0x40194f3b8ee6c29a],
            depth: [0x3ffa0446cb515322, 0x3ffa0446cb515323],
        }],
        edges: &[
            [0xbff52a7fa9d2f8ea, 0xbff52a7fa9d2f8e9],
            [0x400eed016f605088, 0x400eed016f605089],
        ],
        dry: false,
        wet: true,
    },
    Case {
        id: "planned_three_bands_both_knots_near_pure_fan",
        profile: 4,
        input: [
            0x402c000000000000,
            0x0000000000000000,
            0x3ff0000000000000,
            0x400451bec8c475f1,
        ],
        gravity: 0x3ff0000000000000,
        loose_recovery: false,
        rejection: None,
        allowed: &[],
        zero_work: false,
        flux: [
            [0x40194f3b8ee6c299, 0x40194f3b8ee6c29a],
            [0x40261ea4d72778db, 0x40261ea4d72778dc],
        ],
        candidates: &[Candidate {
            area: [0x4016ccf1ffa9e4d9, 0x4016ccf1ffa9e4da],
            discharge: [0x40194f3b8ee6c299, 0x40194f3b8ee6c29a],
            depth: [0x3ffa0446cb515322, 0x3ffa0446cb515323],
        }],
        edges: &[
            [0xbff52a7fa9d2f8ea, 0xbff52a7fa9d2f8e9],
            [0x3fff1ffe8c701057, 0x3fff1ffe8c701058],
            [0x4009137e4b50e3b6, 0x4009137e4b50e3b7],
            [0x4009137e4b50e3b6, 0x4009137e4b50e3b7],
        ],
        dry: false,
        wet: true,
    },
];
