pub type Bounds = [u64; 2];
#[derive(Clone, Copy)]
pub struct ContactCase {
    pub id: &'static str,
    pub distance: u64,
    pub bank_length: u64,
    pub thickness: u64,
    pub expected: Bounds,
    pub allowance: u64,
}
#[derive(Clone, Copy)]
pub struct BinaryRemainingCase {
    pub id: &'static str,
    pub profile_id: &'static str,
    pub coordinates: &'static [[u64; 2]],
    pub cap_stage: u64,
    pub input_high: u64,
    pub input_low: u64,
    pub band_index: usize,
    pub expected: Bounds,
    pub allowance: u64,
}
pub const PRIVATE_CONTACT_CASES: &[ContactCase] = &[
    ContactCase {
        id: "private_contact_lower",
        distance: 0x1a70000000000000,
        bank_length: 0x6570000000000000,
        thickness: 0x6570000000000000,
        expected: [0x1a70000000000000, 0x1a70000000000000],
        allowance: 0x185b7cdfd9d7bdba,
    },
    ContactCase {
        id: "private_contact_upper_remaining",
        distance: 0x1a70000000000000,
        bank_length: 0x6570000000000000,
        thickness: 0x6570000000000000,
        expected: [0x1a70000000000000, 0x1a70000000000000],
        allowance: 0x185b7cdfd9d7bdba,
    },
];
pub const BINARY_REMAINING_CASE: BinaryRemainingCase = BinaryRemainingCase {
    id: "three_band_cap_retained_deficit",
    profile_id: "three_band",
    coordinates: &[
        [0xc010000000000000, 0x4008000000000000],
        [0xc004000000000000, 0x4000000000000000],
        [0xc000000000000000, 0x3ff0000000000000],
        [0xbff0000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000],
        [0x4000000000000000, 0x3ff0000000000000],
        [0x4004000000000000, 0x4000000000000000],
        [0x4010000000000000, 0x4008000000000000],
    ],
    cap_stage: 0x4008000000000000,
    input_high: 0x402c000000000000,
    input_low: 0xbaf0000000000000,
    band_index: 2,
    expected: [0x3adcd82b446159f3, 0x3adcd82b446159f4],
    allowance: 0x38c8c6f3f024fe77,
};
