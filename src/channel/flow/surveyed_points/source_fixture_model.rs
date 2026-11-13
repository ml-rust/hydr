pub(crate) type Bounds = [u64; 2];
#[derive(Clone, Copy)]
pub(crate) struct Profile {
    pub id: &'static str,
    pub coordinates: &'static [[u64; 2]],
    pub cap_stage: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct Reference {
    pub bounds: Bounds,
    pub center: [u64; 2],
    pub residual: u64,
    pub non_dyadic: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub id: &'static str,
    pub owner: usize,
    pub group: usize,
    pub inputs: [[u64; 3]; 4],
    pub dry: [bool; 2],
    pub uniform: bool,
    pub velocity: [Option<Reference>; 2],
    pub difference: Option<Reference>,
    pub difference_zero: bool,
    pub difference_allowance: Option<u64>,
}
