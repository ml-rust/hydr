pub(crate) type Bounds = [u64; 2];

#[derive(Clone, Copy)]
pub(crate) struct Reference {
    pub bounds: Bounds,
    pub native_tolerance: u64,
    pub exact_zero: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct SideReference {
    pub phi: Reference,
    pub sound: Reference,
    pub dry: bool,
    pub physical_eligible: bool,
    pub physical_wet_eligible: bool,
    pub rectangular: bool,
    pub pointed: bool,
    pub within_first_band: bool,
    pub film_exact: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub id: &'static str,
    pub source_index: usize,
    pub gravity_bits: u64,
    pub sides: [SideReference; 2],
}
