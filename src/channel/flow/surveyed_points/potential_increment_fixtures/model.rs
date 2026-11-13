pub(crate) type Bounds = [u64; 2];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReferencePolicy {
    ExactZero,
    PhysicalLocal,
    FilmResidual,
}
#[derive(Clone, Copy)]
pub(crate) struct Reference {
    pub bounds: Bounds,
    pub native_allowance: u64,
    pub policy: ReferencePolicy,
}
#[derive(Clone, Copy)]
pub(crate) struct Profile {
    pub id: &'static str,
    pub coordinates: &'static [[u64; 2]],
    pub cap_stage: u64,
}
