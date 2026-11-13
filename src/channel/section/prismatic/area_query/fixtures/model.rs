#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AnchorKind {
    Source,
    Zero,
    BandStart,
    BandEnd,
    Cap,
}
#[derive(Clone, Copy)]
pub(crate) struct Profile {
    pub id: &'static str,
    pub coordinates: &'static [[u64; 2]],
    pub cap_stage: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct Extrema {
    pub minimum: [u64; 2],
    pub maximum: [u64; 2],
    pub allowances: [u64; 2],
    pub conditioned_scale: Option<u64>,
    pub zero: bool,
    pub positive: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct Chart {
    pub owner: usize,
    pub kind: AnchorKind,
    pub band: i64,
    pub source: [u64; 3],
    pub add: bool,
    pub scale: [u64; 3],
    pub parameter: [u64; 2],
}
#[derive(Clone, Copy)]
pub(crate) struct BandCase {
    pub index: usize,
    pub local: [Extrema; 4],
}
#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub id: &'static str,
    pub group: usize,
    pub chart: Chart,
    pub fields: [Extrema; 5],
    pub bands: &'static [BandCase],
    pub wet: bool,
    pub dry: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct InvalidCase {
    pub id: &'static str,
    pub cause: &'static str,
    pub chart: Chart,
}
