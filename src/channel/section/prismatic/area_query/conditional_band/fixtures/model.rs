pub(crate) type Bounds = [u64; 2];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RangePolicy {
    ExactZero,
    ConditionedLocal,
    OriginalGeometry,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct RangeRef {
    pub minimum: Bounds,
    pub maximum: Bounds,
    pub native_allowances: Bounds,
    pub policy: RangePolicy,
    pub positive: bool,
}
#[derive(Debug)]
pub(crate) struct Sample {
    pub fields: [RangeRef; 10],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GapAuthority {
    ConditionedLocal,
    OriginalGeometry,
}
#[derive(Debug)]
pub(crate) struct GapRef {
    pub bounds: Bounds,
    pub native_allowance: u64,
    pub exact_zero: bool,
    pub authority: GapAuthority,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PanelKind {
    OrdinaryWet,
    DryFirstBand,
    Zero,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClipAuthority {
    BandStart,
    BandEnd,
    Interior,
}
#[derive(Debug)]
pub(crate) struct Panel {
    pub band_index: usize,
    pub kind: PanelKind,
    pub gap: GapRef,
    pub source_clip: ClipAuthority,
    pub target_clip: ClipAuthority,
    pub shared_boundary_identity: bool,
    pub original_source_equals_target: bool,
    pub ordinary_active: bool,
    pub points: Vec<Sample>,
    pub intervals: Vec<Sample>,
    pub endpoint_geometry: Vec<Sample>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PhysicalEndpoint {
    Lower,
    Upper,
}
#[derive(Debug)]
pub(crate) struct Path {
    pub id: String,
    pub binding_id: String,
    pub binding_index: usize,
    pub query_index: usize,
    pub original_parameter_endpoint: usize,
    pub physical_endpoint: PhysicalEndpoint,
    pub panels: Vec<Panel>,
    pub padded_bands: Vec<usize>,
}
#[derive(Debug)]
pub(crate) struct FixtureSet {
    pub paths: Vec<Path>,
}
