use super::super::model::Reference;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnchorKind {
    Source,
    Zero,
    BandStart,
    BandEnd,
    Cap,
}
#[derive(Clone, Copy)]
pub(crate) struct Chart {
    pub id: &'static str,
    pub profile_index: usize,
    pub anchor: AnchorKind,
    pub band: i64,
    pub anchor_source_components: [u64; 2],
    pub add: bool,
    pub scale_components: [u64; 2],
    pub parameter_bounds: [u64; 2],
}
#[derive(Clone, Copy)]
pub(crate) struct Endpoint {
    pub structural_equality: bool,
    pub potential: Reference,
    pub sound: Reference,
    pub increment: Reference,
}
#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub id: &'static str,
    pub query_id: &'static str,
    pub query_index: usize,
    pub source_components: [u64; 2],
    pub source_potential: Reference,
    pub source_sound: Reference,
    pub endpoints: [Endpoint; 2],
    pub minimum_endpoint: usize,
    pub maximum_endpoint: usize,
    pub wet_possible: bool,
    pub dry_possible: bool,
}
