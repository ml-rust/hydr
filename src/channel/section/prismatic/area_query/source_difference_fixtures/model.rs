pub(crate) type Bounds = [u64; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeltaAreaAuthority {
    ExactZero,
    ConditionedLocal,
    OriginalGeometry,
}

#[derive(Clone, Copy)]
pub(crate) struct Endpoint {
    pub delta_area: Bounds,
    pub delta_area_allowance: u64,
    pub delta_area_authority: DeltaAreaAuthority,
    pub structural_equality: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub id: &'static str,
    pub query_id: &'static str,
    pub query_index: usize,
    pub source_components: [u64; 2],
    pub endpoints: [Endpoint; 2],
    pub minimum_endpoint: usize,
    pub maximum_endpoint: usize,
}
