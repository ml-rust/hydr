pub(in crate::channel::flow::surveyed_points) type Bounds = [u64; 2];

#[derive(Clone, Copy, Debug)]
pub(in crate::channel::flow::surveyed_points) struct Quantization {
    pub(in crate::channel::flow::surveyed_points) quantity: &'static str,
    pub(in crate::channel::flow::surveyed_points) exact_rational: &'static str,
    pub(in crate::channel::flow::surveyed_points) projected: Bounds,
    pub(in crate::channel::flow::surveyed_points) maximum_projected_width_rational: &'static str,
    pub(in crate::channel::flow::surveyed_points) positive_water_not_dry: bool,
    pub(in crate::channel::flow::surveyed_points) declaration_identity: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::channel::flow::surveyed_points) struct FluxReference {
    pub(in crate::channel::flow::surveyed_points) id: &'static str,
    pub(in crate::channel::flow::surveyed_points) config_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) expected_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) source_group: &'static str,
    pub(in crate::channel::flow::surveyed_points) physical_group: &'static str,
    pub(in crate::channel::flow::surveyed_points) outcome: &'static str,
    pub(in crate::channel::flow::surveyed_points) truth_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) flux_gate_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) water_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) momentum_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) water: Option<Bounds>,
    pub(in crate::channel::flow::surveyed_points) momentum: Option<Bounds>,
    pub(in crate::channel::flow::surveyed_points) positive_water_required: bool,
    pub(in crate::channel::flow::surveyed_points) quantization: &'static [Quantization],
}
