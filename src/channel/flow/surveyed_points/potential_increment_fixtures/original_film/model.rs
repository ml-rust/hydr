use super::super::model::{Bounds, Reference};
#[derive(Clone, Copy)]
pub(crate) struct Sources {
    pub case_id: &'static str,
    pub gravity_bits: u64,
    pub areas: [[u64; 2]; 2],
    pub discharges: [[u64; 2]; 2],
    pub velocities: [[u64; 2]; 2],
    pub velocity_jump: [u64; 2],
    pub epsilon: [u64; 2],
    pub potential: [Reference; 2],
    pub sound: [Reference; 2],
    pub exact_potential_identity_bits: u64,
    pub potential_identity_requires_zero_radius: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct Target {
    pub id: &'static str,
    pub area_rational: &'static str,
    pub area_bounds: Bounds,
    pub declared_point_input: Option<[u64; 2]>,
    pub potential: Reference,
    pub sound: Reference,
    pub left_increment: Reference,
    pub right_increment: Reference,
    pub residual: Reference,
    pub physical_residual_zero: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct SonicBracket {
    pub zero_anchor: bool,
    pub add: bool,
    pub scale_components: [u64; 2],
    pub parameter_bounds: Bounds,
    pub exact_physical_parameter_rational: &'static str,
    pub endpoints: [Target; 2],
    pub strict_opposite_residual_signs: bool,
}
