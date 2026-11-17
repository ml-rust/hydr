//! Internal face-oriented shallow-water numerics.

#[cfg(test)]
mod flux;
pub(crate) mod rounding;
mod tensor_flux;
mod tensor_hll;
pub(crate) mod tensor_types;
pub(crate) mod tensor_wave;
pub(crate) mod water_transfers;

pub(crate) use tensor_flux::hydrostatic_hll_tensor;
pub(crate) use tensor_hll::{homogeneous_hll_tensor, homogeneous_wave_bounds_tensor};
pub(crate) use tensor_types::TensorFaceState;

pub(crate) use tensor_wave::{curve, dry_front_bounds};
