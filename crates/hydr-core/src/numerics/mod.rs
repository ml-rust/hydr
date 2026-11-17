//! Internal face-oriented shallow-water numerics.

#[cfg(test)]
mod flux;
pub mod outward;
pub mod rounding;
mod tensor_flux;
mod tensor_hll;
pub mod tensor_types;
pub mod tensor_wave;
pub mod water_transfers;

pub use tensor_flux::hydrostatic_hll_tensor;
pub use tensor_hll::{homogeneous_hll_tensor, homogeneous_wave_bounds_tensor};
pub use tensor_types::TensorFaceState;

pub use tensor_wave::{curve, dry_front_bounds};
