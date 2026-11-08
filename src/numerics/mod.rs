//! Internal face-oriented shallow-water numerics.

#[cfg(test)]
mod flux;
mod tensor_flux;
mod tensor_types;
mod tensor_wave;

pub(crate) use tensor_flux::hydrostatic_hll_tensor;
pub(crate) use tensor_types::TensorFaceState;
