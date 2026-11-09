//! Internal face-oriented shallow-water numerics.

mod boundary;
#[cfg(test)]
mod flux;
pub(crate) mod soil;
mod tensor_flux;
mod tensor_types;
mod tensor_wave;

pub(crate) use boundary::{
    TensorBoundaryRequest, TensorBoundaryTrace, characteristic_trace, check_transmissive,
};
pub(crate) use tensor_flux::hydrostatic_hll_tensor;
pub(crate) use tensor_types::TensorFaceState;
