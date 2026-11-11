//! Internal face-oriented shallow-water numerics.

mod boundary;
#[cfg(test)]
mod flux;
pub(crate) mod reservoir;
pub(crate) mod soil;
pub(crate) mod storage;
pub(crate) mod structures;
pub(crate) mod subcatchment;
mod tensor_flux;
mod tensor_hll;
mod tensor_types;
mod tensor_wave;
pub(crate) mod water_transfers;

pub(crate) use boundary::{
    TensorBoundaryRequest, TensorBoundaryTrace, characteristic_trace, check_transmissive,
};
pub(crate) use tensor_flux::hydrostatic_hll_tensor;
pub(crate) use tensor_hll::{homogeneous_hll_tensor, homogeneous_wave_speed_tensor};
pub(crate) use tensor_types::TensorFaceState;
