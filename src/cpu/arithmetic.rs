//! Shared configured CPU F64 environment probes without host constant-folded arithmetic.

use super::CpuTensor;
use numr::ops::{BinaryOps, ScalarOps};
use numr::runtime::cpu::{CpuClient, CpuDevice};

pub(crate) struct F64EnvironmentError {
    even: f64,
    odd: f64,
    subnormal_input: f64,
    subnormal_output: f64,
}

impl F64EnvironmentError {
    pub(crate) fn message(&self, component: &str) -> String {
        let Self {
            even,
            odd,
            subnormal_input,
            subnormal_output,
        } = self;
        format!(
            "unsupported CPU arithmetic: even_tie={even:?}, odd_tie={odd:?}, subnormal_input={subnormal_input:?}, subnormal_output={subnormal_output:?}. Select round-to-nearest ties-to-even and disable FTZ/DAZ before creating the {component} context"
        )
    }
}

pub(crate) fn probe_f64(
    client: &CpuClient,
    device: &CpuDevice,
) -> numr::error::Result<Option<F64EnvironmentError>> {
    let tensor = |values: &[f64]| CpuTensor::from_slice(values, &[values.len()], device);
    let half_ulp = tensor(&[f64::EPSILON / 2.0])?;
    let even = client.add(&tensor(&[1.0])?, &half_ulp)?.item::<f64>()?;
    let odd = client
        .add(&tensor(&[f64::from_bits(1.0f64.to_bits() + 1)])?, &half_ulp)?
        .item::<f64>()?;
    let quantum = tensor(&[f64::from_bits(1)])?;
    let subnormal_input = client.add(&quantum, &quantum)?.item::<f64>()?;
    let subnormal_output = client
        .mul_scalar(&tensor(&[f64::MIN_POSITIVE])?, 0.5)?
        .item::<f64>()?;
    if even.to_bits() == 1.0f64.to_bits()
        && odd.to_bits() == 1.0f64.to_bits() + 2
        && subnormal_input.to_bits() == 2
        && subnormal_output.to_bits() == (1u64 << 51)
    {
        Ok(None)
    } else {
        Ok(Some(F64EnvironmentError {
            even,
            odd,
            subnormal_input,
            subnormal_output,
        }))
    }
}
