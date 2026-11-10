//! Selected-runtime F64 admission and environment probes.

use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps};
use numr::runtime::{FloatArithmeticContract, Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) enum F64EnvironmentError {
    Runtime {
        name: &'static str,
        computes_f64: bool,
        contract: FloatArithmeticContract,
    },
    Mode {
        name: &'static str,
        even: f64,
        odd: f64,
        subnormal_input: f64,
        subnormal_output: f64,
    },
}

impl F64EnvironmentError {
    pub(crate) fn message(&self, component: &str) -> String {
        match self {
            Self::Runtime {
                name,
                computes_f64,
                contract,
            } => format!(
                "unsupported {name} arithmetic: computes_f64={computes_f64}, contract={contract:?}. Select a runtime providing F64 IEEE 754 basic arithmetic before creating the {component} context"
            ),
            Self::Mode {
                name,
                even,
                odd,
                subnormal_input,
                subnormal_output,
            } => {
                let name = name.to_ascii_uppercase();
                format!(
                    "unsupported {name} arithmetic: even_tie={even:?}, odd_tie={odd:?}, subnormal_input={subnormal_input:?}, subnormal_output={subnormal_output:?}. Select round-to-nearest ties-to-even and disable FTZ/DAZ before creating the {component} context"
                )
            }
        }
    }
}

pub(crate) fn probe_f64<R, C>(client: &C) -> numr::error::Result<Option<F64EnvironmentError>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + ScalarOps<R>,
{
    let computes_f64 = R::computes_float_dtype(DType::F64);
    let contract = R::float_arithmetic_contract(DType::F64);
    if !computes_f64 || contract != FloatArithmeticContract::Ieee754Basic {
        return Ok(Some(F64EnvironmentError::Runtime {
            name: R::name(),
            computes_f64,
            contract,
        }));
    }
    let device = client.device();
    let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], device);
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
        Ok(Some(F64EnvironmentError::Mode {
            name: R::name(),
            even,
            odd,
            subnormal_input,
            subnormal_output,
        }))
    }
}
