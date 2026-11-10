//! Explicit independent Solvr observations, separate from accepted trajectory outputs.

use super::{reference_case::Result, reference_inputs::Case};
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;
use solvr::interpolate::{InterpolateError, LinearInterp1dDiagnostics};

pub struct Observation {
    pub status: &'static str,
    pub values: Vec<f64>,
    pub errors: Vec<f64>,
}

pub struct ReceiptObserver {
    client: CpuClient,
    device: CpuDevice,
}

impl ReceiptObserver {
    pub fn new() -> Self {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        Self { client, device }
    }

    pub fn observe(&self, case: &Case, time: f64, width: usize) -> Result<Observation> {
        let x = case.knots.iter().map(|knot| knot.0).collect::<Vec<_>>();
        let y = case.knots.iter().map(|knot| knot.1).collect::<Vec<_>>();
        let x = Tensor::<CpuRuntime>::from_slice(&x, &[x.len()], &self.device)?;
        let y = Tensor::<CpuRuntime>::from_slice(&y, &[y.len()], &self.device)?;
        let query = Tensor::<CpuRuntime>::from_slice(&vec![time; width], &[width], &self.device)?;
        match self.client.interp1d_linear_with_receipts(&x, &y, &query) {
            Ok(receipt) => Ok(Observation {
                status: "success",
                values: receipt.values().try_to_vec::<f64>()?,
                errors: receipt.absolute_error().try_to_vec::<f64>()?,
            }),
            Err(error) => Ok(Observation {
                status: match error {
                    InterpolateError::OutOfDomain { .. } => "OutOfDomain",
                    InterpolateError::NumericalError { .. } => "Numerical",
                    InterpolateError::UnsupportedArithmetic { .. } => "UnsupportedArithmetic",
                    _ => "Other",
                },
                values: Vec::new(),
                errors: Vec::new(),
            }),
        }
    }
}
