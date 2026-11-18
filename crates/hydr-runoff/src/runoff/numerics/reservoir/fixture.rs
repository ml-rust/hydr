use crate::runoff::{ReservoirElement, ReservoirError, ReservoirParameters};

pub(super) fn element(id: &str, n: f64, width: f64) -> Result<ReservoirElement, ReservoirError> {
    Ok(ReservoirElement {
        id: id.into(),
        parameters: ReservoirParameters::from_native(1024.0, 1024.0, width, 1.0 / 64.0, n, 0.125)?,
    })
}

pub(super) fn cpu_client() -> numr::runtime::cpu::CpuClient {
    numr::runtime::cpu::CpuClient::new(numr::runtime::cpu::CpuDevice::new())
        .with_parallelism(numr::runtime::cpu::ParallelismConfig::new(Some(1), None))
}
