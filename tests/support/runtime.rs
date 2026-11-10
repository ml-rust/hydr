//! Sequential CPU setup for tests and reference executables.

use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};

pub type Runtime = numr::runtime::cpu::CpuRuntime;

pub fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
}
