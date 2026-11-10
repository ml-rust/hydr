//! Trusted device-state adoption preserves the destination numerical context.

use super::{ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, SoilError};
use numr::dtype::DType;
use numr::runtime::{Device, Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> ModifiedGreenAmptBatch<R> {
    pub(crate) fn restore_trusted(&mut self, snapshot: &Self) -> Result<(), SoilError> {
        if !self
            .context
            .client
            .device()
            .is_same(snapshot.context.client.device())
        {
            return Err(SoilError::input(
                "trusted soil snapshot device differs. Restore onto the original runtime device",
            ));
        }
        if self.context.parameters.len() != snapshot.context.parameters.len()
            || !self
                .context
                .parameters
                .iter()
                .zip(&snapshot.context.parameters)
                .all(|(left, right)| parameter_bits(left) == parameter_bits(right))
        {
            return Err(SoilError::input(
                "trusted soil definitions differ. Restore into identical ordered parameters",
            ));
        }
        self.state = snapshot.state.clone();
        Ok(())
    }
}

fn parameter_bits(parameters: &ModifiedGreenAmptParameters) -> [u64; 6] {
    [
        parameters.suction.to_bits(),
        parameters.conductivity.to_bits(),
        parameters.deficit.to_bits(),
        parameters.lu.to_bits(),
        parameters.capacity.to_bits(),
        parameters.deficit_ceiling.to_bits(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soil::NativeSoilState;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use std::sync::Arc;

    #[test]
    fn trusted_soil_restore_preserves_destination_client() -> Result<(), SoilError> {
        let client = CpuClient::new(CpuDevice::new());
        let parameters = [ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?];
        let mut target = ModifiedGreenAmptBatch::<CpuRuntime>::initialized(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(64))),
            &parameters,
        )?;
        let state = NativeSoilState {
            imd: 0.25,
            f_ft: 0.05,
            fu_ft: 0.001,
            saturated: true,
            t_s: 25.0,
        };
        let source = ModifiedGreenAmptBatch::<CpuRuntime>::from_native_states(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(128))),
            &parameters,
            &[state],
        )?;
        let context = Arc::clone(&target.context);
        target.restore_trusted(&source)?;
        assert!(Arc::ptr_eq(&target.context, &context));
        assert_eq!(target.context.client.parallelism().chunk_size, Some(64));
        assert_eq!(source.context.client.parallelism().chunk_size, Some(128));
        assert_eq!(target.native_states()?, vec![state]);
        Ok(())
    }
}
