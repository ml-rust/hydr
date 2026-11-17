use super::ChannelPoolError;
use super::snapshot::{
    ChannelPoolCounterData, ChannelPoolIntervalAccountingData, ChannelPoolNativeIntervalData,
};
use super::state::{NativeInterval, NativeIntervalAccounting, ParentCounters};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

impl<R: Runtime<DType = DType>> ParentCounters<R> {
    pub(super) fn data(&self) -> Result<ChannelPoolCounterData, ChannelPoolError> {
        Ok(ChannelPoolCounterData {
            connection_in_m3: self.connection_in_m3.try_to_vec::<f64>()?,
            connection_out_m3: self.connection_out_m3.try_to_vec::<f64>()?,
            connection_in_bound_m3: self.connection_in_bound_m3.try_to_vec::<f64>()?,
            connection_out_bound_m3: self.connection_out_bound_m3.try_to_vec::<f64>()?,
            external_in_m3: self.external_in_m3.try_to_vec::<f64>()?,
            external_out_m3: self.external_out_m3.try_to_vec::<f64>()?,
            external_in_bound_m3: self.external_in_bound_m3.try_to_vec::<f64>()?,
            external_out_bound_m3: self.external_out_bound_m3.try_to_vec::<f64>()?,
            pool_incoming_assembly_bound_m3: self
                .pool_incoming_assembly_bound_m3
                .try_to_vec::<f64>()?,
            pool_outgoing_assembly_bound_m3: self
                .pool_outgoing_assembly_bound_m3
                .try_to_vec::<f64>()?,
            historical_physical_charge_m3: self
                .historical_physical_charge_m3
                .try_to_vec::<f64>()?,
        })
    }
}

impl<R: Runtime<DType = DType>> NativeInterval<R> {
    pub(super) fn data(&self) -> Result<ChannelPoolNativeIntervalData, ChannelPoolError> {
        Ok(ChannelPoolNativeIntervalData {
            input_packet: self.input_packet.try_to_vec::<f64>()?,
            output_packet: self.output_packet.try_to_vec::<f64>()?,
            midpoint_pool_stages_m: self.midpoint_pool_stages_m.try_to_vec::<f64>()?,
            connection_volume_m3: self.connection_volume_m3.try_to_vec::<f64>()?,
            connection_momentum_m4_per_s: self.connection_momentum_m4_per_s.try_to_vec::<f64>()?,
            external_volume_m3: self.external_volume_m3.try_to_vec::<f64>()?,
            cut_available_lower_m3: self.cut_available_lower_m3.try_to_vec::<f64>()?,
            cut_outgoing_upper_m3: self.cut_outgoing_upper_m3.try_to_vec::<f64>()?,
            pool_incoming_bound_m3: self.pool_incoming_bound_m3.try_to_vec::<f64>()?,
            pool_outgoing_bound_m3: self.pool_outgoing_bound_m3.try_to_vec::<f64>()?,
            head_residual_m: self.head_residual_m.try_to_vec::<f64>()?,
            transfer_residual_m3: self.transfer_residual_m3.try_to_vec::<f64>()?,
        })
    }
}

impl<R: Runtime<DType = DType>> NativeIntervalAccounting<R> {
    pub(super) fn data(&self) -> Result<ChannelPoolIntervalAccountingData, ChannelPoolError> {
        Ok(ChannelPoolIntervalAccountingData {
            storage_before_m3: self.storage_before_m3.try_to_vec::<f64>()?,
            storage_after_m3: self.storage_after_m3.try_to_vec::<f64>()?,
            external_in_m3: self.external_in_m3.try_to_vec::<f64>()?,
            external_out_m3: self.external_out_m3.try_to_vec::<f64>()?,
            lateral_in_m3: self.lateral_in_m3.try_to_vec::<f64>()?,
            lateral_out_m3: self.lateral_out_m3.try_to_vec::<f64>()?,
            residual_m3: self.residual_m3.try_to_vec::<f64>()?,
            roundoff_bound_m3: self.roundoff_bound_m3.try_to_vec::<f64>()?,
            physical_charge_m3: self.physical_charge_m3.try_to_vec::<f64>()?,
        })
    }
}

pub(super) fn import_counters<R: Runtime<DType = DType>>(
    client: &R::Client,
    data: &ChannelPoolCounterData,
) -> Result<ParentCounters<R>, ChannelPoolError> {
    let tensor = |values: &[f64]| -> Result<Tensor<R>, ChannelPoolError> {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            client.device(),
        )?)
    };
    Ok(ParentCounters {
        connection_in_m3: tensor(&data.connection_in_m3)?,
        connection_out_m3: tensor(&data.connection_out_m3)?,
        connection_in_bound_m3: tensor(&data.connection_in_bound_m3)?,
        connection_out_bound_m3: tensor(&data.connection_out_bound_m3)?,
        external_in_m3: tensor(&data.external_in_m3)?,
        external_out_m3: tensor(&data.external_out_m3)?,
        external_in_bound_m3: tensor(&data.external_in_bound_m3)?,
        external_out_bound_m3: tensor(&data.external_out_bound_m3)?,
        pool_incoming_assembly_bound_m3: tensor(&data.pool_incoming_assembly_bound_m3)?,
        pool_outgoing_assembly_bound_m3: tensor(&data.pool_outgoing_assembly_bound_m3)?,
        historical_physical_charge_m3: tensor(&data.historical_physical_charge_m3)?,
    })
}
