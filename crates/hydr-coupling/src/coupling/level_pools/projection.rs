use super::{
    CoupledLevelPoolBatch, CoupledLevelPoolError, CoupledLevelPoolIterationData,
    CoupledLevelPoolStepDiagnostics, CoupledLevelPoolTransfer, context::Context, trial::StoredStep,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

impl<R: Runtime<DType = DType>> CoupledLevelPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Explicitly read the complete final interval and physical transfer support.
    pub fn last_step(
        &self,
    ) -> Result<Option<CoupledLevelPoolStepDiagnostics>, CoupledLevelPoolError> {
        self.last
            .as_ref()
            .map(|step| self.context.project(step))
            .transpose()
    }
}
impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    pub fn project(
        &self,
        step: &StoredStep<R>,
    ) -> Result<CoupledLevelPoolStepDiagnostics, CoupledLevelPoolError> {
        let n = self.pool_count();
        let e = self.edge_count();
        let packet = step.input.try_to_vec::<f64>()?;
        let proposed = step.exchange.proposed.try_to_vec::<f64>()?;
        let signed = step.exchange.signed.try_to_vec::<f64>()?;
        let factor = step.exchange.factor.try_to_vec::<f64>()?;
        let a = step.heads.a.try_to_vec::<f64>()?;
        let b = step.heads.b.try_to_vec::<f64>()?;
        let wa = step.heads.wet_a.try_to_vec::<f64>()?;
        let wb = step.heads.wet_b.try_to_vec::<f64>()?;
        if packet.len() != 2 * n + 3 * e
            || [
                proposed.len(),
                signed.len(),
                factor.len(),
                a.len(),
                b.len(),
                wa.len(),
                wb.len(),
            ]
            .iter()
            .any(|v| *v != e)
        {
            return Err(CoupledLevelPoolError::numerical(
                "stored coupled receipt dimensions disagree with immutable topology",
            ));
        }
        let iteration = CoupledLevelPoolIterationData {
            end_stages_m: packet[..n].to_vec(),
            signed_volumes_m3: packet[n..n + e].to_vec(),
            pool_wet: packet[n + e..2 * n + e].iter().map(|v| *v == 1.0).collect(),
            attachment_a_wet: packet[2 * n + e..2 * n + 2 * e]
                .iter()
                .map(|v| *v == 1.0)
                .collect(),
            attachment_b_wet: packet[2 * n + 2 * e..].iter().map(|v| *v == 1.0).collect(),
        };
        let transfers = (0..e)
            .map(|i| CoupledLevelPoolTransfer {
                structure_id: self.setup.structures[i].id.clone(),
                proposed_volume_m3: proposed[i],
                signed_volume_m3: signed[i],
                donor_limit_factor: factor[i],
                midpoint_stage_a_m: if wa[i] == 1.0 { Some(a[i]) } else { None },
                midpoint_stage_b_m: if wb[i] == 1.0 { Some(b[i]) } else { None },
            })
            .collect();
        Ok(CoupledLevelPoolStepDiagnostics {
            external_inflows: self.inflows.project_step(&self.client, &step.external)?,
            forcing_interval: step
                .forcing_interval
                .as_ref()
                .map(|forcing| forcing.project(&self.client, &self.inflows))
                .transpose()?,
            start_s: step.start,
            end_s: step.end,
            requested_end_s: step.requested,
            iterations: step.iterations,
            retries: step.retries,
            head_residual_m: step.head_residual,
            transfer_residual_m3: step.transfer_residual,
            mass_residual_m3: step.mass_residual,
            aggregation_roundoff_bound_m3: step.exchange.aggregation_bound,
            iteration,
            transfers,
        })
    }
    pub fn import_packet(
        &self,
        data: &CoupledLevelPoolIterationData,
    ) -> Result<Tensor<R>, CoupledLevelPoolError> {
        let n = self.pool_count();
        let e = self.edge_count();
        if data.end_stages_m.len() != n
            || data.signed_volumes_m3.len() != e
            || data.pool_wet.len() != n
            || data.attachment_a_wet.len() != e
            || data.attachment_b_wet.len() != e
        {
            return Err(CoupledLevelPoolError::input(
                "checkpoint iteration dimensions differ from immutable topology",
            ));
        }
        let mut values = Vec::with_capacity(2 * n + 3 * e);
        values.extend_from_slice(&data.end_stages_m);
        values.extend_from_slice(&data.signed_volumes_m3);
        for flags in [
            &data.pool_wet,
            &data.attachment_a_wet,
            &data.attachment_b_wet,
        ] {
            values.extend(flags.iter().map(|v| if *v { 1.0 } else { 0.0 }));
        }
        let packet = Tensor::<R>::from_slice(&values, &[values.len()], self.client.device())?;
        self.check_packet(&packet)?;
        Ok(packet)
    }
}
