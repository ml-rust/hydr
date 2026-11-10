//! Original head support and exact input identities, uploaded once per proposal.

use super::CpuStructureContext;
use crate::cpu::CpuTensor;
use crate::structures::{
    HeadConvention, HeadSupport, StructureError, StructureHeads, StructureInputErrorKind,
};

#[derive(Debug)]
pub(crate) struct CpuStructureInput {
    pub gate: Option<super::gate::CpuGateInput>,
    pub pump: Option<super::pump::CpuPumpInput>,
    pub stage_a: CpuTensor,
    pub stage_b: CpuTensor,
    pub bed_a: CpuTensor,
    pub bed_b: CpuTensor,
    pub wet_a: CpuTensor,
    pub wet_b: CpuTensor,
    pub unknown_a: CpuTensor,
    pub unknown_b: CpuTensor,
}
#[derive(Debug)]
pub(crate) struct CpuStructureHeads {
    stage_a: CpuTensor,
    stage_b: CpuTensor,
    bed_a: CpuTensor,
    bed_b: CpuTensor,
    wet_a: CpuTensor,
    wet_b: CpuTensor,
    unknown_a: CpuTensor,
    unknown_b: CpuTensor,
}
impl CpuStructureInput {
    pub(crate) fn with_operations(
        context: &CpuStructureContext,
        heads: &[StructureHeads],
        operations: &[crate::structures::StructureOperation],
    ) -> Result<Self, StructureError> {
        let heads = Self::heads(context, heads)?;
        let operations = super::operations::CheckedOperations::new(context, operations)?;
        Self::checked(context, heads, &operations)
    }
    pub(crate) fn with_trajectory(
        context: &CpuStructureContext,
        heads: &[StructureHeads],
        evaluation: &crate::structures::GateTrajectoryEvaluationData,
        other: &[crate::structures::StructureOperation],
    ) -> Result<Self, StructureError> {
        if !std::ptr::eq(context, evaluation.binding.context.as_ref()) {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "trajectory evaluation belongs to another complete context",
            ));
        }
        let heads = Self::heads(context, heads)?;
        let operations =
            super::operations::CheckedOperations::trajectory(context, evaluation, other)?;
        Self::checked(context, heads, &operations)
    }
    pub(crate) fn with_derived_heads(
        context: &CpuStructureContext,
        heads: CpuStructureHeads,
        manual: &[crate::structures::StructureOperation],
        gates: Option<super::operations::DerivedGateOperations<'_>>,
        pumps: Option<super::operations::DerivedPumpOperations<'_>>,
    ) -> Result<Self, StructureError> {
        let operations =
            super::operations::CheckedOperations::derived(context, manual, gates, pumps)?;
        Self::checked(context, heads, &operations)
    }
    pub(crate) fn heads(
        context: &CpuStructureContext,
        heads: &[StructureHeads],
    ) -> Result<CpuStructureHeads, StructureError> {
        if heads.len() != context.elements.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                format!(
                    "head packet count={},structure count={}: supply complete original element order",
                    heads.len(),
                    context.elements.len()
                ),
            ));
        }
        let n = heads.len();
        let (
            mut stage_a,
            mut stage_b,
            mut bed_a,
            mut bed_b,
            mut wet_a,
            mut wet_b,
            mut unknown_a,
            mut unknown_b,
        ) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for (index, (input, element)) in heads.iter().zip(&context.elements).enumerate() {
            if input.structure_id != element.id
                || input.endpoint_a.id != element.endpoint_a
                || input.endpoint_b.id != element.endpoint_b
            {
                return Err(StructureError::input(StructureInputErrorKind::Identity,format!("input={input:?},structure={:?}: match exact structure and ordered endpoint identities",element.id)).at(index));
            }
            if input.datum != element.datum {
                return Err(StructureError::input(StructureInputErrorKind::Datum,format!("input datum={:?},expected={:?}: supply the common reference without transformation",input.datum,element.datum)).at(index));
            }
            if input.convention != HeadConvention::QuiescentStage {
                return Err(StructureError::input(
                    StructureInputErrorKind::HeadConvention,
                    "energy-head inputs are unsupported. Supply explicitly quiescent stage heads",
                )
                .at(index));
            }
            let (a, ba, wa, ua) = support(&input.endpoint_a.support).map_err(|e| e.at(index))?;
            let (b, bb, wb, ub) = support(&input.endpoint_b.support).map_err(|e| e.at(index))?;
            stage_a.push(a);
            stage_b.push(b);
            bed_a.push(ba);
            bed_b.push(bb);
            wet_a.push(wa);
            wet_b.push(wb);
            unknown_a.push(ua);
            unknown_b.push(ub);
        }
        Ok(CpuStructureHeads {
            stage_a: context.tensor(&stage_a)?,
            stage_b: context.tensor(&stage_b)?,
            bed_a: context.tensor(&bed_a)?,
            bed_b: context.tensor(&bed_b)?,
            wet_a: context.tensor(&wet_a)?,
            wet_b: context.tensor(&wet_b)?,
            unknown_a: context.tensor(&unknown_a)?,
            unknown_b: context.tensor(&unknown_b)?,
        })
    }
    fn checked(
        context: &CpuStructureContext,
        heads: CpuStructureHeads,
        operations: &super::operations::CheckedOperations<'_>,
    ) -> Result<Self, StructureError> {
        let input = Self {
            gate: super::gate::CpuGateInput::new(context, operations)?,
            pump: super::pump::CpuPumpInput::new(context, operations)?,
            stage_a: heads.stage_a,
            stage_b: heads.stage_b,
            bed_a: heads.bed_a,
            bed_b: heads.bed_b,
            wet_a: heads.wet_a,
            wet_b: heads.wet_b,
            unknown_a: heads.unknown_a,
            unknown_b: heads.unknown_b,
        };
        input.check_unknown(context, operations.isolated())?;
        Ok(input)
    }
    fn check_unknown(
        &self,
        context: &CpuStructureContext,
        isolated: &CpuTensor,
    ) -> Result<(), StructureError> {
        use numr::ops::BinaryOps;
        let c = &context.client;
        context.reject_input(
            &c.mul(&c.maximum(&self.unknown_a, &self.unknown_b)?, &c.sub(&context.one, isolated)?)?,
            StructureInputErrorKind::UnknownSupport,
            "unknown influential head support cannot become dry or zero. Supply known wet/dry support",
        )
    }
}
fn support(value: &HeadSupport) -> Result<(f64, f64, f64, f64), StructureError> {
    match value {
        HeadSupport::Wet { stage_m } if stage_m.is_finite() => Ok((*stage_m, 0.0, 1.0, 0.0)),
        HeadSupport::Dry { bed_elevation_m } if bed_elevation_m.is_finite() => {
            Ok((0.0, *bed_elevation_m, 0.0, 0.0))
        }
        HeadSupport::Unknown => Ok((0.0, 0.0, 0.0, 1.0)),
        _ => Err(StructureError::input(
            StructureInputErrorKind::NonfiniteInput,
            format!("support={value:?}: supply a finite original stage or dry bed"),
        )),
    }
}
