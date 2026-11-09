//! Original head support and exact input identities, uploaded once per proposal.

use super::CpuStructureContext;
use crate::cpu::CpuTensor;
use crate::structures::{
    HeadConvention, HeadSupport, StructureError, StructureHeads, StructureInputErrorKind,
};

#[derive(Debug)]
pub(crate) struct CpuStructureInput {
    pub pump: Option<super::pump::CpuPumpInput>,
    pub stage_a: CpuTensor,
    pub stage_b: CpuTensor,
    pub bed_a: CpuTensor,
    pub bed_b: CpuTensor,
    pub wet_a: CpuTensor,
    pub wet_b: CpuTensor,
}
impl CpuStructureInput {
    pub(crate) fn with_operations(
        context: &CpuStructureContext,
        heads: &[StructureHeads],
        operations: &[crate::structures::StructureOperation],
    ) -> Result<Self, StructureError> {
        let pump = super::pump::CpuPumpInput::new(context, operations)?;
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
        let (mut stage_a, mut stage_b, mut bed_a, mut bed_b, mut wet_a, mut wet_b) = (
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
            let isolated = operations[index].operating_point
                == crate::structures::StructureOperatingPoint::PumpStoppedIsolated;
            let (a, ba, wa) =
                support(&input.endpoint_a.support, isolated).map_err(|e| e.at(index))?;
            let (b, bb, wb) =
                support(&input.endpoint_b.support, isolated).map_err(|e| e.at(index))?;
            stage_a.push(a);
            stage_b.push(b);
            bed_a.push(ba);
            bed_b.push(bb);
            wet_a.push(wa);
            wet_b.push(wb);
        }
        Ok(Self {
            pump,
            stage_a: context.tensor(&stage_a)?,
            stage_b: context.tensor(&stage_b)?,
            bed_a: context.tensor(&bed_a)?,
            bed_b: context.tensor(&bed_b)?,
            wet_a: context.tensor(&wet_a)?,
            wet_b: context.tensor(&wet_b)?,
        })
    }
}
fn support(value: &HeadSupport, isolated: bool) -> Result<(f64, f64, f64), StructureError> {
    match value {
        HeadSupport::Wet { stage_m } if stage_m.is_finite() => Ok((*stage_m, 0.0, 1.0)),
        HeadSupport::Dry { bed_elevation_m } if bed_elevation_m.is_finite() => {
            Ok((0.0, *bed_elevation_m, 0.0))
        }
        HeadSupport::Unknown if isolated => Ok((0.0, 0.0, 0.0)),
        HeadSupport::Unknown => Err(StructureError::input(
            StructureInputErrorKind::UnknownSupport,
            "unknown influential head support cannot become dry or zero. Supply known wet/dry support",
        )),
        _ => Err(StructureError::input(
            StructureInputErrorKind::NonfiniteInput,
            format!("support={value:?}: supply a finite original stage or dry bed"),
        )),
    }
}
