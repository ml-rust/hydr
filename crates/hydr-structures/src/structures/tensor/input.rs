//! Original head support and exact input identities, uploaded once per proposal.
use numr::{dtype::DType, runtime::Runtime};

use super::TensorStructureContext;
use crate::structures::{
    HeadConvention, HeadSupport, StructureError, StructureHeads, StructureInputErrorKind,
};
use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorStructureInput<R: Runtime<DType = DType>> {
    pub gate: Option<super::gate::TensorGateInput<R>>,
    pub pump: Option<super::pump::TensorPumpInput<R>>,
    pub stage_a: Tensor<R>,
    pub stage_b: Tensor<R>,
    pub bed_a: Tensor<R>,
    pub bed_b: Tensor<R>,
    pub wet_a: Tensor<R>,
    pub wet_b: Tensor<R>,
    pub unknown_a: Tensor<R>,
    pub unknown_b: Tensor<R>,
    /// Outlet roots of an earlier evaluation, where the culvert root searches start.
    pub culvert_hint: Option<super::culvert::CulvertHint<R>>,
}
#[derive(Debug)]
pub(crate) struct TensorStructureHeads<R: Runtime<DType = DType>> {
    stage_a: Tensor<R>,
    stage_b: Tensor<R>,
    bed_a: Tensor<R>,
    bed_b: Tensor<R>,
    wet_a: Tensor<R>,
    wet_b: Tensor<R>,
    unknown_a: Tensor<R>,
    unknown_b: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorStructureInput<R> {
    pub(crate) fn with_static_operations(
        context: &TensorStructureContext<R>,
        heads: TensorStructureHeads<R>,
        operations: &super::TensorStaticOperations<R>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R> + numr::ops::IndexingOps<R> + numr::ops::ReduceOps<R>,
    {
        Self::from_parts(
            context,
            heads,
            operations.gate(),
            operations.pump(),
            &operations.isolated,
        )
    }

    pub(crate) fn with_operations(
        context: &TensorStructureContext<R>,
        heads: &[StructureHeads],
        operations: &[crate::structures::StructureOperation],
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let heads = Self::heads(context, heads)?;
        let operations = super::operations::CheckedOperations::new(context, operations)?;
        Self::checked(context, heads, &operations)
    }
    pub(crate) fn with_trajectory(
        context: &TensorStructureContext<R>,
        heads: &[StructureHeads],
        evaluation: &crate::structures::GateTrajectoryEvaluationData<R>,
        other: &[crate::structures::StructureOperation],
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
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
        context: &TensorStructureContext<R>,
        heads: TensorStructureHeads<R>,
        manual: &[crate::structures::StructureOperation],
        gates: Option<super::operations::DerivedGateOperations<'_, R>>,
        pumps: Option<super::operations::DerivedPumpOperations<'_, R>>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let operations =
            super::operations::CheckedOperations::derived(context, manual, gates, pumps)?;
        Self::checked(context, heads, &operations)
    }
    pub(crate) fn heads(
        context: &TensorStructureContext<R>,
        heads: &[StructureHeads],
    ) -> Result<TensorStructureHeads<R>, StructureError> {
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
        Ok(TensorStructureHeads {
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
        context: &TensorStructureContext<R>,
        heads: TensorStructureHeads<R>,
        operations: &super::operations::CheckedOperations<'_, R>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        Self::from_parts(
            context,
            heads,
            super::gate::TensorGateInput::new(context, operations)?,
            super::pump::TensorPumpInput::new(context, operations)?,
            operations.isolated(),
        )
    }
    fn from_parts(
        context: &TensorStructureContext<R>,
        heads: TensorStructureHeads<R>,
        gate: Option<super::gate::TensorGateInput<R>>,
        pump: Option<super::pump::TensorPumpInput<R>>,
        isolated: &Tensor<R>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R> + numr::ops::IndexingOps<R> + numr::ops::ReduceOps<R>,
    {
        let input = Self {
            gate,
            pump,
            stage_a: heads.stage_a,
            stage_b: heads.stage_b,
            bed_a: heads.bed_a,
            bed_b: heads.bed_b,
            wet_a: heads.wet_a,
            wet_b: heads.wet_b,
            unknown_a: heads.unknown_a,
            unknown_b: heads.unknown_b,
            culvert_hint: None,
        };
        input.check_unknown(context, isolated)?;
        Ok(input)
    }
    fn check_unknown(
        &self,
        context: &TensorStructureContext<R>,
        isolated: &Tensor<R>,
    ) -> Result<(), StructureError>
    where
        R::Client: numr::ops::BinaryOps<R> + numr::ops::IndexingOps<R> + numr::ops::ReduceOps<R>,
    {
        use numr::ops::BinaryOps;
        let c = &context.client;
        context.reject_input(
            &c.mul(&c.maximum(&self.unknown_a, &self.unknown_b)?, &c.sub(&context.one, isolated)?)?,
            StructureInputErrorKind::UnknownSupport,
            "unknown influential head support cannot become dry or zero. Supply known wet/dry support",
        )
    }
}

impl<R: Runtime<DType = DType>> TensorStructureHeads<R> {
    pub(crate) fn from_tensors(
        context: &TensorStructureContext<R>,
        stage_a: &Tensor<R>,
        stage_b: &Tensor<R>,
        bed_a: &Tensor<R>,
        bed_b: &Tensor<R>,
        wet_a: &Tensor<R>,
        wet_b: &Tensor<R>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        use numr::ops::{BinaryOps, CompareOps, ConditionalOps, TypeConversionOps, UnaryOps};

        for (name, value) in [
            ("stage_a", stage_a),
            ("stage_b", stage_b),
            ("bed_a", bed_a),
            ("bed_b", bed_b),
            ("wet_a", wet_a),
            ("wet_b", wet_b),
        ] {
            check_tensor(context, value, name)?;
        }
        let c = &context.client;
        let mut nonfinite = context.zero.clone();
        for value in [stage_a, stage_b, bed_a, bed_b] {
            let invalid = c.maximum(
                &c.cast(&c.isnan(value)?, DType::F64)?,
                &c.cast(&c.isinf(value)?, DType::F64)?,
            )?;
            nonfinite = c.maximum(&nonfinite, &invalid)?;
        }
        context.reject_input(
            &nonfinite,
            StructureInputErrorKind::NonfiniteInput,
            "derived stage/bed contains a nonfinite value",
        )?;
        let mut invalid_flags = context.zero.clone();
        for value in [wet_a, wet_b] {
            let valid = c.maximum(&c.eq(value, &context.zero)?, &c.eq(value, &context.one)?)?;
            invalid_flags = c.maximum(&invalid_flags, &c.sub(&context.one, &valid)?)?;
        }
        context.reject_input(
            &invalid_flags,
            StructureInputErrorKind::Definition,
            "derived wet_a/wet_b require exact zero/one support flags",
        )?;
        Ok(Self {
            stage_a: c.where_cond(wet_a, stage_a, &context.zero)?,
            stage_b: c.where_cond(wet_b, stage_b, &context.zero)?,
            bed_a: c.where_cond(wet_a, &context.zero, bed_a)?,
            bed_b: c.where_cond(wet_b, &context.zero, bed_b)?,
            wet_a: wet_a.clone(),
            wet_b: wet_b.clone(),
            unknown_a: context.zero.clone(),
            unknown_b: context.zero.clone(),
        })
    }
}

pub(crate) fn check_tensor<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    value: &Tensor<R>,
    name: &str,
) -> Result<(), StructureError> {
    use numr::runtime::Device;

    if value.shape() != [context.elements.len()].as_slice() {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            format!(
                "derived {name} shape={:?},structure count={}: supply the complete original vector",
                value.shape(),
                context.elements.len()
            ),
        ));
    }
    if value.dtype() != DType::F64 {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            format!(
                "derived {name} dtype={:?}: supply an F64 tensor",
                value.dtype()
            ),
        ));
    }
    if !context.device.is_same(value.device()) {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            format!(
                "derived {name} uses another device context: supply tensors on the configured structure device"
            ),
        ));
    }
    Ok(())
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
