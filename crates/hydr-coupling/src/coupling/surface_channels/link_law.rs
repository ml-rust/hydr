//! Link laws at node stages: the equal-heads rule and law rejections at the accepted start.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::structures::LinkOperations;
use crate::coupling::links::culvert_roots::CulvertRoots;
use crate::structures::{StructureError, StructureInputErrorKind, StructureLaw};
use hydr_structures::internal::TensorStructureResult;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Structures regime codes of submerged-type passive flow: fully submerged orifice, submerged
/// weir, mixed gate aperture, and submerged gate aperture.
///
/// Each one moves water down a head difference between two supported sides. At level heads it
/// counts as equal heads and carries no water. Pump regimes never take that rule, since a pump
/// moves water at equal heads.
const SUBMERGED_CODES: [f64; 4] = [1.0, 3.0, 11.0, 12.0];

/// Structures regime codes of flowing culverts: inlet control, outlet control, and a tie.
///
/// A culvert counts as submerged when both sides stand above their inverts.
const CULVERT_CODES: [f64; 3] = [13.0, 14.0, 15.0];

/// Structures regime code of equal heads above the crest.
const EQUAL_HEADS_CODE: f64 = 5.0;

/// Stages of one law evaluation.
pub(super) enum LawAt<'a, R: Runtime<DType = DType>> {
    /// The accepted start stages.
    Start,
    /// Stages inside an attempt, whose latest culvert roots start the root searches.
    Trial(&'a CulvertRoots<R>),
}

impl<R: Runtime<DType = DType>> Clone for LawAt<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime<DType = DType>> Copy for LawAt<'_, R> {}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Law of every link under `operations` at group `stage` with support `wet`, both
    /// `[S + C + P]`, at the stages `at`.
    ///
    /// A submerged link where `level` is one counts as equal heads and carries no water.
    pub fn weir(
        &self,
        operations: &LinkOperations<R>,
        (stage, wet): (&Tensor<R>, &Tensor<R>),
        level: &Tensor<R>,
        at: LawAt<'_, R>,
    ) -> Result<TensorStructureResult<R>, SurfaceChannelError> {
        let law = self.raw_weir(operations, (stage, wet), at)?;
        self.level_weir(operations, law, stage, level)
    }

    /// Law of every link under `operations` at group `stage` with support `wet`, before the
    /// equal-heads rule of [`Context::level_weir`].
    pub fn raw_weir(
        &self,
        operations: &LinkOperations<R>,
        (stage, wet): (&Tensor<R>, &Tensor<R>),
        at: LawAt<'_, R>,
    ) -> Result<TensorStructureResult<R>, SurfaceChannelError> {
        let c = &self.client;
        let gather = |values: &Tensor<R>, index: &Tensor<R>| c.index_select(values, 0, index);
        self.link_law(
            operations,
            [&gather(stage, &self.link_a)?, &gather(stage, &self.link_b)?],
            [&gather(wet, &self.link_a)?, &gather(wet, &self.link_b)?],
            at,
        )
    }

    /// The equal-heads rule of [`Context::link_weir`] over `law`, a [`Context::raw_weir`] result
    /// at group `stage`.
    pub fn level_weir(
        &self,
        operations: &LinkOperations<R>,
        law: TensorStructureResult<R>,
        stage: &Tensor<R>,
        level: &Tensor<R>,
    ) -> Result<TensorStructureResult<R>, SurfaceChannelError> {
        let c = &self.client;
        let gather = |index: &Tensor<R>| c.index_select(stage, 0, index);
        self.level_law(
            operations,
            law,
            [&gather(&self.link_a)?, &gather(&self.link_b)?],
            level,
        )
    }

    /// One where a link's endpoint stages in `stage`, `[S + C + P]`, differ by at most the head
    /// allowance, `[L]`.
    ///
    /// The allowance is `head_absolute_m + head_relative * max(|H_s|, |H_c|)`.
    pub fn level_heads(&self, stage: &Tensor<R>) -> Result<Tensor<R>, SurfaceChannelError> {
        let c = &self.client;
        let surface = c.index_select(stage, 0, &self.link_a)?;
        let channel = c.index_select(stage, 0, &self.link_b)?;
        let allowance = self.head_allowance(&surface, &channel)?;
        Ok(c.le(&c.abs(&c.sub(&surface, &channel)?)?, &allowance)?)
    }

    /// The coordinator error of law rejection `error`, at the accepted `start` stages or not.
    ///
    /// An orifice outside full submergence at the accepted start rejects without retry: partial
    /// orifice flow is no admitted law, and no shorter interval changes the start. Inside an
    /// attempt the rejection stays numerical, so a crown crossing shortens the interval.
    fn law_error(&self, error: &StructureError, start: bool) -> SurfaceChannelError {
        if start
            && error.input_kind() == Some(StructureInputErrorKind::UnsupportedRegime)
            && let Some(element) = error.element().and_then(|e| self.link_elements.get(e))
            && matches!(element.law, StructureLaw::SubmergedOrifice(_))
        {
            return SurfaceChannelError::input(format!(
                "orifice-partial-submergence: structure={:?} at the accepted start: {error}: raise both sides above the crown, or declare a weir or gate for partial flow",
                element.id
            ));
        }
        SurfaceChannelError::Numerical(format!("link weir: {error}"))
    }

    /// Law of every link under `operations` at its surface and channel `stages` and `wet` masks,
    /// each `[L]`, at the stages `at`.
    ///
    /// A side supports flow when it is wet and stands above its attachment bed. A link in a
    /// submerged-type regime where `level` is one counts as equal heads and carries no water. A
    /// flowing culvert with both sides above their inverts is submerged-type.
    pub fn link_weir(
        &self,
        operations: &LinkOperations<R>,
        [surface, channel]: [&Tensor<R>; 2],
        [wet_surface, wet_channel]: [&Tensor<R>; 2],
        (level, at): (&Tensor<R>, LawAt<'_, R>),
    ) -> Result<TensorStructureResult<R>, SurfaceChannelError> {
        let law = self.link_law(
            operations,
            [surface, channel],
            [wet_surface, wet_channel],
            at,
        )?;
        self.level_law(operations, law, [surface, channel], level)
    }

    /// Law of every link under `operations` at its surface and channel `stages` and `wet` masks,
    /// each `[L]`, before the equal-heads rule.
    fn link_law(
        &self,
        operations: &LinkOperations<R>,
        [surface, channel]: [&Tensor<R>; 2],
        [wet_surface, wet_channel]: [&Tensor<R>; 2],
        at: LawAt<'_, R>,
    ) -> Result<TensorStructureResult<R>, SurfaceChannelError> {
        let c = &self.client;
        let structures = operations.structures()?;
        let stage = [surface, channel];
        let bed = [&self.bed_a, &self.bed_b];
        let wet_a = c.mul(wet_surface, &c.gt(surface, &self.bed_a)?)?;
        let wet_b = c.mul(wet_channel, &c.gt(channel, &self.bed_b)?)?;
        let wet = [&wet_a, &wet_b];
        match at {
            LawAt::Start => structures.evaluate_heads_from(stage, bed, wet, None),
            LawAt::Trial(roots) => roots.evaluate(structures, stage, bed, wet),
        }
        .map_err(|error| self.law_error(&error, matches!(at, LawAt::Start)))
    }

    /// The equal-heads rule over `law` at end `stages`, each `[L]`.
    ///
    /// A link in a submerged-type regime where `level` is one carries no water.
    fn level_law(
        &self,
        operations: &LinkOperations<R>,
        mut law: TensorStructureResult<R>,
        [surface, channel]: [&Tensor<R>; 2],
        level: &Tensor<R>,
    ) -> Result<TensorStructureResult<R>, SurfaceChannelError> {
        let c = &self.client;
        let mut submerged = c.eq(&law.regime, &self.scalar(SUBMERGED_CODES[0])?)?;
        for code in &SUBMERGED_CODES[1..] {
            submerged = c.maximum(&submerged, &c.eq(&law.regime, &self.scalar(*code)?)?)?;
        }
        if operations.culverts.is_some() {
            let mut flowing = c.eq(&law.regime, &self.scalar(CULVERT_CODES[0])?)?;
            for code in &CULVERT_CODES[1..] {
                flowing = c.maximum(&flowing, &c.eq(&law.regime, &self.scalar(*code)?)?)?;
            }
            let above = c.mul(
                &c.gt(surface, &operations.cut[0])?,
                &c.gt(channel, &operations.cut[1])?,
            )?;
            submerged = c.maximum(&submerged, &c.mul(&flowing, &above)?)?;
        }
        let level = c.mul(&submerged, level)?;
        law.regime = c.where_cond(&level, &self.scalar(EQUAL_HEADS_CODE)?, &law.regime)?;
        law.discharge = c.where_cond(&level, &self.scalar(0.0)?, &law.discharge)?;
        Ok(law)
    }
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelError;
    use super::super::structures::scenario::{
        DT, EQUAL_HEADS, PUMP_CURVE, Tanks, TestResult, check, gate_law, orifice_law, pump_law,
        receipt, step, structure,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// An orifice and a gate between tanks at rest, beside a pump between the same tanks.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Over 5 intervals both passive links take equal heads and carry `0.0`.
    /// - The pump keeps its curve regime and moves water at equal heads.
    /// - Every interval passes its mass bound, both pairings, and the composite balance.
    #[test]
    fn passive_links_rest_at_equal_heads_while_a_pump_runs() -> TestResult {
        let passive = |id: &str, law, operation| {
            structure(
                id,
                law,
                DirectionPolicy::Bidirectional,
                operation,
                [0.0, 0.0],
            )
        };
        let structures = vec![
            passive(
                "culvert",
                orifice_law(0.1, 0.3)?,
                StructureOperatingPoint::Passive,
            ),
            passive(
                "gate",
                gate_law(0.2)?,
                StructureOperatingPoint::GateOpening { opening_m: 0.2 },
            ),
        ];
        let mut batch = Tanks::at(0.6, 0.6).build(structures)?;
        for _ in 0..5 {
            batch = step(&batch, DT, &[])?;
            check(&batch)?;
            for id in ["culvert", "gate"] {
                let link = receipt(&batch, id)?;
                assert_eq!(link.regime, EQUAL_HEADS, "{id}: {link:?}");
                assert_eq!(link.signed, 0.0, "{id}: {link:?}");
            }
        }
        let pump = structure(
            "pump",
            pump_law(0.3, None)?,
            DirectionPolicy::AtoBOnly,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            [0.0, 0.0],
        );
        let batch = step(&Tanks::at(0.6, 0.6).build(vec![pump])?, DT, &[])?;
        check(&batch)?;
        let link = receipt(&batch, "pump")?;
        assert_eq!(link.regime, PUMP_CURVE, "{link:?}");
        assert!(link.signed > 0.0, "{link:?}");
        Ok(())
    }

    /// Frozen criterion, fixed before the first run: an orifice with one side below its crown at
    /// the accepted start rejects the first trial without retry, naming
    /// `orifice-partial-submergence` and the structure.
    #[test]
    fn a_partial_orifice_at_the_start_rejects() -> TestResult {
        let culvert = structure(
            "culvert",
            orifice_law(0.1, 0.3)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            [0.0, 0.0],
        );
        match step(&Tanks::at(0.6, 0.2).build(vec![culvert])?, DT, &[]) {
            Err(error @ SurfaceChannelError::Input(_)) => {
                assert!(!error.is_retryable());
                let message = error.to_string();
                assert!(
                    message.contains("orifice-partial-submergence")
                        && message.contains("\"culvert\""),
                    "{message}"
                );
                Ok(())
            }
            other => Err(format!("a partial orifice at the start rejects: {other:?}").into()),
        }
    }
}
