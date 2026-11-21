//! Warm outlet windows: where each outlet root search starts, from an earlier root.
//!
//! A hint holds an earlier root enclosure `[a, b]` at heads `HW_h` and `TW_h`, with bounds
//! `s_h` on `|dQ/dHW| / Q` and `s_t` on `|dQ/dTW| / Q`. To first order the root at heads `HW`
//! and `TW` moves by at most `q (s_h |HW - HW_h| + s_t |TW - TW_h|)`, with `q` the midpoint.
//! The window widens `[a, b]` on each side by twice that, plus its own width `b - a` and
//! [`WINDOW_FLOOR`] times `q`, inside `[0, ceiling]`.
//!
//! A window is only a guess. The root search certifies both its ends by residual signs, and a
//! lane whose window misses the root searches again from zero. So a window moves bits, but every
//! enclosure stays certified.

use super::lanes::Lanes;
use crate::structures::tensor::culvert::{Bounds, CulvertHint, TensorBarrelContext};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Relative widening `2^-44` of every window beyond its root enclosure.
const WINDOW_FLOOR: f64 = 1.0 / 17_592_186_044_416.0;

/// Margin over the first-order root move.
const MOVE_MARGIN: f64 = 2.0;

/// Start bracket of every outlet root search.
pub(crate) struct Window<R: Runtime<DType = DType>> {
    /// 1 where the bracket comes from the hint.
    pub(crate) warm: Tensor<R>,
    /// Start bracket: the window on warm lanes, `[0, ceiling]` elsewhere.
    pub(crate) bounds: Bounds<R>,
}

impl<R: Runtime<DType = DType>> Window<R> {
    /// The bracket `[0, ceiling]` on every lane.
    pub(crate) fn cold(barrel: &TensorBarrelContext<R>, ceiling: &Tensor<R>) -> Self {
        Self {
            warm: barrel.zero.clone(),
            bounds: [barrel.zero.clone(), ceiling.clone()],
        }
    }
}

/// Start brackets of the `search` lanes, whose roots lie in `[0, ceiling]`.
pub(crate) fn window<R>(
    barrel: &TensorBarrelContext<R>,
    lanes: &Lanes<R>,
    search: &Tensor<R>,
    ceiling: &Tensor<R>,
    hint: Option<&CulvertHint<R>>,
) -> Result<Window<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let c = &barrel.client;
    let zero = &barrel.zero;
    let Some(hint) = hint else {
        return Ok(Window::cold(barrel, ceiling));
    };
    let rows = c.cast(&lanes.rows, DType::F64)?;
    let same = c.mul(&hint.held, &c.eq(&rows, &hint.rows)?)?;
    let [per_head, per_tail] = &hint.slope;
    let moved = c.add(
        &c.mul(
            per_head,
            &c.abs(&c.sub(&lanes.headwater, &hint.headwater)?)?,
        )?,
        &c.mul(
            per_tail,
            &c.abs(&c.sub(&lanes.tailwater, &hint.tailwater)?)?,
        )?,
    )?;
    let [low, high] = &hint.root;
    let width = c.sub(high, low)?;
    let middle = c.add(&c.mul_scalar(low, 0.5)?, &c.mul_scalar(high, 0.5)?)?;
    let reach = c.add_scalar(&c.mul_scalar(&moved, MOVE_MARGIN)?, WINDOW_FLOOR)?;
    let step = c.add(&c.mul(&middle, &reach)?, &width)?;
    let lower = c.maximum(&c.sub(low, &step)?, zero)?;
    let upper = c.minimum(&c.add(high, &step)?, ceiling)?;
    let open = c.lt(&lower, &upper)?;
    let warm = c.mul(&c.mul(search, &same)?, &open)?;
    Ok(Window {
        bounds: [
            c.where_cond(&warm, &lower, zero)?,
            c.where_cond(&warm, &upper, ceiling)?,
        ],
        warm,
    })
}

#[cfg(test)]
mod tests {
    use crate::structures::coupling::CouplingStructures;
    use crate::structures::numerics::sensitivity::fixture::{
        BED, TestResult, column, culvert_structures,
    };
    use crate::structures::tensor::TensorStructureResult;
    use crate::structures::tensor::culvert::CulvertHint;
    use crate::structures::{
        CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds,
        CulvertInlet, CulvertInletForm, CulvertParameters, DirectionPolicy, StructureLaw,
        StructureOperatingPoint,
    };
    use numr::runtime::cpu::CpuRuntime;

    /// A two-way mild circular culvert whose explicit inlet is so weak that the outlet governs.
    fn outlet_law() -> TestResult<StructureLaw> {
        let inlet = CulvertInlet::explicit(CulvertInletForm::Two, [1e-3, 1.0, 1e-4, 0.05], false)?;
        let end = CulvertEnd::new(inlet, 0.5)?;
        Ok(StructureLaw::Culvert(CulvertParameters::new(
            CulvertBarrel::circular(1.2)?,
            CulvertAlignment {
                length_m: 60.0,
                invert_a_m: 0.12,
                invert_b_m: 0.0,
            },
            0.013,
            1,
            CulvertEnds {
                a: Some(end),
                b: Some(end),
            },
            9.81,
            CulvertApplicability {
                coefficient_provenance: "Analytical coefficients".into(),
                barrel_condition_assertion: "Clean analytical barrel".into(),
            },
        )?))
    }

    /// Three lanes of the outlet law.
    fn structures() -> TestResult<CouplingStructures<CpuRuntime>> {
        let law = outlet_law()?;
        culvert_structures(
            |_| {
                (
                    law.clone(),
                    DirectionPolicy::Bidirectional,
                    StructureOperatingPoint::Passive,
                )
            },
            3,
        )
    }

    /// Laws at wet `stages` of ends `[A, B]`, with the root searches started at `hint`.
    fn laws(
        structures: &CouplingStructures<CpuRuntime>,
        stages: &[[f64; 2]],
        hint: Option<&CulvertHint<CpuRuntime>>,
    ) -> TestResult<TensorStructureResult<CpuRuntime>> {
        let side = |k: usize| column(stages.iter().map(|s| s[k]).collect());
        let bed = column(vec![BED; stages.len()])?;
        let wet = column(vec![1.0; stages.len()])?;
        Ok(structures.evaluate_heads_from(
            [&side(0)?, &side(1)?],
            [&bed, &bed],
            [&wet, &wet],
            hint,
        )?)
    }

    /// Per-barrel discharge enclosure of every lane.
    fn enclosures(law: &TensorStructureResult<CpuRuntime>) -> TestResult<Vec<(f64, f64)>> {
        let culvert = law.culvert.as_ref().ok_or("culvert lanes")?;
        let lower = culvert.lower.try_to_vec::<f64>()?;
        let upper = culvert.upper.try_to_vec::<f64>()?;
        Ok(lower.into_iter().zip(upper).collect())
    }

    /// Frozen criteria, fixed before the first run:
    /// - The cold laws hold an outlet root on every lane.
    /// - At heads moved by up to `1e-3` m, laws started at those roots report the cold regimes.
    ///   Each enclosure overlaps the cold one. It spans at most twice the cold width, or `2^-40`
    ///   of its upper end when that is wider.
    /// - A hint from heads `0.4` m away meets the same two criteria.
    #[test]
    fn hinted_roots_enclose_the_cold_roots() -> TestResult {
        let structures = structures()?;
        let start = [[1.0, 0.5], [0.9, 0.7], [1.1, 0.3]];
        let cold = laws(&structures, &start, None)?;
        let hint = cold.culvert_hint().ok_or("a culvert hint")?;
        assert_eq!(hint.held.try_to_vec::<f64>()?, [1.0, 1.0, 1.0]);
        for moved in [
            [[1.0005, 0.501], [0.9, 0.699], [1.1001, 0.3]],
            [[1.4, 0.5], [1.3, 0.7], [1.5, 0.3]],
        ] {
            let hinted = laws(&structures, &moved, Some(hint))?;
            let reference = laws(&structures, &moved, None)?;
            assert_eq!(
                hinted.regime.try_to_vec::<f64>()?,
                reference.regime.try_to_vec::<f64>()?,
                "{moved:?}"
            );
            for ((low, high), (cold_low, cold_high)) in enclosures(&hinted)?
                .into_iter()
                .zip(enclosures(&reference)?)
            {
                assert!(low <= cold_high && cold_low <= high, "{low} {high}");
                let limit = (2.0 * (cold_high - cold_low)).max(high * 2f64.powi(-40));
                assert!(
                    high - low <= limit,
                    "{low} {high} cold {cold_low} {cold_high}"
                );
            }
        }
        Ok(())
    }

    /// Frozen criterion, fixed before the first run: a hint from flow `A` to `B` leaves a lane
    /// that now flows from `B` to `A` on the cold search, bit for bit.
    #[test]
    fn a_reversed_lane_ignores_its_hint() -> TestResult {
        let structures = structures()?;
        let forward = laws(&structures, &[[1.0, 0.5], [0.9, 0.7], [1.1, 0.3]], None)?;
        let reversed = [[0.5, 1.0], [0.7, 0.9], [0.3, 1.1]];
        let hinted = laws(&structures, &reversed, forward.culvert_hint())?;
        let cold = laws(&structures, &reversed, None)?;
        let bits = |law: &TensorStructureResult<CpuRuntime>| -> TestResult<Vec<u64>> {
            Ok(law
                .discharge
                .try_to_vec::<f64>()?
                .iter()
                .map(|q| q.to_bits())
                .collect())
        };
        assert_eq!(bits(&hinted)?, bits(&cold)?);
        Ok(())
    }
}
