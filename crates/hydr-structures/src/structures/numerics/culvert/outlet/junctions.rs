//! Junction knots and sliver lanes of arc rows.
//!
//! A three-radius row has two junction angles, each an enclosure a few F64 wide. Its profile
//! takes a knot at both ends of each enclosure that lies strictly inside the profile and ahead
//! of the last guarded node. Every other knot moves to the profile start, where it leaves an
//! empty lane. So each row gains [`KNOTS`] lanes for them, and the graded nodes keep their
//! places.
//!
//! On an arc band `ds/dpsi = r_k sin psi` can exceed a circle's `1/2` many times, so a lane
//! of a sixteenth of the span can hold a large share of the depth, and its interval jets
//! overstate the fourth derivative. Arc rows therefore split each uniform middle lane into
//! four. Other rows place those nodes at the start too, where they leave empty lanes.
//!
//! A lane between the two ends of one enclosure is a sliver. Either band may hold its
//! coordinates, so no Simpson rule applies there. Its integral is its width times the hull of
//! both bands' integrand ranges. Every other arc lane lies inside one band and carries only
//! that band, so its jets are smooth.

use super::bands::ArcShape;
use super::integrand::{ProfileParams, integrand};
use super::jet;
use super::partition::ProfileSpec;
use super::quadrature::Enclosure;
use super::trig::fill;
use crate::structures::StructureError;
use crate::structures::numerics::culvert::BarrelClient;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ShapeOps, SortingOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

/// Knots per row: both ends of two junction angle enclosures.
pub(crate) const KNOTS: usize = 4;

/// `nodes` `[m, count]` with the junction knots merged in order along the profile.
pub(super) fn knots<R>(
    c: &R::Client,
    spec: &ProfileSpec<R>,
    arcs: &ArcShape<R>,
    nodes: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let m = spec.start.numel();
    let count = nodes.shape().get(1).copied().unwrap_or(1);
    let (Some(corner), Some(top)) = (arcs.bands.get(1), arcs.bands.get(2)) else {
        return Err(StructureError::numerical(
            "culvert outlet profile lost its arc bands: report this runtime",
        ));
    };
    let column = |value: &Tensor<R>| value.reshape(&[m, 1]);
    let values = c.cat(
        &[
            &column(&corner.start_angle[0])?,
            &column(&corner.start_angle[1])?,
            &column(&top.start_angle[0])?,
            &column(&top.start_angle[1])?,
        ],
        1,
    )?;
    let wide = |value: &Tensor<R>, width: usize| -> numr::error::Result<Tensor<R>> {
        value
            .reshape(&[m, 1])?
            .broadcast_to(&[m, width])?
            .contiguous()
    };
    let sign = wide(&spec.sign, KNOTS)?;
    let start = wide(&spec.start, KNOTS)?;
    // An asymptotic row's last lane reaches the pole, so its knots stop at the node before it.
    let guarded = nodes.narrow(1, count.saturating_sub(2), 1)?.contiguous()?;
    let end = c.where_cond(&spec.asymptotic, &guarded.reshape(&[m])?, &spec.limit)?;
    let end = wide(&end, KNOTS)?;
    let oriented = c.mul(&sign, &values)?;
    let after = c.gt(&oriented, &c.mul(&sign, &start)?)?;
    let before = c.lt(&oriented, &c.mul(&sign, &end)?)?;
    let three = c.mul(
        &arcs.arced,
        &c.sub(&fill(&arcs.arced, 1.0)?, &arcs.floored)?,
    )?;
    let row = c.mul(&three, &spec.active)?;
    let inside = c.mul(&c.mul(&after, &before)?, &wide(&row, KNOTS)?)?;
    let knots = c.where_cond(&inside, &values, &start)?;

    // Arc rows split each uniform middle lane into SPLIT lanes.
    let fractions = splits();
    let extra = fractions.len();
    let fraction = Tensor::<R>::from_slice(&fractions, &[1, extra], spec.start.device())?
        .broadcast_to(&[m, extra])?
        .contiguous()?;
    let from = wide(&spec.start, extra)?;
    let span = c.sub(&wide(&spec.limit, extra)?, &from)?;
    let placed = c.add(&from, &c.mul(&span, &fraction)?)?;
    let arc_rows = wide(&c.mul(&arcs.arced, &spec.active)?, extra)?;
    let placed = c.where_cond(&arc_rows, &placed, &from)?;

    let merged = c.cat(&[nodes, &knots, &placed], 1)?;
    let sign = wide(&spec.sign, count + KNOTS + extra)?;
    let sorted = c.sort(&c.mul(&sign, &merged)?, 1, false)?;
    Ok(c.mul(&sign, &sorted)?)
}

/// Lanes each uniform middle lane `1/16` of an arc profile splits into.
const SPLIT: usize = 4;

/// Fractions `j / (16 SPLIT)` inside the uniform middle lanes `[1/16, 15/16]`, off their ends.
fn splits() -> Vec<f64> {
    let parts = 16 * SPLIT;
    (SPLIT + 1..parts - SPLIT)
        .filter(|j| j % SPLIT != 0)
        .map(|j| j as f64 / parts as f64)
        .collect()
}

/// Lanes an arc batch adds to each row: the junction knots and the middle splits.
#[cfg(test)]
pub(crate) fn added() -> usize {
    KNOTS + splits().len()
}

/// Sliver lanes with the three-band parameters of their rows.
pub(super) struct Sliver<R: Runtime<DType = DType>> {
    /// 1 on lanes inside a junction angle enclosure.
    pub(super) mask: Tensor<R>,
    pub(super) rows: ProfileParams<R>,
}

/// Lane parameters `rows` with each arc lane's own band, and the sliver lanes.
pub(super) fn lane_params<R>(
    c: &R::Client,
    rows: ProfileParams<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
) -> Result<(ProfileParams<R>, Option<Sliver<R>>), StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Some(arcs) = &rows.arcs else {
        return Ok((rows, None));
    };
    let (lane, mask) = arcs.lanes(c, left, right)?;
    let params = ProfileParams {
        arcs: Some(lane),
        ..rows.clone()
    };
    Ok((params, Some(Sliver { mask, rows })))
}

/// Integral bounds of `G` over each sliver lane `[left, right]`: width times the hull of `G`.
///
/// Lanes outside `sliver.mask` report zeros.
pub(super) fn sliver_integrals<R>(
    c: &R::Client,
    sliver: &Sliver<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
) -> Result<Enclosure<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let rows = &sliver.rows;
    let mask = &sliver.mask;
    let [lower, upper, known] =
        with_interval_arithmetic(c, DType::F64, "culvert outlet junction slivers", |a| {
            let ac = a.client();
            let high = ac.where_cond(mask, right, left)?;
            let v = a.bounds(left, &high)?;
            let over = integrand(a, rows, &v, 1, None)?;
            let value = jet::coefficient(a, &over.value, 0, &rows.rise)?;
            let start = a.point(left)?;
            let end = a.point(&high)?;
            let width = a.sub(&end, &start)?;
            let piece = a.mul(&width, &value)?;
            let zero = fill(left, 0.0)?;
            let held = ac.mul(&over.valid, mask)?;
            Ok([
                ac.where_cond(&held, &ac.maximum(piece.lower(), &zero)?, &zero)?,
                ac.where_cond(&held, piece.upper(), &zero)?,
                held,
            ])
        })?;
    Ok(Enclosure {
        bounds: [lower, upper],
        known,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::numerics::culvert::RowNames;
    use crate::structures::numerics::culvert::outlet::partition::LANES;
    use crate::structures::numerics::culvert::outlet::{Accuracy, march, prepare};
    use crate::structures::tensor::culvert::TensorBarrelContext;
    use crate::structures::{CulvertBarrel, CulvertOutletRow};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    type Cpu = CpuRuntime;

    /// Frozen criteria: a pipe-arch profile from the crown down to a critical depth below the
    /// first junction holds both junction enclosures. No lane crosses an enclosure end, each
    /// enclosure is one known sliver lane, and each sliver bound holds its width times `G` at
    /// the sliver's lower end.
    #[test]
    fn a_profile_across_both_junctions_keeps_every_lane_in_one_band() -> Result<(), StructureError>
    {
        let device = CpuDevice::new();
        let client = Cpu::default_client(&device);
        let barrels = [CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?];
        let rows = [CulvertOutletRow {
            barrel: barrels[0],
            length_m: 40.0,
            manning_n: 0.013,
            entrance_loss: 0.5,
            slope: 0.05,
            gravity_m_per_s2: 9.81,
        }];
        let barrel = TensorBarrelContext::<Cpu>::new(client.clone(), &barrels)?;
        let unit = barrel.unit()?;
        let outlet = prepare(&barrel, &rows)?;
        let q = Tensor::<Cpu>::from_slice(&[0.03], &[1], &device)?;
        let tw = Tensor::<Cpu>::from_slice(&[1.3 * 1.397], &[1], &device)?;
        let found = march(
            RowNames::new(&barrels),
            &barrel,
            &unit,
            &outlet,
            &q,
            &tw,
            Accuracy::Fine,
        )?;
        let profile = &found.profile;
        assert_eq!(profile.lanes(), LANES + added());
        let arcs = outlet
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("no arc bands"))?;
        let mut enclosures = Vec::new();
        for band in &arcs.bands[1..] {
            enclosures.push([
                band.start_angle[0].try_to_vec::<f64>()?[0],
                band.start_angle[1].try_to_vec::<f64>()?[0],
            ]);
        }
        let mut nodes = profile.nodes.try_to_vec::<f64>()?;
        nodes.sort_by(f64::total_cmp);
        let known = profile.known.try_to_vec::<f64>()?;
        assert!(
            known.iter().all(|&k| k == 1.0),
            "an unknown lane: {known:?}"
        );
        let mut slivers = Vec::new();
        for pair in nodes.windows(2) {
            let [left, right] = pair else { continue };
            if right <= left {
                continue;
            }
            for &[low, high] in &enclosures {
                let crosses = (left < &low && right > &low) || (left < &high && right > &high);
                assert!(
                    !crosses,
                    "lane [{left:e}, {right:e}] crosses [{low:e}, {high:e}]"
                );
                if *left >= low && *right <= high {
                    slivers.push([*left, *right]);
                }
            }
        }
        assert_eq!(slivers.len(), 2, "slivers {slivers:?} of nodes {nodes:?}");
        for [left, right] in slivers {
            let rows = profile.spec.params.clone();
            let mask = Tensor::<Cpu>::ones(&[1], DType::F64, &device)?;
            let low = Tensor::<Cpu>::from_slice(&[left], &[1], &device)?;
            let high = Tensor::<Cpu>::from_slice(&[right], &[1], &device)?;
            let sliver = Sliver { mask, rows };
            let bound = sliver_integrals(&client, &sliver, &low, &high)?;
            let point = with_interval_arithmetic(&client, DType::F64, "sliver point", |a| {
                let v = a.point(&low)?;
                let over = integrand(a, &sliver.rows, &v, 1, None)?;
                let value = jet::coefficient(a, &over.value, 0, &sliver.rows.rise)?;
                Ok([
                    value.lower().try_to_vec::<f64>()?[0],
                    value.upper().try_to_vec::<f64>()?[0],
                ])
            })?;
            let lower = bound.bounds[0].try_to_vec::<f64>()?[0];
            let upper = bound.bounds[1].try_to_vec::<f64>()?[0];
            let width = right - left;
            assert_eq!(bound.known.try_to_vec::<f64>()?, [1.0]);
            assert!(
                lower <= width * point[1] * (1.0 + 1e-12)
                    && width * point[0] * (1.0 - 1e-12) <= upper,
                "sliver [{left:e}, {right:e}]: [{lower:e}, {upper:e}] misses {width:e} x {point:?}"
            );
        }
        Ok(())
    }
}
