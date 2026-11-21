//! Hydraulic jump of a steep barrel whose inlet runs at critical depth.
//!
//! The supercritical profile S2 starts at `yc` at the inlet and falls toward `yn`. Along any
//! profile the momentum `M = Q^2 / (g A) + m` changes by `dM/dx = A (S - Sf)`. Below the
//! conveyance peak, `A (S - Sf)` rises with depth, so `dM_sub/dx > dM_sup/dx` and the momentum
//! gap `M_sub(x) - M_sup(x)` rises along the barrel. A jump sits inside the barrel when the gap at
//! the outlet, `M(TW) - M(y_S2(L))`, is positive, and its position is the unique zero of the gap.
//!
//! The gap is written in the subcritical depth `y2`, so it rises with `y2`. Each evaluation takes
//! the S1 distance `x(y2)` by Simpson and the S2 depth at `x` by certified bisection. A scan of the
//! S1 nodes brackets the zero, and solvr certified bisection encloses it with the slope floor
//! `[phi(y2_low) - phi(yc)] G1_min`, `phi = A (S - Sf)`.

mod gap;
mod scan;
mod steep;

use super::March;
use super::coordinates::depth;
use super::depth::depth_at;
use super::levels::Accuracy;
use super::partition::{LANES, build};
use super::trig::fill;
use crate::structures::StructureError;
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use gap::{Solve, gap, position, upstream_distance};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ShapeOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use scan::{ScanInputs, scan};
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootProbe,
    CertifiedRootStatus, CertifiedRootValues,
};
use std::cell::Cell;
use steep::{owners, pair, steep};

/// Probe rounds allowed for the jump solve.
const MAX_ITERATIONS: usize = 256;

/// Relative width at which the jump enclosure stops contracting.
const RELATIVE_TOLERANCE: f64 = 1.0 / 1_125_899_906_842_624.0;

/// Jump status codes with position and depth bounds in metres. Rows without a value hold NaN.
pub(crate) struct OutletJump<R: Runtime<DType = DType>> {
    pub(crate) status: Tensor<R>,
    pub(crate) position: Bounds<R>,
    pub(crate) upstream: Bounds<R>,
    pub(crate) downstream: Bounds<R>,
    pub(crate) outlet: Bounds<R>,
}

/// Classifies every inlet-critical row and encloses the jump position and depths.
pub(crate) fn jump<R>(
    barrel: &TensorBarrelContext<R>,
    outlet: &TensorOutletContext<R>,
    march: &March<R>,
    discharge: &Tensor<R>,
    tailwater: &Tensor<R>,
) -> Result<OutletJump<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R> + CertifiedSimpsonAlgorithms<R>,
{
    let c = &barrel.client;
    let m = barrel.width();
    let count = LANES + 1;
    let per_row = count + 1;
    let steep = with_interval_arithmetic(c, DType::F64, "culvert outlet jump profile", |a| {
        steep(a, barrel, outlet, march, discharge)
    })?;
    let s2 = build(c, steep.spec.clone(), Accuracy::Fine)?;

    // Targets: the outlet, then L - U1 at every S1 node.
    let profile = &march.profile;
    let targets = with_interval_arithmetic(c, DType::F64, "culvert outlet jump targets", |a| {
        let wide = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
            value
                .reshape(&[m, 1])?
                .broadcast_to(&[m, count])?
                .contiguous()
        };
        let l = a.point(&wide(&outlet.length)?)?;
        let u = a.bounds(&profile.reach[0], &profile.reach[1])?;
        let x = a.sub(&l, &u)?;
        let length = outlet.length.reshape(&[m, 1])?;
        let lower = a
            .client()
            .cat(&[&length, x.lower()], 1)?
            .reshape(&[m * per_row])?;
        let upper = a
            .client()
            .cat(&[&length, x.upper()], 1)?
            .reshape(&[m * per_row])?;
        Ok([lower, upper])
    })?;
    let wide = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value
            .reshape(&[m, 1])?
            .broadcast_to(&[m, per_row])?
            .contiguous()
    };
    let first: Vec<f64> = (0..per_row).map(|k| f64::from(u8::from(k == 0))).collect();
    let first = Tensor::<R>::from_slice(&first, &[1, per_row], &barrel.device)?
        .broadcast_to(&[m, per_row])?
        .contiguous()?;
    let outlet_rows = wide(&steep.supercritical)?;
    let node_rows = wide(&steep.candidates)?;
    let wanted = c
        .where_cond(&first, &outlet_rows, &node_rows)?
        .reshape(&[m * per_row])?;
    let rows = owners::<R>(&barrel.device, m, per_row)?;
    let rtol = Accuracy::Fine.depth_tolerance();
    let found = depth_at(c, &s2, &rows, &targets, &wanted, rtol)?;
    let inputs = ScanInputs {
        barrel,
        outlet,
        march,
        steep: &steep,
        tailwater,
        found: &found,
    };
    let bracket = scan(c, &inputs)?;

    let solve = Solve {
        barrel,
        outlet,
        march,
        steep: &steep,
        s2: &s2,
        bracket: &bracket,
    };
    // Rows outside the solver get the single point v_high, so one endpoint settles them.
    let lower = c.where_cond(&bracket.bisect, &bracket.low, &bracket.high)?;
    let upper = bracket.high.clone();
    let options = CertifiedBisectionOptions {
        atol: 0.0,
        rtol: RELATIVE_TOLERANCE,
        max_iterations: MAX_ITERATIONS,
        max_points: m.saturating_mul(2 * (MAX_ITERATIONS + 1)),
        probe: CertifiedRootProbe::SlackSecant,
    };
    let cause = Cell::new(None);
    let solved = c.bisect_certified(
        |query| match gap(c, &solve, &query.owner, &query.point) {
            Ok([lower, upper]) => Ok(CertifiedRootValues { lower, upper }),
            Err(error) => {
                let reason = error.to_string();
                cause.set(Some(error));
                Err(numr::error::Error::InvalidArgument {
                    arg: "culvert outlet jump gap",
                    reason,
                })
            }
        },
        &lower,
        &upper,
        &bracket.slope,
        &options,
    );
    let roots = match (solved, cause.take()) {
        (_, Some(error)) => return Err(error),
        (Ok(roots), None) => roots,
        (Err(error), None) => return Err(error.into()),
    };
    let status = c.cast(roots.status(), DType::F64)?;
    let code = |value: CertifiedRootStatus| fill(&status, value as i64 as f64);
    let accepted = c.maximum(
        &c.eq(&status, &code(CertifiedRootStatus::Converged)?)?,
        &c.eq(&status, &code(CertifiedRootStatus::AccuracyFloor)?)?,
    )?;
    let owner = roots.root_owner();
    let (mut v_low, mut v_high) = (lower.clone(), upper.clone());
    if owner.numel() != 0 {
        v_low = c.index_put(&v_low, 0, owner, roots.root_lower())?;
        v_high = c.index_put(&v_high, 0, owner, roots.root_upper())?;
    }
    let good = c.mul(&bracket.bisect, &accepted)?;
    let v_low = c.where_cond(&bracket.direct, &bracket.low, &v_low)?;
    let v_high = c.where_cond(&bracket.direct, &bracket.high, &v_high)?;
    let located = c.maximum(&good, &bracket.direct)?;

    // Position from both ends of the S1 enclosure, then the S2 depth over that position.
    let all = owners::<R>(&barrel.device, m, 1)?;
    let at_low = upstream_distance(c, profile, &bracket, &all, &v_low)?;
    let at_high = upstream_distance(c, profile, &bracket, &all, &v_high)?;
    let near = position(c, &outlet.length, &at_low)?;
    let far = position(c, &outlet.length, &at_high)?;
    let place = [near[0].clone(), far[1].clone()];
    let place_ok = c.mul(&located, &c.le(&place[0], &place[1])?)?;
    let safe = [
        c.where_cond(&place_ok, &place[0], &outlet.length)?,
        c.where_cond(&place_ok, &place[1], &outlet.length)?,
    ];
    let upstream = depth_at(c, &s2, &all, &safe, &place_ok, rtol)?;
    let depths = with_interval_arithmetic(c, DType::F64, "culvert outlet jump depths", |a| {
        let ok = c.mul(&place_ok, &upstream.solved)?;
        let half = &fill(&barrel.one, 0.5)?;
        let v1 = a.bounds(
            &c.where_cond(&ok, &upstream.lower, half)?,
            &c.where_cond(&ok, &upstream.upper, half)?,
        )?;
        let y1 = depth(a, &barrel.circular, &barrel.rise, &v1)?;
        let v2 = a.bounds(
            &c.where_cond(&ok, &v_low, half)?,
            &c.where_cond(&ok, &v_high, half)?,
        )?;
        let y2 = depth(a, &barrel.circular, &barrel.rise, &v2)?;
        Ok((ok, pair(&y1), pair(&y2)))
    })?;
    let (jumped, upstream_depth, downstream_depth) = depths;
    let jumped = c.mul(&jumped, &bracket.inside)?;

    let one = &barrel.one;
    let critical = c.eq(
        &march.headwater.status,
        &fill(one, codes::OUTLET_INLET_CRITICAL)?,
    )?;
    let unresolved = c.eq(
        &march.headwater.status,
        &fill(one, codes::OUTLET_UNRESOLVED)?,
    )?;
    let outside = c.mul(&bracket.outside, &c.sub(one, &bracket.inside)?)?;
    let code = |value: f64| fill(one, value);
    let mut status = code(codes::JUMP_NOT_STEEP)?;
    status = c.where_cond(&critical, &code(codes::JUMP_UNRESOLVED)?, &status)?;
    status = c.where_cond(&unresolved, &code(codes::JUMP_UNRESOLVED)?, &status)?;
    status = c.where_cond(&outside, &code(codes::JUMP_SUPERCRITICAL)?, &status)?;
    status = c.where_cond(&jumped, &code(codes::JUMP_IN_BARREL)?, &status)?;
    let missing = code(f64::NAN)?;
    let keep = |mask: &Tensor<R>, value: &Tensor<R>| c.where_cond(mask, value, &missing);
    Ok(OutletJump {
        status,
        position: [keep(&jumped, &place[0])?, keep(&jumped, &place[1])?],
        upstream: [
            keep(&jumped, &upstream_depth[0])?,
            keep(&jumped, &upstream_depth[1])?,
        ],
        downstream: [
            keep(&jumped, &downstream_depth[0])?,
            keep(&jumped, &downstream_depth[1])?,
        ],
        outlet: [
            keep(&outside, &bracket.outlet[0])?,
            keep(&outside, &bracket.outlet[1])?,
        ],
    })
}
