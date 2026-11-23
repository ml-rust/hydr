//! Bounds of `d r_c / ds` and `d r_n / ds` over depth pieces of open sections above a floor.
//!
//! Depths run up from the bed in each row's frame, with `y = D' s` and `D'` the open rise. On a
//! wall arc with centre height `y_k` above the bed and radius `r`, with `h = r sin psi`:
//!
//! - `d r_c / ds = D' (3 T / A + 2 (y - y_k) / (h T))`
//! - `d r_n / ds = D' (5 T / A - 4 r / (h P_n))`
//!
//! The bed adds the constant `rho^(3/2) b` to `P_n`, so `dP_n/dy = dP_w/dy = 2 r / h` and the
//! wall terms are those of the barrel. `A`, `T` and `P_n` come from range geometry over the
//! whole piece. Each depth of a piece lies in the closed depth range of one of the three bands
//! from the bed, so the least lower and greatest upper bound of the band parts hold for the piece.

use super::slopes::{Flow, Part, PieceSlopes, guard, part};
use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, floor};
use crate::structures::tensor::culvert::{Bounds, FloorRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// The open sections of `unit`, which must hold them.
fn rows<R: Runtime<DType = DType>>(
    unit: &TensorBarrelContext<R>,
) -> ArithmeticResult<&FloorRows<R>> {
    unit.floor.as_deref().ok_or_else(|| {
        numr::error::Error::InvalidArgument {
            arg: "unit",
            reason: "floor certificates need floor rows".into(),
        }
        .into()
    })
}

/// A lower bound of each row's widest depth in `s = y / D'`. Zero when the bed lies above it.
pub(super) fn widest<R>(
    a: &mut Arithmetic<'_, R>,
    unit: &TensorBarrelContext<R>,
) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let floor = rows(unit)?;
    let depth = a.point(&floor.widest_depth[0])?;
    let fraction = a.div_positive(&depth, &floor.top)?;
    Ok(fraction.lower().clone())
}

/// `A` and `P_n` over the fractions `[lower, upper]`, in each row's frame.
pub(super) fn conveyance<R>(
    a: &mut Arithmetic<'_, R>,
    unit: &TensorBarrelContext<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<[Interval<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let floor = rows(unit)?;
    let [low, high] = floor::frame_depths(a, floor, lower, upper)?;
    let geometry = floor::over_depth(a, unit, floor, &low, &high)?.geometry;
    Ok([geometry.area, geometry.weighted])
}

/// Encloses both slopes over the pieces `[lower, upper]` in `s` of the open sections of `unit`.
///
/// Each piece lies in `[2^-30, 1 - 2^-52]`, so its frame depths stay below the crown.
pub(super) fn over_pieces<R>(
    a: &mut Arithmetic<'_, R>,
    unit: &TensorBarrelContext<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<PieceSlopes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let floor = rows(unit)?;
    let [low, high] = floor::frame_depths(a, floor, lower, upper)?;
    let geometry = floor::over_depth(a, unit, floor, &low, &high)?.geometry;
    let zero = &unit.zero;
    let positive = c.mul(
        &c.gt(geometry.area.lower(), zero)?,
        &c.gt(geometry.top_width.lower(), zero)?,
    )?;
    let positive = c.mul(&positive, &c.gt(geometry.weighted.lower(), zero)?)?;
    let one = a.point(&unit.one)?;
    let area = guard(a, &positive, &geometry.area, &one)?;
    let width = guard(a, &positive, &geometry.top_width, &one)?;
    let weighted = guard(a, &positive, &geometry.weighted, &one)?;
    let flow = Flow {
        area: &area,
        width: &width,
        perimeter: &weighted,
    };
    let scale = a.point(&floor.top)?;

    let [bed, corner, crown] = &floor.bands;
    let [bed_middle, corner_middle, crown_middle] = &floor.middles;
    let [corner_low, corner_high] = &corner.start_depth;
    let [crown_low, crown_high] = &crown.start_depth;
    // Each band serves the depths of its closed range, as point geometry selects them.
    let on_bed = c.le(&low, corner_high)?;
    let bed_ends = [low.clone(), c.minimum(&high, corner_high)?];
    let on_corner = c.mul(&c.ge(&high, corner_low)?, &c.le(&low, crown_high)?)?;
    let corner_ends = [c.maximum(&low, corner_low)?, c.minimum(&high, crown_high)?];
    let on_crown = c.ge(&high, crown_low)?;
    let crown_ends = [c.maximum(&low, crown_low)?, high.clone()];
    let mut band_part =
        |band: &ArcBand<R>, middle: &Tensor<R>, applies: &Tensor<R>, [from, to]: &Bounds<R>| {
            let from = c.where_cond(applies, from, middle)?;
            let to = c.where_cond(applies, to, middle)?;
            part(a, band, &flow, Some(&scale), applies, [&from, &to])
        };
    let parts: [Part<R>; 3] = [
        band_part(bed, bed_middle, &on_bed, &bed_ends)?,
        band_part(corner, corner_middle, &on_corner, &corner_ends)?,
        band_part(crown, crown_middle, &on_crown, &crown_ends)?,
    ];

    // A part that does not apply holds +inf lower and -inf upper bounds.
    let [first, second, third] = parts;
    let (critical, normal_lower, normal_upper, known) = first;
    let mut slopes = PieceSlopes {
        critical_lower: critical,
        normal_lower,
        normal_upper,
        known: c.mul(&known, &positive)?,
    };
    for (critical, normal_lower, normal_upper, known) in [second, third] {
        slopes = PieceSlopes {
            critical_lower: c.minimum(&slopes.critical_lower, &critical)?,
            normal_lower: c.minimum(&slopes.normal_lower, &normal_lower)?,
            normal_upper: c.maximum(&slopes.normal_upper, &normal_upper)?,
            known: c.mul(&slopes.known, &known)?,
        };
    }
    Ok(slopes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{CulvertBarrel, CulvertFloor, StructureError};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    #[test]
    fn pieces_at_both_domain_ends_stay_below_the_crown() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        // An open rise of 0.72 m, whose top fraction times D' rounds to D' itself.
        let barrel = CulvertBarrel::circular(1.2)?.with_floor(CulvertFloor {
            sediment_depth_m: 0.48,
            roughness_ratio: 2.5,
        })?;
        let top = 1.0 - f64::EPSILON;
        let lower = [2f64.powi(-30), 0.5, 0.99, 0.999_999, top];
        let upper = [1e-3, 0.6, top, top, top];
        let unit = TensorBarrelContext::<CpuRuntime>::rows(client.clone(), &[barrel; 5], None)?;
        let lower = Tensor::<CpuRuntime>::from_slice(&lower, &[5], &device)?;
        let upper = Tensor::<CpuRuntime>::from_slice(&upper, &[5], &device)?;
        let slopes = with_interval_arithmetic(&client, DType::F64, "floor pieces", |a| {
            let s = over_pieces(a, &unit, &lower, &upper)?;
            Ok([s.critical_lower, s.normal_lower, s.normal_upper, s.known])
        })?;
        let [critical, normal_lower, normal_upper, known] = slopes;
        let read = |t: Tensor<CpuRuntime>| t.try_to_vec::<f64>();
        let (critical, normal_lower) = (read(critical)?, read(normal_lower)?);
        let (normal_upper, known) = (read(normal_upper)?, read(known)?);
        assert_eq!(known, [1.0; 5]);
        assert!(critical.iter().all(|&k| k > 0.0), "{critical:?}");
        // The conveyance rises from the bed and falls under the crown.
        assert!(
            normal_lower.first().is_some_and(|&k| k > 3.0),
            "{normal_lower:?}"
        );
        assert!(
            normal_upper
                .get(2..)
                .is_some_and(|tail| tail.iter().all(|&k| k < 0.0)),
            "{normal_upper:?}"
        );
        Ok(())
    }
}
