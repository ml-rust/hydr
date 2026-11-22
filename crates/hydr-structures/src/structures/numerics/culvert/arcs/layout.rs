//! Band start values chained from the invert, and the full-state constants.
//!
//! The bottom band of a three-radius section is a cap of the bottom circle up to `psi_1`. The
//! corner band adds, from `psi_1` to `psi_2` with `dy = r_c (cos psi_1 - cos psi_2)`:
//!
//! - `A += 2 x_c dy + r_c^2 ((psi_2 - psi_1) - sin psi_2 cos psi_2 + sin psi_1 cos psi_1)`
//! - `P += 2 r_c (psi_2 - psi_1)`
//! - `M += A_1 dy + x_c dy^2 + r_c^3 I_2`, with
//!   `I_2 = [sin t - t cos t - sin^3 t / 3] from psi_1 to psi_2 - (psi_1 - sin psi_1 cos psi_1) dy / r_c`
//!
//! The top band is a cap of the top circle seen from the crown, of height `h_t = D - y_2`. It
//! closes the full state: `A_full = A_2 + A_cap`, `P_full = P_2 + P_cap` and
//! `M(D) = M_2 + A_2 h_t + h_t A_cap - M_cap`.
//!
//! A flat-floor arch is one cap of height `D` above the floor chord: `A_full = A_cap`,
//! `P_full = B + P_cap` and `M(D) = D A_cap - M_cap`.

use super::cap::{Cap, Shared, cap};
use super::inputs::ArcInputs;
use super::normals::{FlatArch, ThreeRadius, copy, flat_arch, nonnegative, three_radius};
use super::{ArcBand, ArcLayout, meet, pick};
use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::SeriesTensors;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Band arc and start values inside the interval call, in [`ArcBand`] field order.
struct Band<R: Runtime<DType = DType>> {
    centre_x: Interval<R>,
    centre_y: Interval<R>,
    radius: Interval<R>,
    angle: Interval<R>,
    sin: Interval<R>,
    cos: Interval<R>,
    depth: Interval<R>,
    area: Interval<R>,
    perimeter: Interval<R>,
    moment: Interval<R>,
}

/// Full-state constants inside the interval call.
struct Full<R: Runtime<DType = DType>> {
    area: Interval<R>,
    perimeter: Interval<R>,
    radius: Interval<R>,
    moment: Interval<R>,
    widest: Interval<R>,
}

/// Bottom, corner and top bands with the full state.
type ThreeBands<R> = (Band<R>, Band<R>, Band<R>, Full<R>);

/// Encloses the layout of every row. Rows other than arc rows hold a harmless barrel's layout.
pub(crate) fn layout<R>(
    a: &mut Arithmetic<'_, R>,
    inputs: &ArcInputs<R>,
) -> ArithmeticResult<ArcLayout<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let like = &inputs.arch;
    let device = like.device();
    let fill = |value: f64| Tensor::<R>::full_scalar(like.shape(), DType::F64, value, device);
    let pi = a.bounds(&fill(f64::from_bits(PI[0]))?, &fill(f64::from_bits(PI[1]))?)?;
    let half = a.point(&fill(0.5)?)?;
    let half_pi = a.mul(&pi, &half)?;
    let shared = Shared {
        zero: fill(0.0)?,
        one: fill(1.0)?,
        two: a.point(&fill(2.0)?)?,
        three: fill(3.0)?,
        half,
        pi,
        half_pi,
        series: SeriesTensors::constant(like.shape()[0], device)?,
    };
    let three = three_radius(a, inputs)?;
    let (bottom, corner, top, full) = three_radius_bands(a, &shared, &three)?;
    let arch = flat_arch(a, &inputs.arch_operands)?;
    let (arch_band, arch_full) = arch_band(a, &shared, &arch)?;
    let mask = &inputs.arch;
    let band = |three: &Band<R>| -> ArithmeticResult<ArcBand<R>> {
        Ok(ArcBand {
            centre_x: pick(a, mask, &arch_band.centre_x, &three.centre_x)?,
            centre_y: pick(a, mask, &arch_band.centre_y, &three.centre_y)?,
            radius: pick(a, mask, &arch_band.radius, &three.radius)?,
            start_angle: pick(a, mask, &arch_band.angle, &three.angle)?,
            start_sin: pick(a, mask, &arch_band.sin, &three.sin)?,
            start_cos: pick(a, mask, &arch_band.cos, &three.cos)?,
            start_depth: pick(a, mask, &arch_band.depth, &three.depth)?,
            start_area: pick(a, mask, &arch_band.area, &three.area)?,
            start_perimeter: pick(a, mask, &arch_band.perimeter, &three.perimeter)?,
            start_moment: pick(a, mask, &arch_band.moment, &three.moment)?,
        })
    };
    let bands = [band(&bottom)?, band(&corner)?, band(&top)?];
    Ok(ArcLayout {
        bands,
        full_area: pick(a, mask, &arch_full.area, &full.area)?,
        full_perimeter: pick(a, mask, &arch_full.perimeter, &full.perimeter)?,
        full_radius: pick(a, mask, &arch_full.radius, &full.radius)?,
        full_moment: pick(a, mask, &arch_full.moment, &full.moment)?,
        widest_depth: pick(a, mask, &arch_full.widest, &full.widest)?,
    })
}

/// Bottom, corner and top bands of the three-radius section, then its full state.
fn three_radius_bands<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<R>,
    t: &ThreeRadius<R>,
) -> ArithmeticResult<ThreeBands<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let zero = a.point(&shared.zero)?;
    let one = a.point(&shared.one)?;
    let bottom = cap(a, shared, &t.bottom, &t.sin1, &t.cos1)?;
    let top = cap(a, shared, &t.top, &t.sin2, &t.minus_cos2)?;
    let psi2 = a.sub(&shared.pi, &top.angle)?;

    // Junction depths, each from both arcs that meet there.
    let drop = a.mul(&t.corner, &t.cos1)?;
    let by_corner = a.sub(&t.height, &drop)?;
    let y1 = meet(a, &bottom.height, &by_corner)?;
    let lift = a.mul(&t.corner, &t.minus_cos2)?;
    let by_corner = a.add(&t.height, &lift)?;
    let by_top = a.sub(&t.rise, &top.height)?;
    let y2 = meet(a, &by_corner, &by_top)?;
    let cos_sum = a.add(&t.cos1, &t.minus_cos2)?;
    let by_arc = a.mul(&t.corner, &cos_sum)?;
    let by_depth = a.sub(&y2, &y1)?;
    let dy = meet(a, &by_arc, &by_depth)?;
    let crown_height = a.sub(&t.rise, &y2)?;
    let crown_height = meet(a, &top.height, &crown_height)?;

    // Corner band, psi_1 to psi_2. Every bracket term is nonnegative.
    let psi1 = &bottom.angle;
    let sweep = a.sub(&psi2, psi1)?;
    let end2 = a.mul(&t.sin2, &t.minus_cos2)?;
    let end1 = a.mul(&t.sin1, &t.cos1)?;
    let bracket = a.add(&sweep, &end2)?;
    let bracket = a.add(&bracket, &end1)?;
    let corner_square = a.mul(&t.corner, &t.corner)?;
    let swept = a.mul(&corner_square, &bracket)?;
    let strip = a.mul(&t.offset, &dy)?;
    let strip = a.mul(&shared.two, &strip)?;
    let area = a.add(&bottom.area, &strip)?;
    let area = a.add(&area, &swept)?;
    let arc = a.mul(&t.corner, &sweep)?;
    let arc = a.mul(&shared.two, &arc)?;
    let perimeter = a.add(&bottom.perimeter, &arc)?;

    let at2 = antiderivative(a, shared, &psi2, &t.sin2, &t.minus_cos2, true)?;
    let at1 = antiderivative(a, shared, psi1, &t.sin1, &t.cos1, false)?;
    let segment = a.sub(psi1, &end1)?;
    let tail = a.mul(&segment, &cos_sum)?;
    let i2 = a.sub(&at2, &at1)?;
    let i2 = a.sub(&i2, &tail)?;
    let corner_cube = a.mul(&corner_square, &t.corner)?;
    let swept_moment = a.mul(&corner_cube, &i2)?;
    let lifted = a.mul(&bottom.area, &dy)?;
    let dy_square = a.mul(&dy, &dy)?;
    let strip_moment = a.mul(&t.offset, &dy_square)?;
    let moment = a.add(&bottom.moment, &lifted)?;
    let moment = a.add(&moment, &strip_moment)?;
    let moment = a.add(&moment, &swept_moment)?;

    let full = close(a, &area, &perimeter, &moment, &top, &crown_height)?;
    let full = Full {
        widest: copy(a, &t.height)?,
        ..full
    };
    let corner_band = Band {
        centre_x: copy(a, &t.offset)?,
        centre_y: copy(a, &t.height)?,
        radius: copy(a, &t.corner)?,
        angle: copy(a, psi1)?,
        sin: copy(a, &t.sin1)?,
        cos: copy(a, &t.cos1)?,
        depth: copy(a, &y1)?,
        area: copy(a, &bottom.area)?,
        perimeter: copy(a, &bottom.perimeter)?,
        moment: copy(a, &bottom.moment)?,
    };
    let top_cos = a.sub(&zero, &t.minus_cos2)?;
    let top_centre = a.sub(&t.rise, &t.top)?;
    let top_band = Band {
        centre_x: copy(a, &zero)?,
        centre_y: top_centre,
        radius: copy(a, &t.top)?,
        angle: psi2,
        sin: copy(a, &t.sin2)?,
        cos: top_cos,
        depth: y2,
        area,
        perimeter,
        moment,
    };
    let bottom_band = Band {
        centre_x: copy(a, &zero)?,
        centre_y: copy(a, &t.bottom)?,
        radius: copy(a, &t.bottom)?,
        angle: copy(a, &zero)?,
        sin: copy(a, &zero)?,
        cos: one,
        depth: copy(a, &zero)?,
        area: copy(a, &zero)?,
        perimeter: copy(a, &zero)?,
        moment: zero,
    };
    Ok((bottom_band, corner_band, top_band, full))
}

/// `sin t - t cos t - sin^3 t / 3` from the sine, the cosine (negated when `negated`) and `t`.
fn antiderivative<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<R>,
    angle: &Interval<R>,
    sin: &Interval<R>,
    cos: &Interval<R>,
    negated: bool,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let angle_cos = a.mul(angle, cos)?;
    let square = a.mul(sin, sin)?;
    let cube = a.mul(&square, sin)?;
    let third = a.div_positive(&cube, &shared.three)?;
    let linear = if negated {
        a.add(sin, &angle_cos)?
    } else {
        a.sub(sin, &angle_cos)?
    };
    a.sub(&linear, &third)
}

/// Full state from the start values of the band below the cap and the cap of height `height`.
fn close<R>(
    a: &mut Arithmetic<'_, R>,
    area: &Interval<R>,
    perimeter: &Interval<R>,
    moment: &Interval<R>,
    crown: &Cap<R>,
    height: &Interval<R>,
) -> ArithmeticResult<Full<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let full_area = a.add(area, &crown.area)?;
    let full_perimeter = a.add(perimeter, &crown.perimeter)?;
    let lifted = a.mul(area, height)?;
    let stacked = a.mul(height, &crown.area)?;
    let first = a.sub(&stacked, &crown.moment)?;
    let full_moment = a.add(moment, &lifted)?;
    let full_moment = a.add(&full_moment, &first)?;
    let radius = a.div_positive_interval(&full_area, &full_perimeter)?;
    Ok(Full {
        area: full_area,
        perimeter: full_perimeter,
        radius,
        moment: full_moment,
        widest: copy(a, height)?,
    })
}

/// The band every arch row repeats three times, and the arch full state.
fn arch_band<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<R>,
    f: &FlatArch<R>,
) -> ArithmeticResult<(Band<R>, Full<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let zero = a.point(&shared.zero)?;
    let arc = cap(a, shared, &f.radius, &f.sin0, &f.cap_cos)?;
    let psi0 = a.sub(&shared.pi, &arc.angle)?;
    let cos0 = a.sub(&zero, &f.cap_cos)?;
    let centre = a.sub(&f.rise, &f.radius)?;
    let widest = nonnegative(a, &centre)?;
    // The cap above the floor has height D, so its stacked moment uses D exactly.
    let full = close(a, &zero, &f.span, &zero, &arc, &f.rise)?;
    let full = Full { widest, ..full };
    let band = Band {
        centre_x: copy(a, &zero)?,
        centre_y: centre,
        radius: copy(a, &f.radius)?,
        angle: psi0,
        sin: copy(a, &f.sin0)?,
        cos: cos0,
        depth: copy(a, &zero)?,
        area: copy(a, &zero)?,
        perimeter: copy(a, &f.span)?,
        moment: zero,
    };
    Ok((band, full))
}

#[cfg(test)]
mod tests {
    use super::super::layout_rows::{BARRELS, Barrel, LAYOUT_SHA256, Shape, VALUES};
    use super::*;
    use crate::structures::tensor::culvert::Bounds;
    use crate::structures::{CulvertBarrel, StructureError};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Cpu = CpuRuntime;
    type Layouts = (Vec<CulvertBarrel>, ArcLayout<Cpu>);

    fn barrel(entry: &Barrel) -> Result<CulvertBarrel, StructureError> {
        let inputs: Vec<f64> = entry.inputs.iter().map(|b| f64::from_bits(*b)).collect();
        match (entry.shape, inputs.as_slice()) {
            (Shape::HorizontalEllipse, &[b, d, r]) => CulvertBarrel::horizontal_ellipse(b, d, r),
            (Shape::VerticalEllipse, &[b, d, r]) => CulvertBarrel::vertical_ellipse(b, d, r),
            (Shape::PipeArch, &[b, d, c, r]) => CulvertBarrel::pipe_arch(b, d, c, r),
            (Shape::Arch, &[b, d]) => CulvertBarrel::arch(b, d),
            _ => Err(StructureError::definition(format!(
                "fixture barrel {}: wrong input count",
                entry.name
            ))),
        }
    }

    /// Every evidence barrel, then a circle and a box, with their layouts.
    fn layouts() -> Result<Layouts, StructureError> {
        let device = CpuDevice::new();
        let client = Cpu::default_client(&device);
        let mut barrels = BARRELS.iter().map(barrel).collect::<Result<Vec<_>, _>>()?;
        barrels.push(CulvertBarrel::circular(1.0)?);
        barrels.push(CulvertBarrel::rectangular_box(2.0, 1.0)?);
        let inputs = ArcInputs::<Cpu>::new(&barrels, &device)?;
        let found =
            with_interval_arithmetic(&client, DType::F64, "arc layout", |a| layout(a, &inputs))?;
        Ok((barrels, found))
    }

    fn named<'a>(layout: &'a ArcLayout<Cpu>, name: &str) -> Option<&'a Bounds<Cpu>> {
        let Some((band, field)) = name.split_once('.') else {
            return match name {
                "full_area" => Some(&layout.full_area),
                "full_perimeter" => Some(&layout.full_perimeter),
                "full_radius" => Some(&layout.full_radius),
                "full_moment" => Some(&layout.full_moment),
                "widest_depth" => Some(&layout.widest_depth),
                _ => None,
            };
        };
        let index = ["bottom", "corner", "top"]
            .iter()
            .position(|n| *n == band)?;
        let band = layout.bands.get(index)?;
        Some(match field {
            "centre_x" => &band.centre_x,
            "centre_y" => &band.centre_y,
            "radius" => &band.radius,
            "start_angle" => &band.start_angle,
            "start_sin" => &band.start_sin,
            "start_cos" => &band.start_cos,
            "start_depth" => &band.start_depth,
            "start_area" => &band.start_area,
            "start_perimeter" => &band.start_perimeter,
            "start_moment" => &band.start_moment,
            _ => return None,
        })
    }

    #[test]
    fn layout_holds_the_frozen_references() -> Result<(), StructureError> {
        assert_eq!(
            LAYOUT_SHA256,
            "c4a952e33384cb545f04d9a41358e381c1fa190245b4be8e61b92ddf101501b4"
        );
        let (barrels, found) = layouts()?;
        let mut failures = Vec::new();
        for value in VALUES {
            let Some([lower, upper]) = named(&found, value.name) else {
                failures.push(format!("unknown layout value {}", value.name));
                continue;
            };
            let (lower, upper) = (lower.try_to_vec::<f64>()?, upper.try_to_vec::<f64>()?);
            let row = value.barrel;
            let (Some(&lo), Some(&hi), Some(entry)) =
                (lower.get(row), upper.get(row), barrels.get(row))
            else {
                failures.push(format!("row {row} missing for {}", value.name));
                continue;
            };
            let [down, up] = value.bounds.map(f64::from_bits);
            let scale = down
                .abs()
                .max(up.abs())
                .max(entry.rise_m().powi(value.power));
            let allowed = value.relative_width * scale;
            if !(lo <= down && up <= hi && hi - lo <= allowed) {
                failures.push(format!(
                    "{} {}: [{lo:e}, {hi:e}] against [{down:e}, {up:e}], width allowance {allowed:e}",
                    BARRELS.get(row).map_or("?", |b| b.name),
                    value.name
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }

    #[test]
    fn junctions_join_tangent_arcs() -> Result<(), StructureError> {
        let (barrels, found) = layouts()?;
        let device = CpuDevice::new();
        let client = Cpu::default_client(&device);
        // Each entry must hold zero: sin^2 + cos^2 - 1 per band, then the x and y gaps between
        // the two arcs at each junction and the gap from the junction point to the start depth.
        let gaps = with_interval_arithmetic(&client, DType::F64, "arc junctions", |a| {
            let enter =
                |a: &mut Arithmetic<'_, Cpu>, [lower, upper]: &Bounds<Cpu>| a.bounds(lower, upper);
            let one = a.point(&Tensor::<Cpu>::ones(&[barrels.len()], DType::F64, &device)?)?;
            let mut gaps = Vec::new();
            for band in &found.bands {
                let sin = enter(a, &band.start_sin)?;
                let cos = enter(a, &band.start_cos)?;
                let sin_square = a.mul(&sin, &sin)?;
                let cos_square = a.mul(&cos, &cos)?;
                let sum = a.add(&sin_square, &cos_square)?;
                gaps.push(a.sub(&sum, &one)?);
            }
            for pair in found.bands.windows(2) {
                let [below, above] = pair else { continue };
                let sin = enter(a, &above.start_sin)?;
                let cos = enter(a, &above.start_cos)?;
                let mut points = Vec::new();
                for band in [below, above] {
                    let radius = enter(a, &band.radius)?;
                    let across = a.mul(&radius, &sin)?;
                    let x = enter(a, &band.centre_x)?;
                    let down = a.mul(&radius, &cos)?;
                    let y = enter(a, &band.centre_y)?;
                    points.push((a.add(&x, &across)?, a.sub(&y, &down)?));
                }
                let [(x0, y0), (x1, y1)] = points.as_slice() else {
                    continue;
                };
                gaps.push(a.sub(x0, x1)?);
                gaps.push(a.sub(y0, y1)?);
                let depth = enter(a, &above.start_depth)?;
                gaps.push(a.sub(y1, &depth)?);
            }
            Ok(gaps)
        })?;
        for (k, gap) in gaps.iter().enumerate() {
            let lower = gap.lower().try_to_vec::<f64>()?;
            let upper = gap.upper().try_to_vec::<f64>()?;
            for (row, (lo, hi)) in lower.iter().zip(&upper).enumerate().take(BARRELS.len()) {
                assert!(
                    *lo <= 0.0 && 0.0 <= *hi,
                    "gap {k} row {row}: [{lo:e}, {hi:e}]"
                );
            }
        }
        Ok(())
    }
}
