//! Construction-time depth certificates of the arc rows of a barrel context.
//!
//! The certificates read unit-rise geometry. Each arc barrel is first scaled by a power of two
//! to a rise in `[1, 2)`, so the scaled barrel is the same shape and no value underflows. Its
//! layout divided by the scaled rise gives the unit-rise rows.
//!
//! A row fails when the scaled barrel overflows its admission checks, or when its cover finds
//! no certified single conveyance peak below the crown. The context then rejects, naming the
//! first failing row. One scalar host read counts the failing rows. The failing mask reaches
//! the host only on rejection.

use super::context::{Bounds, TensorBarrelContext};
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::numerics::culvert::certify::{self, ArcCertificates};
use crate::structures::numerics::culvert::constants::CIRCLE_CRITICAL_TOP;
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::ops::{ConditionalOps, IndexingOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

/// The barrel scaled by the power of two that takes its rise into `[1, 2)`.
///
/// Circles and boxes stay as they are. Rejects, naming `row`, an arc barrel whose scaled
/// dimensions fail admission.
pub(super) fn unit_barrel(
    row: usize,
    barrel: &CulvertBarrel,
) -> Result<CulvertBarrel, StructureError> {
    let shape = barrel.shape();
    if !shape.arced() {
        return Ok(*barrel);
    }
    // An admitted rise is normal, so its biased exponent gives the scale exactly.
    let exponent = ((barrel.rise_m().to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let scale = 2f64.powi(-exponent);
    let span = barrel.span_m() * scale;
    let rise = barrel.rise_m() * scale;
    let given: Vec<f64> = barrel
        .given_radius_m()
        .unwrap_or(&[])
        .iter()
        .map(|radius| radius * scale)
        .collect();
    let scaled = match (shape, given.as_slice()) {
        (CulvertBarrelShape::HorizontalEllipse, &[side]) => {
            CulvertBarrel::horizontal_ellipse(span, rise, side)
        }
        (CulvertBarrelShape::VerticalEllipse, &[end]) => {
            CulvertBarrel::vertical_ellipse(span, rise, end)
        }
        (CulvertBarrelShape::PipeArch, &[corner, bottom]) => {
            CulvertBarrel::pipe_arch(span, rise, corner, bottom)
        }
        (CulvertBarrelShape::Arch, &[]) => CulvertBarrel::arch(span, rise),
        _ => Err(StructureError::definition(
            "an arc barrel lost its given radii",
        )),
    };
    scaled.map_err(|error| {
        StructureError::definition(format!(
            "shape={shape:?},span_m={:?},rise_m={:?}: the depth certificates scale the barrel to a unit rise, and the scaled barrel fails admission ({error}). Use dimensions closer in size",
            barrel.span_m(),
            barrel.rise_m()
        ))
        .at(row)
    })
}

impl<R: Runtime<DType = DType>> TensorBarrelContext<R> {
    /// Certifies the arc rows and writes their depth domains, slope floors and peak bounds.
    ///
    /// Each distinct arc barrel is certified once and its certificate copied to its other rows.
    /// Rejects, naming the first arc row without a certificate.
    pub(super) fn certify_arcs(
        mut self,
        barrels: &[CulvertBarrel],
        rounds: usize,
    ) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        // The first row of each distinct arc barrel, and each arc row's position in that list.
        let mut owners: Vec<usize> = Vec::new();
        let mut arc_rows = Vec::new();
        let mut copies = Vec::new();
        for (row, culvert) in barrels.iter().enumerate() {
            if !culvert.shape().arced() {
                continue;
            }
            let found = owners
                .iter()
                .position(|&owner| barrels.get(owner) == Some(culvert));
            let position = found.unwrap_or(owners.len());
            if found.is_none() {
                owners.push(row);
            }
            arc_rows.push(row as i64);
            copies.push(position as i64);
        }
        if owners.is_empty() {
            return Ok(self);
        }
        let indices: Vec<i64> = owners.iter().map(|&row| row as i64).collect();
        let device = &self.device;
        let index = Tensor::<R>::from_slice(&indices, &[indices.len()], device)?;
        let targets = Tensor::<R>::from_slice(&arc_rows, &[arc_rows.len()], device)?;
        let copies = Tensor::<R>::from_slice(&copies, &[copies.len()], device)?;
        let unit = self.unit()?.select(&index)?;
        let Some(rows) = &unit.arcs else {
            return Err(StructureError::numerical(
                "the unit-rise context lost its arc rows: report this runtime",
            ));
        };
        let certificates = with_interval_arithmetic(
            &self.client,
            DType::F64,
            "culvert arc barrel certificates",
            |a| certify::certify(a, &unit, rows, rounds),
        )?;
        let c = &self.client;
        let failed = c.sum(&certificates.failed, &[0], false)?.item::<f64>()?;
        if failed != 0.0 {
            return Err(rejection(barrels, &owners, &certificates));
        }
        let ArcCertificates {
            critical_slope,
            normal_top,
            normal_slope,
            peak,
            log_peak,
            ..
        } = certificates;
        let put = |base: &Tensor<R>, value: &Tensor<R>| {
            c.index_put(base, 0, &targets, &c.index_select(value, 0, &copies)?)
        };
        let pair =
            |[lower, upper]: &Bounds<R>, [l, u]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
                Ok([put(lower, l)?, put(upper, u)?])
            };
        let arced = put(&self.zero, &unit.one)?;
        self.critical_top = c.where_cond(
            &arced,
            &self.constant(CIRCLE_CRITICAL_TOP)?,
            &self.critical_top,
        )?;
        self.critical_slope = put(&self.critical_slope, &critical_slope)?;
        self.normal_top = put(&self.normal_top, &normal_top)?;
        self.normal_slope = put(&self.normal_slope, &normal_slope)?;
        self.conveyance_peak = pair(&self.conveyance_peak, &peak)?;
        self.log_peak = pair(&self.log_peak, &log_peak)?;
        Ok(self)
    }
}

/// Names the first failing arc row. Reads the failing mask to the host.
fn rejection<R: Runtime<DType = DType>>(
    barrels: &[CulvertBarrel],
    owners: &[usize],
    certificates: &ArcCertificates<R>,
) -> StructureError {
    let failed = match certificates.failed.try_to_vec::<f64>() {
        Ok(failed) => failed,
        Err(error) => return error.into(),
    };
    let first = failed.iter().position(|&flag| flag != 0.0);
    match first.and_then(|k| owners.get(k).map(|&row| (row, barrels.get(row)))) {
        Some((row, Some(barrel))) => StructureError::definition(format!(
            "shape={:?},span_m={:?},rise_m={:?},given_radius_m={:?}: conveyance has no certified single peak below the crown. Change the dimensions or report this barrel",
            barrel.shape(),
            barrel.span_m(),
            barrel.rise_m(),
            barrel.given_radius_m().unwrap_or(&[]),
        ))
        .at(row),
        _ => StructureError::numerical(
            "the device certificate count found a failing arc row that the host copy does not show: report this runtime",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;
    use crate::structures::numerics::culvert::certify::REFINEMENT_ROUNDS;
    use crate::structures::numerics::culvert::constants::{CONVEYANCE_PEAK, LOG_PEAK_CONVEYANCE};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    type Cpu = CpuRuntime;

    fn client() -> <Cpu as Runtime>::Client {
        Cpu::default_client(&CpuDevice::new())
    }

    #[test]
    fn unrefined_cover_rejects_naming_the_row() -> Result<(), StructureError> {
        let barrels = [
            CulvertBarrel::circular(1.0)?,
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
        ];
        let error = TensorBarrelContext::<Cpu>::build(client(), &barrels, 0).err();
        assert!(matches!(
            error,
            Some(StructureError::InvalidInput { element: Some(1), kind: StructureInputErrorKind::Definition, ref context })
                if context == "shape=PipeArch,span_m=1.8542,rise_m=1.397,given_radius_m=[0.4572, 1.938528]: conveyance has no certified single peak below the crown. Change the dimensions or report this barrel"
        ));
        assert!(TensorBarrelContext::<Cpu>::build(client(), &barrels, REFINEMENT_ROUNDS).is_ok());
        Ok(())
    }

    #[test]
    fn circle_and_box_rows_hold_the_frozen_peak_bits() -> Result<(), StructureError> {
        let barrels = [
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
            CulvertBarrel::arch(3.048, 1.6002)?,
        ];
        let context = TensorBarrelContext::<Cpu>::new(client(), &barrels)?;
        let [peak_lower, peak_upper] = &context.conveyance_peak;
        let [log_lower, log_upper] = &context.log_peak;
        let bits = |t: &Tensor<Cpu>| -> Result<Vec<u64>, StructureError> {
            Ok(t.try_to_vec::<f64>()?.iter().map(|v| v.to_bits()).collect())
        };
        assert_eq!(bits(peak_lower)?[..2], [CONVEYANCE_PEAK[0], 1f64.to_bits()]);
        assert_eq!(bits(peak_upper)?[..2], [CONVEYANCE_PEAK[1], 1f64.to_bits()]);
        assert_eq!(bits(log_lower)?[..2], [LOG_PEAK_CONVEYANCE[0], 0]);
        assert_eq!(bits(log_upper)?[..2], [LOG_PEAK_CONVEYANCE[1], 0]);
        let tops = context.normal_top.try_to_vec::<f64>()?;
        assert_eq!(tops[..2], [0.93818, 1.0]);
        let lower = peak_lower.try_to_vec::<f64>()?;
        let upper = peak_upper.try_to_vec::<f64>()?;
        assert!(
            tops[2] == lower[2] && lower[2] < upper[2],
            "{lower:?} {upper:?}"
        );
        Ok(())
    }

    #[test]
    fn unit_barrels_keep_the_shape_at_a_rise_in_one_to_two() -> Result<(), StructureError> {
        let tiny = 2f64.powi(-400);
        let barrel =
            CulvertBarrel::pipe_arch(1.8542 * tiny, 1.397 * tiny, 0.4572 * tiny, 1.938_528 * tiny)?;
        let unit = unit_barrel(0, &barrel)?;
        assert!((1.0..2.0).contains(&unit.rise_m()));
        let ratio = unit.rise_m() / barrel.rise_m();
        assert_eq!(ratio, 2f64.powi(ratio.log2().round() as i32));
        assert_eq!(unit.span_m(), barrel.span_m() * ratio);
        let circle = CulvertBarrel::circular(tiny)?;
        assert_eq!(unit_barrel(0, &circle)?.rise_m(), tiny);
        Ok(())
    }
}
