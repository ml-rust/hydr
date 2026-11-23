//! Construction-time depth certificates of the floored rows of a barrel context.
//!
//! A floored circle or arc row certifies its open section in `s = y/D'`, read in its frame: the
//! barrel and its floor scaled by a power of two to a rise in `[1, 2)`. Each distinct floored
//! barrel is certified once and its certificate copied to its other rows. A floored box keeps
//! the box constants, which hold for every bed roughness.
//!
//! The context rejects, naming the first failing row. One scalar host read counts the failing
//! rows. The failing mask reaches the host only on rejection.

use super::certificates::rejection;
use super::context::{Bounds, TensorBarrelContext};
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::numerics::culvert::certify::{self, DepthCertificates, Reading};
use crate::structures::numerics::culvert::constants::FLOOR_CRITICAL_TOP;
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::ops::{ConditionalOps, IndexingOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

impl<R: Runtime<DType = DType>> TensorBarrelContext<R> {
    /// Certifies the floored circle and arc rows and writes their depth domains, slope floors
    /// and peak bounds over the barrel's own.
    ///
    /// Rejects, naming the first floored row without a certificate.
    pub(super) fn certify_floors(
        mut self,
        barrels: &[CulvertBarrel],
        rounds: usize,
    ) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        // The first row of each distinct floored barrel, and each floored row's position in it.
        let mut owners: Vec<usize> = Vec::new();
        let mut floored_rows = Vec::new();
        let mut copies = Vec::new();
        for (row, culvert) in barrels.iter().enumerate() {
            if culvert.floor().is_none() || culvert.shape() == CulvertBarrelShape::Box {
                continue;
            }
            let found = owners
                .iter()
                .position(|&owner| barrels.get(owner) == Some(culvert));
            let position = found.unwrap_or(owners.len());
            if found.is_none() {
                owners.push(row);
            }
            floored_rows.push(row as i64);
            copies.push(position as i64);
        }
        if owners.is_empty() {
            return Ok(self);
        }
        let indices: Vec<i64> = owners.iter().map(|&row| row as i64).collect();
        let device = &self.device;
        let index = Tensor::<R>::from_slice(&indices, &[indices.len()], device)?;
        let targets = Tensor::<R>::from_slice(&floored_rows, &[floored_rows.len()], device)?;
        let copies = Tensor::<R>::from_slice(&copies, &[copies.len()], device)?;
        let ones = Tensor::<R>::ones(&[indices.len()], DType::F64, device)?;
        let frame = self.frame.as_ref().map_or(&self, |frame| &frame.context);
        let framed = frame.select(&index)?;
        if framed.floor.is_none() {
            return Err(StructureError::numerical(
                "the barrel frame lost its floor rows: report this runtime",
            ));
        }
        let certificates = with_interval_arithmetic(
            &self.client,
            DType::F64,
            "culvert floor certificates",
            |a| certify::certify(a, &framed, Reading::Floors, rounds),
        )?;
        let c = &self.client;
        let failed = c.sum(&certificates.failed, &[0], false)?.item::<f64>()?;
        if failed != 0.0 {
            return Err(rejection(barrels, &owners, &certificates, true));
        }
        let DepthCertificates {
            critical_slope,
            normal_top,
            normal_slope,
            peak,
            log_peak,
            limb,
            ..
        } = certificates;
        let Some(limb) = limb else {
            return Err(StructureError::numerical(
                "the floor certificates lost their rising-limb knots: report this runtime",
            ));
        };
        let put = |base: &Tensor<R>, value: &Tensor<R>| {
            c.index_put(base, 0, &targets, &c.index_select(value, 0, &copies)?)
        };
        let pair =
            |[lower, upper]: &Bounds<R>, [l, u]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
                Ok([put(lower, l)?, put(upper, u)?])
            };
        let floored = put(&self.zero, &ones)?;
        self.critical_top = c.where_cond(
            &floored,
            &self.constant(FLOOR_CRITICAL_TOP)?,
            &self.critical_top,
        )?;
        self.critical_slope = put(&self.critical_slope, &critical_slope)?;
        self.normal_top = put(&self.normal_top, &normal_top)?;
        self.normal_slope = put(&self.normal_slope, &normal_slope)?;
        self.conveyance_peak = pair(&self.conveyance_peak, &peak)?;
        self.log_peak = pair(&self.log_peak, &log_peak)?;
        let harmless = [self.constant(0.5)?, self.one.clone()];
        self.limb = Some(
            limb.iter()
                .map(|knot| pair(&harmless, knot))
                .collect::<numr::error::Result<Vec<_>>>()?,
        );
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::numerics::culvert::certify::REFINEMENT_ROUNDS;
    use crate::structures::numerics::culvert::constants::CIRCLE_CRITICAL_TOP;
    use crate::structures::{CulvertFloor, StructureInputErrorKind};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    type Cpu = CpuRuntime;

    fn client() -> <Cpu as Runtime>::Client {
        Cpu::default_client(&CpuDevice::new())
    }

    fn bed(barrel: CulvertBarrel, depth: f64, ratio: f64) -> Result<CulvertBarrel, StructureError> {
        barrel.with_floor(CulvertFloor {
            sediment_depth_m: depth,
            roughness_ratio: ratio,
        })
    }

    /// Every certificate tensor of `context`, as bits per field.
    fn bits(context: &TensorBarrelContext<Cpu>) -> Result<Vec<Vec<u64>>, StructureError> {
        let [peak_lower, peak_upper] = &context.conveyance_peak;
        let [log_lower, log_upper] = &context.log_peak;
        let mut fields = vec![
            &context.critical_top,
            &context.critical_slope,
            &context.normal_top,
            &context.normal_slope,
            peak_lower,
            peak_upper,
            log_lower,
            log_upper,
        ];
        for [knot, floor] in context.limb.iter().flatten() {
            fields.extend([knot, floor]);
        }
        fields
            .into_iter()
            .map(|t| Ok(t.try_to_vec::<f64>()?.iter().map(|v| v.to_bits()).collect()))
            .collect()
    }

    #[test]
    fn unrefined_floor_cover_rejects_naming_the_row_and_its_floor() -> Result<(), StructureError> {
        let barrels = [
            CulvertBarrel::circular(1.0)?,
            bed(CulvertBarrel::circular(1.2)?, 0.24, 2.5)?,
        ];
        let error = TensorBarrelContext::<Cpu>::build(client(), &barrels, 0).err();
        assert!(matches!(
            error,
            Some(StructureError::InvalidInput { element: Some(1), kind: StructureInputErrorKind::Definition, ref context })
                if context == "shape=Circular,span_m=1.2,rise_m=1.2,given_radius_m=[],sediment_depth_m=0.24,roughness_ratio=2.5: conveyance has no certified single peak below the crown. Change the dimensions or report this barrel"
        ));
        assert!(TensorBarrelContext::<Cpu>::build(client(), &barrels, REFINEMENT_ROUNDS).is_ok());
        Ok(())
    }

    #[test]
    fn floored_rows_take_their_own_domains_and_boxes_keep_the_box_constants()
    -> Result<(), StructureError> {
        let circle = CulvertBarrel::circular(1.2)?;
        let boxed = CulvertBarrel::rectangular_box(2.0, 1.5)?;
        let barrels = [
            circle,
            bed(circle, 0.24, 0.5)?,
            bed(circle, 0.24, 2.5)?,
            bed(boxed, 0.0, 2.5)?,
            bed(boxed, 0.45, 0.5)?,
            bed(circle, 0.24, 0.5)?,
        ];
        let context = TensorBarrelContext::<Cpu>::new(client(), &barrels)?;
        let read = |t: &Tensor<Cpu>| t.try_to_vec::<f64>();
        let tops = read(&context.critical_top)?;
        assert_eq!(
            tops,
            [
                CIRCLE_CRITICAL_TOP,
                FLOOR_CRITICAL_TOP,
                FLOOR_CRITICAL_TOP,
                1.0,
                1.0,
                FLOOR_CRITICAL_TOP
            ]
        );
        assert_eq!(read(&context.critical_slope)?[3..5], [3.0, 3.0]);
        assert_eq!(read(&context.normal_slope)?[3..5], [3.0, 3.0]);
        assert_eq!(read(&context.normal_top)?[3..5], [1.0, 1.0]);
        let [low, high] = [
            read(&context.conveyance_peak[0])?,
            read(&context.conveyance_peak[1])?,
        ];
        assert_eq!((low[3], high[3]), (1.0, 1.0));
        // A rougher bed raises the peak, and repeated barrels share one certificate.
        assert!(high[1] < low[2], "{low:?} {high:?}");
        assert_eq!((low[1], high[1]), (low[5], high[5]));
        let knots = context.limb.as_ref().map_or(0, Vec::len);
        assert_eq!(knots, 8);
        for [knot, floor] in context.limb.iter().flatten() {
            let (knot, floor) = (read(knot)?, read(floor)?);
            for row in [1, 2, 5] {
                assert!(knot[row] <= read(&context.normal_top)?[row], "{knot:?}");
                assert!(floor[row] >= read(&context.normal_slope)?[row], "{floor:?}");
            }
            assert_eq!((knot[0], floor[0]), (0.5, 1.0));
        }
        Ok(())
    }

    #[test]
    fn floor_certificates_do_not_change_with_the_scale_of_the_barrel() -> Result<(), StructureError>
    {
        let shapes = |s: f64| -> Result<Vec<CulvertBarrel>, StructureError> {
            Ok(vec![
                bed(CulvertBarrel::circular(1.2 * s)?, 0.24 * s, 2.5)?,
                bed(
                    CulvertBarrel::horizontal_ellipse(1.524 * s, 0.9652 * s, 0.419_475_92 * s)?,
                    0.4826 * s,
                    2.5,
                )?,
                bed(
                    CulvertBarrel::pipe_arch(1.8542 * s, 1.397 * s, 0.4572 * s, 1.938_528 * s)?,
                    0.2794 * s,
                    0.5,
                )?,
                bed(CulvertBarrel::arch(3.048 * s, 1.6002 * s)?, 0.0, 2.5)?,
            ])
        };
        let unit = TensorBarrelContext::<Cpu>::new(client(), &shapes(1.0)?)?;
        for power in [-900, 300] {
            let scaled = TensorBarrelContext::<Cpu>::new(client(), &shapes(2f64.powi(power))?)?;
            assert_eq!(bits(&scaled)?, bits(&unit)?, "2^{power}");
        }
        Ok(())
    }

    #[test]
    fn rows_without_a_floor_keep_their_certificate_bits() -> Result<(), StructureError> {
        let plain = [
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
            CulvertBarrel::arch(3.048, 1.6002)?,
        ];
        let alone = TensorBarrelContext::<Cpu>::new(client(), &plain)?;
        // A unit rise once put the harmless crown-band depth of non-arc rows outside its arc.
        for rise in [1.0, 1.5] {
            let pairs = [
                [
                    CulvertBarrel::rectangular_box(2.0, rise)?,
                    bed(plain[1], 0.349, 2.5)?,
                ],
                [
                    bed(CulvertBarrel::rectangular_box(2.0, rise)?, 0.25, 2.5)?,
                    plain[1],
                ],
                [CulvertBarrel::circular(rise)?, bed(plain[3], 0.4, 0.5)?],
            ];
            for pair in pairs {
                TensorBarrelContext::<Cpu>::new(client(), &pair)?;
            }
        }
        let mut mixed = plain.to_vec();
        for barrel in plain {
            mixed.push(bed(barrel, 0.25 * barrel.rise_m(), 2.5)?);
        }
        let beside = TensorBarrelContext::<Cpu>::new(client(), &mixed)?;
        let first = |fields: Vec<Vec<u64>>| -> Vec<Vec<u64>> {
            fields
                .into_iter()
                .take(8)
                .map(|f| f.into_iter().take(4).collect())
                .collect()
        };
        assert_eq!(first(bits(&beside)?), first(bits(&alone)?));
        Ok(())
    }
}
