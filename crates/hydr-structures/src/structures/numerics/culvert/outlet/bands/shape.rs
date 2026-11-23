//! Construction of the march bands from unit-rise arc bands.

use super::super::trig::fill;
use super::{ArcShape, MarchBand};
use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::ArcRows;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

impl<R: Runtime<DType = DType>> ArcShape<R> {
    /// The march bands of the unit-rise arc layouts `rows`.
    pub(crate) fn new(a: &mut Arithmetic<'_, R>, rows: &ArcRows<R>) -> ArithmeticResult<Self>
    where
        R::Client: BarrelClient<R>,
    {
        Self::from_bands(a, &rows.bands, [&rows.arced, &rows.floored, &rows.floored])
    }

    /// The march bands of the unit-rise bands `bands_in` on the rows `arced`. `floored` marks the
    /// rows whose three bands repeat one arc, which take the top band. `topped` marks the rows
    /// whose harmless coordinate lies in the middle of the top band.
    pub(crate) fn from_bands(
        a: &mut Arithmetic<'_, R>,
        bands_in: &[ArcBand<R>; 3],
        [arced, floored, topped]: [&Tensor<R>; 3],
    ) -> ArithmeticResult<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = a.client();
        let like = arced;
        let pi = [
            fill(like, f64::from_bits(PI[0]))?,
            fill(like, f64::from_bits(PI[1]))?,
        ];
        let crown = [fill(like, 1.0)?, fill(like, 1.0)?];
        let two = a.point(&fill(like, 2.0)?)?;
        let mut bands = Vec::with_capacity(bands_in.len());
        for (k, band) in bands_in.iter().enumerate() {
            let (end_angle, end_depth) = match bands_in.get(k + 1) {
                Some(next) => (next.start_angle.clone(), next.start_depth.clone()),
                None => (pi.clone(), crown.clone()),
            };
            let x = a.bounds(&band.centre_x[0], &band.centre_x[1])?;
            let r = a.bounds(&band.radius[0], &band.radius[1])?;
            let angle = a.bounds(&band.start_angle[0], &band.start_angle[1])?;
            let sine = a.bounds(&band.start_sin[0], &band.start_sin[1])?;
            let cosine = a.bounds(&band.start_cos[0], &band.start_cos[1])?;
            let start = a.bounds(&band.start_area[0], &band.start_area[1])?;
            let lever = a.mul(&two, &x)?;
            let lever = a.mul(&lever, &r)?;
            let lift = a.mul(&lever, &cosine)?;
            let product = a.mul(&sine, &cosine)?;
            let segment = a.sub(&angle, &product)?;
            let square = a.mul(&r, &r)?;
            let segment = a.mul(&square, &segment)?;
            let raised = a.add(&start, &lift)?;
            let offset = a.sub(&raised, &segment)?;
            bands.push(MarchBand {
                centre_x: band.centre_x.clone(),
                centre_y: band.centre_y.clone(),
                radius: band.radius.clone(),
                start_angle: band.start_angle.clone(),
                end_angle,
                start_depth: band.start_depth.clone(),
                end_depth,
                start_perimeter: band.start_perimeter.clone(),
                area_offset: [offset.lower().clone(), offset.upper().clone()],
            });
        }
        let [first, rest @ ..] = bands_in;
        let mut least_radius = first.radius[0].clone();
        for band in rest {
            least_radius = c.minimum(&least_radius, &band.radius[0])?;
        }
        // The arch arc runs from its floor normal to pi, and an open top band from its start.
        let [.., top_band] = bands_in;
        let top = &top_band.start_angle[1];
        let middle = c.mul_scalar(&c.add(top, &pi[0])?, 0.5)?;
        let harmless = c.where_cond(topped, &middle, &fill(like, 0.5)?)?;
        Ok(Self {
            arced: arced.clone(),
            floored: floored.clone(),
            bands,
            least_radius,
            harmless,
        })
    }
}
