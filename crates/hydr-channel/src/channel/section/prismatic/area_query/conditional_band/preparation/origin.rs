use super::super::{
    ConditionalBandSegments,
    clipping::{self, clip},
    endpoint_authority,
    selection::{capture, compact},
    target_coordinates::{matrix, target},
    types::LocalEndpoint,
};
use super::SourceOrigin;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use crate::channel::section::prismatic::{
    area_query::{SupportedAreaQuery, admission},
    boundary_area::BoundaryAreas,
    config,
    hydraulic_geometry::GatheredBands,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

impl<R> ConditionalBandSegments<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn prepare_origin(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        endpoint: &SupportedAreaQuery<R>,
        boundaries: &BoundaryAreas<R>,
        origin: SourceOrigin<'_, R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        admission::reject(
            a,
            &c.ne(
                endpoint.query.parameter.lower(),
                endpoint.query.parameter.upper(),
            )?,
            "conditional band preparation requires one physical endpoint. Select a supported physical endpoint",
        )?;
        let n = bands.section_owner.numel();
        let b = bands.band_count;
        let boundary_count =
            b.checked_add(1)
                .ok_or_else(|| numr::error::Error::InvalidArgument {
                    arg: "band_count",
                    reason: format!("band_count={b} overflows boundary count. Split the batch"),
                })?;
        config::check_work(n, boundary_count)?;
        let m = config::check_work(n, b)?;
        let shape = [n, b];
        let zero = Tensor::zeros(&shape, DType::F64, c.device())?;
        let indices = c.arange(0.0, b as f64, 1.0, DType::I64)?.reshape(&[1, b])?;
        let boundary =
            |a: &mut Arithmetic<'_, R>, index| -> HydraulicResult<CertifiedExpansion<R>> {
                let values = boundaries.areas();
                Ok(a.expansion_bounds(
                    &values.high().narrow(1, index, b)?,
                    &values.low().narrow(1, index, b)?,
                    &values.abs_error().narrow(1, index, b)?,
                )?)
            };
        let source_area = matrix(a, origin.area, &shape)?;
        let start_area = boundary(a, 0)?;
        let end_area = boundary(a, 1)?;
        let source_above = a.expansion_sub(&source_area, &start_area)?;
        let source_below = a.expansion_sub(&end_area, &source_area)?;
        let exact_zero = |value: &CertifiedExpansion<R>| -> HydraulicResult<Tensor<R>> {
            Ok(c.mul(
                &c.eq(value.high(), &zero)?,
                &c.mul(&c.eq(value.low(), &zero)?, &c.eq(value.abs_error(), &zero)?)?,
            )?)
        };
        let first_band = c.mul(
            &bands.valid,
            &c.cast(
                &c.eq(&indices, &Tensor::zeros(&[], DType::I64, c.device())?)?,
                DType::F64,
            )?,
        )?;
        let source_start_identity = if origin.exact_dry {
            first_band.clone()
        } else {
            exact_zero(&source_above)?
        };
        let source_end_identity = if origin.exact_dry {
            zero.clone()
        } else {
            exact_zero(&source_below)?
        };
        let delta = matrix(a, origin.delta, &shape)?;
        let target = target(a, bands, endpoint, boundaries, &indices)?;
        let source_above = a.expansion_interval(&source_above)?;
        let source_below = a.expansion_interval(&source_below)?;
        let target_above = target.above;
        let target_below = target.below;
        let start_identity = target.at_start;
        let end_identity = target.at_end;
        let inside = |above: &Interval<R>,
                      below: &Interval<R>,
                      at_start: &Tensor<R>,
                      at_end: &Tensor<R>|
         -> HydraulicResult<Tensor<R>> {
            Ok(c.mul(
                &c.maximum(&c.ge(above.lower(), &zero)?, at_start)?,
                &c.maximum(&c.ge(below.lower(), &zero)?, at_end)?,
            )?)
        };
        let source_inside = if origin.exact_dry {
            first_band.clone()
        } else {
            inside(
                &source_above,
                &source_below,
                &source_start_identity,
                &source_end_identity,
            )?
        };
        let target_inside = inside(&target_above, &target_below, &start_identity, &end_identity)?;
        let unclipped = c.mul(&source_inside, &target_inside)?;
        let start = if origin.exact_dry {
            LocalEndpoint {
                above_start: a.point(&zero)?,
                below_end: a.bounds(bands.area_increment.lower(), bands.area_increment.upper())?,
                at_start: bands.valid.clone(),
                at_end: zero.clone(),
            }
        } else {
            clip(
                a,
                &source_above,
                &source_below,
                &bands.area_increment,
                &source_start_identity,
                &source_end_identity,
                &zero,
            )?
        };
        let end = clip(
            a,
            &target_above,
            &target_below,
            &bands.area_increment,
            &start_identity,
            &end_identity,
            &zero,
        )?;
        let clipped_gap = clipping::gap(a, &bands.valid, &start, &end, &delta, &unclipped, &zero)?;
        let target_area = matrix(a, origin.target_area, &shape)?;
        let source_dry = c.mul(&first_band, &start.at_start)?;
        let target_dry = c.mul(&first_band, &end.at_start)?;
        let retained_start = endpoint_authority::retain(
            a,
            &source_area,
            (&start_area, &end_area),
            &start,
            &source_inside,
            &source_dry,
            &bands.valid,
        )?;
        let retained_end = endpoint_authority::retain(
            a,
            &target_area,
            (&start_area, &end_area),
            &end,
            &target_inside,
            &target_dry,
            &bands.valid,
        )?;
        let gap = clipped_gap.retained;
        let endpoint_area =
            |a: &mut Arithmetic<'_, R>, local: &LocalEndpoint<R>| -> HydraulicResult<Interval<R>> {
                let from_start = a.add(&bands.start_area, &local.above_start)?;
                let from_end = a.sub(&bands.end_area, &local.below_end)?;
                Ok(a.bounds(
                    &c.maximum(from_start.lower(), from_end.lower())?,
                    &c.minimum(from_start.upper(), from_end.upper())?,
                )?)
            };
        let first = endpoint_area(a, &start)?;
        let second = endpoint_area(a, &end)?;
        let endpoint_hull = a.bounds(
            &c.minimum(first.lower(), second.lower())?,
            &c.maximum(first.upper(), second.upper())?,
        )?;
        let flatten =
            |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
                Ok(a.bounds(&value.lower().reshape(&[m])?, &value.upper().reshape(&[m])?)?)
            };
        let local = |a: &mut Arithmetic<'_, R>,
                     value: LocalEndpoint<R>|
         -> HydraulicResult<LocalEndpoint<R>> {
            Ok(LocalEndpoint {
                above_start: a.bounds(
                    &value.above_start.lower().reshape(&[m, 1])?,
                    &value.above_start.upper().reshape(&[m, 1])?,
                )?,
                below_end: a.bounds(
                    &value.below_end.lower().reshape(&[m, 1])?,
                    &value.below_end.upper().reshape(&[m, 1])?,
                )?,
                at_start: value.at_start.reshape(&[m, 1])?,
                at_end: value.at_end.reshape(&[m, 1])?,
            })
        };
        let original_rows = capture(
            a,
            &endpoint
                .query
                .original_rows
                .unsqueeze(1)?
                .broadcast_to(&shape)?
                .contiguous()?
                .reshape(&[m])?,
        )?;
        Ok(Self {
            start_area: retained_start,
            end_area: retained_end,
            rows: compact(a, bands, m)?,
            original_rows,
            band_index: indices.broadcast_to(&shape)?.contiguous()?.reshape(&[m])?,
            active: clipped_gap.active.reshape(&[m])?,
            zero_gap: clipped_gap.zero.reshape(&[m])?,
            start: local(a, start)?,
            end: local(a, end)?,
            gap: a.expansion_bounds(
                &gap.high().reshape(&[m])?,
                &gap.low().reshape(&[m])?,
                &gap.abs_error().reshape(&[m])?,
            )?,
            physical_gap: flatten(a, &clipped_gap.physical)?,
            endpoint_hull: flatten(a, &endpoint_hull)?,
        })
    }
}

#[cfg(test)]
pub(in super::super) mod tests {
    use super::*;
    use crate::channel::section::prismatic::area_query::{
        AreaDirection, AreaQuery, test_support as support,
    };
    use crate::channel::{
        flow::{
            surveyed::{equation_ops::scalar, error},
            surveyed_points::potential_callbacks,
        },
        section::{HydraulicGeometry, PrismaticRecoveryConfig},
    };
    use support::{Client, Result, hydraulic};

    pub(in super::super::super) fn run<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let owners = Tensor::<R>::from_slice(&[1_i64; 2], &[2], client.device())?;
        let areas = support::expansion::<R>(client, &[[4.0_f64.to_bits(), 0, 0]; 2], false)?;
        let source = section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?;
        let distance = f64::from_bits(0x3c30000000000000);
        hydraulic(error::checked::<R, _>(
            client,
            "tiny unclipped conditional gaps",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[distance; 2], &[2], client.device())?)?;
                let parameter =
                    a.point(&Tensor::from_slice(&[1.0, -1.0], &[2], client.device())?)?;
                let endpoint = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &endpoint, &boundaries)?;
                let lanes = Tensor::from_slice(&[0_i64, 3], &[2], client.device())?;
                let selected = segments.select(a, &lanes)?;
                assert_eq!(
                    selected.retained_gap().high().try_to_vec::<f64>()?,
                    [distance, -distance]
                );
                assert_eq!(selected.retained_gap().low().try_to_vec::<f64>()?, [0.0; 2]);
                assert_eq!(
                    selected.retained_gap().abs_error().try_to_vec::<f64>()?,
                    [0.0; 2]
                );
                assert_eq!(
                    selected.physical_gap().lower().try_to_vec::<f64>()?,
                    [distance, -distance]
                );
                assert_eq!(
                    selected.physical_gap().upper().try_to_vec::<f64>()?,
                    [distance, -distance]
                );
                let middle = a.point(&Tensor::from_slice(&[0.5; 2], &[2], client.device())?)?;
                let panel = segments.panel(a, &lanes, &middle)?;
                assert_eq!(
                    panel.gap.lower().try_to_vec::<f64>()?,
                    [distance, -distance]
                );
                assert_eq!(
                    panel.gap.upper().try_to_vec::<f64>()?,
                    [distance, -distance]
                );
                let gravity = scalar(a, 1.0)?;
                let integrand =
                    potential_callbacks::integrand(a, &segments, &lanes, &middle, &gravity)?;
                let lower = integrand.lower().try_to_vec::<f64>()?;
                let upper = integrand.upper().try_to_vec::<f64>()?;
                // Rectangle G=D/sqrt(2*A), with A=4+D/2 and D=+/-2^-60.
                let reference = [
                    f64::from_bits(0x3c16a09e667f3bcc),
                    f64::from_bits(0x3c16a09e667f3bcd),
                ];
                for (lane, expected) in [reference, [-reference[1], -reference[0]]]
                    .into_iter()
                    .enumerate()
                {
                    assert!(lower[lane] <= expected[0] && expected[1] <= upper[lane]);
                    let excess = (lower[lane] - expected[1])
                        .abs()
                        .max((upper[lane] - expected[0]).abs());
                    assert!(excess <= 1e-10 * reference[0]);
                }
                let reordered = Tensor::from_slice(&[3_i64, 0, 3, 1, 4], &[5], client.device())?;
                let parameter = a.point(&Tensor::from_slice(&[0.5; 5], &[5], client.device())?)?;
                let value =
                    potential_callbacks::integrand(a, &segments, &reordered, &parameter, &gravity)?;
                let bound = potential_callbacks::fourth_bound(
                    a, &segments, &reordered, &parameter, &gravity,
                )?;
                let value_lower = value.lower().try_to_vec::<f64>()?;
                let value_upper = value.upper().try_to_vec::<f64>()?;
                let fourth = bound.try_to_vec::<f64>()?;
                for (row, original) in [1_usize, 0, 1].into_iter().enumerate() {
                    assert_eq!(
                        [value_lower[row], value_upper[row]],
                        [lower[original], upper[original]]
                    );
                    assert!(fourth[row].is_finite() && fourth[row] > 0.0);
                }
                for row in [3_usize, 4] {
                    assert_eq!([value_lower[row], value_upper[row]], [0.0; 2]);
                    assert_eq!(fourth[row], 0.0);
                }
                Ok(())
            },
        ))?;
        exact_rectangle_fourth::<R>(client, &section, &geometry)
    }

    fn exact_rectangle_fourth<R>(
        client: &R::Client,
        section: &crate::channel::section::PrismaticSectionBatch<R>,
        geometry: &HydraulicGeometry<R>,
    ) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let owners = Tensor::<R>::from_slice(&[1_i64; 2], &[2], client.device())?;
        let areas = support::expansion::<R>(client, &[[2.0_f64.to_bits(), 0, 0]; 2], false)?;
        let source = section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?;
        let distance = f64::from_bits(0x3c30000000000000);
        hydraulic(error::checked::<R, _>(
            client,
            "rectangular ordinary fourth derivative",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[distance; 2], &[2], client.device())?)?;
                let direction =
                    a.point(&Tensor::from_slice(&[1.0, -1.0], &[2], client.device())?)?;
                let query = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &direction,
                )?
                .prove_support(a, &bands)?;
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &query, &boundaries)?;
                let lanes = Tensor::from_slice(&[0_i64, 3], &[2], client.device())?;
                let parameter = a.point(&Tensor::zeros(&[2], DType::F64, client.device())?)?;
                let gravity = scalar(a, 1.0)?;
                let bound =
                    potential_callbacks::fourth_bound(a, &segments, &lanes, &parameter, &gravity)?;
                // For A=2, T=2, g=1, the exact magnitude at s=0 is 105*|D|^5/512.
                let independent = 105.0 * distance.powi(5) / 512.0;
                assert!(independent > 0.0);
                for value in bound.try_to_vec::<f64>()? {
                    assert!(
                        value.is_finite() && value >= independent,
                        "M4={value}, exact={independent}"
                    );
                }
                let first = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let mut widened = segments.select(a, &first)?;
                let at_start = a.point(&Tensor::zeros(&[1], DType::F64, client.device())?)?;
                let original = widened.panel(a, &first, &at_start)?;
                let radius = 2.0 * distance;
                let center = Tensor::zeros(&[1], DType::F64, client.device())?;
                let uncertainty = Tensor::from_slice(&[radius], &[1], client.device())?;
                // Only the gap certificates widen; the original A=2 to A=2+D endpoint geometry remains intact.
                widened.gap = a.expansion_bounds(&center, &center, &uncertainty)?;
                widened.physical_gap = a.bounds(
                    &Tensor::from_slice(&[-radius], &[1], client.device())?,
                    &uncertainty,
                )?;
                assert_eq!(widened.retained_gap().high().item::<f64>()?, 0.0);
                assert_eq!(widened.retained_gap().low().item::<f64>()?, 0.0);
                assert_eq!(widened.retained_gap().abs_error().item::<f64>()?, radius);
                assert!(-radius <= distance && distance <= radius);
                let retained_geometry = widened.panel(a, &first, &at_start)?;
                for (before, after) in [
                    (&original.area, &retained_geometry.area),
                    (
                        &original.hydraulic_depth,
                        &retained_geometry.hydraulic_depth,
                    ),
                ] {
                    assert_eq!(
                        before.lower().try_to_vec::<f64>()?,
                        after.lower().try_to_vec::<f64>()?
                    );
                    assert_eq!(
                        before.upper().try_to_vec::<f64>()?,
                        after.upper().try_to_vec::<f64>()?
                    );
                }
                let value =
                    potential_callbacks::integrand(a, &widened, &first, &at_start, &gravity)?;
                let lower = value.lower().item::<f64>()?;
                let upper = value.upper().item::<f64>()?;
                assert!(lower < 0.0 && upper > 0.0);
                assert!(lower <= distance / 2.0 && distance / 2.0 <= upper);
                let bound =
                    potential_callbacks::fourth_bound(a, &widened, &first, &at_start, &gravity)?;
                let bound = bound.item::<f64>()?;
                assert!(
                    bound.is_finite() && bound >= independent,
                    "widened M4={bound}, exact={independent}"
                );
                Ok(())
            },
        ))
    }
}
