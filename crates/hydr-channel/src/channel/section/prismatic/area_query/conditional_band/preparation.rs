use super::super::super::{
    boundary_area::BoundaryAreas, config, hydraulic_geometry::GatheredBands,
};
use super::super::{SupportedAreaQuery, admission, source_difference::source_difference};
use super::clipping::{self, clip};
use super::endpoint_authority;
use super::selection::{capture, compact};
use super::{
    ConditionalBandSegments, PreparedAreaPrefix,
    target_coordinates::{matrix, target},
    types::LocalEndpoint,
};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::PrismaticAreaPointRecovery,
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

struct SourceOrigin<'a, R: Runtime<DType = DType>> {
    area: &'a CertifiedExpansion<R>,
    delta: &'a CertifiedExpansion<R>,
    target_area: &'a CertifiedExpansion<R>,
    exact_dry: bool,
}

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
    pub(crate) fn prepare(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        source: &PrismaticAreaPointRecovery<R>,
        endpoint: &SupportedAreaQuery<R>,
        boundaries: &BoundaryAreas<R>,
    ) -> HydraulicResult<Self> {
        let difference = source_difference(a, bands, endpoint, source, boundaries)?;
        let target_area = endpoint.retained_endpoint_area(a, bands, boundaries)?;
        Self::prepare_origin(
            a,
            bands,
            endpoint,
            boundaries,
            SourceOrigin {
                area: source.requested_areas_m2(),
                delta: &difference.lower,
                target_area: &target_area,
                exact_dry: false,
            },
        )
    }

    pub(crate) fn prefix_from_exact_dry(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        endpoint: &SupportedAreaQuery<R>,
        boundaries: &BoundaryAreas<R>,
    ) -> HydraulicResult<PreparedAreaPrefix<R>> {
        let endpoint_area = endpoint.retained_endpoint_area(a, bands, boundaries)?;
        let zeros = Tensor::zeros(bands.section_owner.shape(), DType::F64, a.client().device())?;
        let origin = a.expansion_point(&zeros)?;
        let segments = Self::prepare_origin(
            a,
            bands,
            endpoint,
            boundaries,
            SourceOrigin {
                area: &origin,
                delta: &endpoint_area,
                target_area: &endpoint_area,
                exact_dry: true,
            },
        )?;
        Ok(PreparedAreaPrefix {
            segments,
            endpoint_area,
            band_count: bands.band_count,
        })
    }

    fn prepare_origin(
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
