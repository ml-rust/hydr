use super::{
    band_geometry::BandRows,
    batch::PrismaticSectionBatch,
    context::Context,
    inverse_bands,
    query_input::{self, Inputs, QueryError},
};
use crate::channel::flow::surveyed::error::{
    Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection,
};
use crate::channel::section::PrismaticRecoveryConfig;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps, UnaryOps,
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::sync::Arc;

pub(crate) struct HydraulicGeometry<R: Runtime<DType = DType>> {
    context: Arc<Context<R>>,
}

pub(crate) struct GatheredBands<R: Runtime<DType = DType>> {
    pub(super) model: Arc<Context<R>>,
    pub(crate) section_owner: Tensor<R>,
    pub(crate) band_count: usize,
    pub(crate) valid: Tensor<R>,
    pub(crate) pointed: Tensor<R>,
    pub(crate) start_depth: Interval<R>,
    pub(crate) end_depth: Interval<R>,
    pub(crate) thickness: Interval<R>,
    pub(crate) start_area: Interval<R>,
    pub(crate) end_area: Interval<R>,
    pub(crate) area_increment: Interval<R>,
    pub(crate) start_width: Interval<R>,
    pub(crate) slope: Interval<R>,
    pub(crate) start_integral: Interval<R>,
    pub(crate) cap_depth: Interval<R>,
    pub(crate) cap_area: Interval<R>,
    pub(crate) cap_width: Interval<R>,
    pub(crate) cap_integral: Interval<R>,
}

pub(crate) struct Constitutive<R: Runtime<DType = DType>> {
    pub(crate) depth: Interval<R>,
    pub(crate) forward_area: Interval<R>,
    pub(crate) area: Interval<R>,
    pub(crate) top_width: Interval<R>,
    pub(crate) integral: Interval<R>,
    pub(crate) hydraulic_depth: Interval<R>,
    pub(crate) wet_possible: Tensor<R>,
    pub(crate) dry_possible: Tensor<R>,
}

impl<R: Runtime<DType = DType>> HydraulicGeometry<R> {
    pub(crate) fn same_model(&self, bands: &GatheredBands<R>) -> bool {
        Arc::ptr_eq(&self.context, &bands.model)
    }

    pub(crate) fn from_batch(batch: &PrismaticSectionBatch<R>) -> Self {
        Self {
            context: Arc::clone(&batch.context),
        }
    }
    pub(crate) fn client(&self) -> &R::Client {
        &self.context.source.client
    }
}

impl<R> GatheredBands<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn select(
        &self,
        a: &mut Arithmetic<'_, R>,
        row: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let gather =
            |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
                Ok(a.bounds(
                    &c.index_select(value.lower(), 0, row)?,
                    &c.index_select(value.upper(), 0, row)?,
                )?)
            };
        Ok(Self {
            model: Arc::clone(&self.model),
            section_owner: c.index_select(&self.section_owner, 0, row)?,
            band_count: self.band_count,
            valid: c.index_select(&self.valid, 0, row)?,
            pointed: c.index_select(&self.pointed, 0, row)?,
            start_depth: gather(a, &self.start_depth)?,
            end_depth: gather(a, &self.end_depth)?,
            thickness: gather(a, &self.thickness)?,
            start_area: gather(a, &self.start_area)?,
            end_area: gather(a, &self.end_area)?,
            area_increment: gather(a, &self.area_increment)?,
            start_width: gather(a, &self.start_width)?,
            slope: gather(a, &self.slope)?,
            start_integral: gather(a, &self.start_integral)?,
            cap_depth: gather(a, &self.cap_depth)?,
            cap_area: gather(a, &self.cap_area)?,
            cap_width: gather(a, &self.cap_width)?,
            cap_integral: gather(a, &self.cap_integral)?,
        })
    }
    pub(super) fn rows(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<BandRows<R>> {
        Ok(BandRows {
            valid: self.valid.clone(),
            pointed: self.pointed.clone(),
            start_depth: a.bounds(self.start_depth.lower(), self.start_depth.upper())?,
            end_depth: a.bounds(self.end_depth.lower(), self.end_depth.upper())?,
            thickness: a.bounds(self.thickness.lower(), self.thickness.upper())?,
            start_area: a.bounds(self.start_area.lower(), self.start_area.upper())?,
            end_area: a.bounds(self.end_area.lower(), self.end_area.upper())?,
            area_increment: a.bounds(self.area_increment.lower(), self.area_increment.upper())?,
            start_width: a.bounds(self.start_width.lower(), self.start_width.upper())?,
            slope: a.bounds(self.slope.lower(), self.slope.upper())?,
            start_integral: a.bounds(self.start_integral.lower(), self.start_integral.upper())?,
            cap_depth: a.bounds(self.cap_depth.lower(), self.cap_depth.upper())?,
            cap_area: a.bounds(self.cap_area.lower(), self.cap_area.upper())?,
            cap_width: a.bounds(self.cap_width.lower(), self.cap_width.upper())?,
            cap_integral: a.bounds(self.cap_integral.lower(), self.cap_integral.upper())?,
        })
    }
}

pub(super) fn query_error(error: QueryError) -> HydraulicError {
    match error {
        QueryError::Geometry(error) => HydraulicError::Geometry(error),
        QueryError::Arithmetic(error) => HydraulicError::Arithmetic(error),
    }
}

impl<R> HydraulicGeometry<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn boundary_areas(
        &self,
        a: &mut Arithmetic<'_, R>,
    ) -> HydraulicResult<super::boundary_area::BoundaryAreaTable<R>> {
        super::boundary_area::BoundaryAreaTable::build(a, &self.context)
    }

    pub(crate) fn recover_query(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        query: &super::area_query::SupportedAreaQuery<R>,
    ) -> HydraulicResult<super::area_query::AreaQueryGeometry<R>> {
        super::area_query::evaluation::recover(a, &self.context, bands, query)
    }

    pub(crate) fn gather(
        &self,
        a: &mut Arithmetic<'_, R>,
        owners: &Tensor<R>,
    ) -> HydraulicResult<GatheredBands<R>> {
        let rows = BandRows::gather(a, &self.context, owners).map_err(query_error)?;
        let captured = Tensor::empty(owners.shape(), DType::I64, self.client().device())?;
        self.client().copy_into(&captured, owners)?;
        Ok(GatheredBands {
            model: Arc::clone(&self.context),
            section_owner: captured,
            band_count: self.context.maximum_bands,
            valid: rows.valid,
            pointed: rows.pointed,
            start_depth: rows.start_depth,
            end_depth: rows.end_depth,
            thickness: rows.thickness,
            start_area: rows.start_area,
            end_area: rows.end_area,
            area_increment: rows.area_increment,
            start_width: rows.start_width,
            slope: rows.slope,
            start_integral: rows.start_integral,
            cap_depth: rows.cap_depth,
            cap_area: rows.cap_area,
            cap_width: rows.cap_width,
            cap_integral: rows.cap_integral,
        })
    }

    pub(crate) fn recover(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        original_area: &Tensor<R>,
        config: &PrismaticRecoveryConfig,
    ) -> HydraulicResult<Constitutive<R>> {
        let c = a.client();
        let zero = &self.context.source.zero;
        if original_area.numel() == 0 {
            let depth = a.point(original_area)?;
            return self.at_depth(a, bands, &depth, Some(original_area));
        }
        let wet = c.gt(original_area, zero)?;
        let dry = c.eq(original_area, zero)?;
        for (reason, mask) in [
            (
                PrismaticRiemannRejection::CapExceeded,
                c.mul(&wet, &c.gt(original_area, bands.cap_area.upper())?)?,
            ),
            (
                PrismaticRiemannRejection::CapUnresolved,
                c.mul(
                    &wet,
                    &c.mul(
                        &c.gt(original_area, bands.cap_area.lower())?,
                        &c.le(original_area, bands.cap_area.upper())?,
                    )?,
                )?,
            ),
        ] {
            if let Some(query) = self.context.source.rejected_lane(&mask)? {
                let owner = bands.section_owner.narrow(0, query, 1)?.item::<i64>()?;
                return Err(HydraulicError::Rejected { reason, query: Some(query), section_owner: usize::try_from(owner).ok(), context: "conserved area exceeds or overlaps certified cap support. Check input units and survey conditioning".into() });
            }
        }
        let rows = bands.rows(a)?;
        let inputs = Inputs {
            owners: bands.section_owner.clone(),
            values: original_area.clone(),
            wet: wet.clone(),
            dry,
        };
        let depth =
            inverse_bands::recover(a, &self.context, &rows, &inputs).map_err(query_error)?;
        query_input::reject(
            &self.context,
            &bands.section_owner,
            &c.mul(&wet, &c.le(depth.lower(), zero)?)?,
            true,
            "positive area has no positive recovered depth. Check profile conditioning",
        )
        .map_err(query_error)?;
        let lower = a.point(depth.lower())?;
        let upper = a.point(depth.upper())?;
        let width = a.sub(&upper, &lower)?;
        let scale = a.point(&c.maximum(&c.abs(depth.lower())?, &c.abs(depth.upper())?)?)?;
        let relative = a.point(&Tensor::from_slice(&[config.depth_rtol], &[], c.device())?)?;
        let absolute = a.point(&Tensor::from_slice(
            &[config.depth_atol_m],
            &[],
            c.device(),
        )?)?;
        let relative = a.mul(&relative, &scale)?;
        let allowance = a.add(&absolute, &relative)?;
        query_input::reject(
            &self.context,
            &bands.section_owner,
            &c.gt(width.upper(), &c.maximum(allowance.lower(), zero)?)?,
            true,
            "recovered depth exceeds its width allowance. Check profile conditioning",
        )
        .map_err(query_error)?;
        let state = self.at_depth(a, bands, &depth, Some(original_area))?;
        let outside = c.maximum(
            &c.gt(state.forward_area.lower(), original_area)?,
            &c.lt(state.forward_area.upper(), original_area)?,
        )?;
        query_input::reject(
            &self.context,
            &bands.section_owner,
            &outside,
            true,
            "recovered forward area excludes conserved area. Check profile conditioning",
        )
        .map_err(query_error)?;
        Ok(state)
    }

    pub(crate) fn at_depth(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        depth: &Interval<R>,
        original_area: Option<&Tensor<R>>,
    ) -> HydraulicResult<Constitutive<R>> {
        super::hydraulic_evaluation::evaluate(a, &self.context, bands, depth, original_area)
    }

    pub(crate) fn at_normalized_depth(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        t: &Interval<R>,
    ) -> HydraulicResult<Constitutive<R>> {
        let c = a.client();
        let zero = &self.context.source.zero;
        let one = &self.context.source.one;
        let t = a.bounds(&c.maximum(t.lower(), zero)?, &c.minimum(t.upper(), one)?)?;
        let product = a.mul(&t, &bands.cap_depth)?;
        let depth = a.bounds(
            &c.maximum(product.lower(), zero)?,
            &c.minimum(product.upper(), bands.cap_depth.upper())?,
        )?;
        let mut state = self.at_depth(a, bands, &depth, None)?;
        let at_cap = c.mul(&c.eq(t.lower(), one)?, &c.eq(t.upper(), one)?)?;
        let choose = |a: &mut Arithmetic<'_, R>,
                      value: &Interval<R>,
                      cap: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.where_cond(&at_cap, cap.lower(), value.lower())?,
                &c.where_cond(&at_cap, cap.upper(), value.upper())?,
            )?)
        };
        state.depth = choose(a, &state.depth, &bands.cap_depth)?;
        state.forward_area = choose(a, &state.forward_area, &bands.cap_area)?;
        state.area = choose(a, &state.area, &bands.cap_area)?;
        state.top_width = choose(a, &state.top_width, &bands.cap_width)?;
        state.integral = choose(a, &state.integral, &bands.cap_integral)?;
        let cap_hydraulic = a.div_positive_interval(&bands.cap_area, &bands.cap_width)?;
        state.hydraulic_depth = choose(a, &state.hydraulic_depth, &cap_hydraulic)?;
        Ok(state)
    }
}
