use super::area_point_query::PrismaticAreaPointRecovery;
use super::{area_query, config, query::Fields};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use std::sync::Arc;

impl<R> PrismaticAreaPointRecovery<R>
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
        rows: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        area_query::selection_metadata(a, rows, &[rows.numel()], DType::I64)?;
        area_query::selection_metadata(a, &self.owners, &[self.owners.numel()], DType::I64)?;
        config::check_work(rows.numel(), self.context.maximum_bands + 1)?;
        let c = a.client();
        let captured = Tensor::empty(rows.shape(), DType::I64, c.device())?;
        c.copy_into(&captured, rows)?;
        let rows = &captured;
        Ok(Self {
            context: Arc::clone(&self.context),
            owners: c.index_select(&self.owners, 0, rows)?,
            requested_areas: area_query::select_expansion(a, &self.requested_areas, rows)?,
            dry: c.index_select(&self.dry, 0, rows)?,
            fields: Fields {
                depth: area_query::select_interval(a, &self.fields.depth, rows)?,
                area: area_query::select_interval(a, &self.fields.area, rows)?,
                top_width: area_query::select_interval(a, &self.fields.top_width, rows)?,
                integral: area_query::select_interval(a, &self.fields.integral, rows)?,
                hydraulic_depth: area_query::select_interval(
                    a,
                    &self.fields.hydraulic_depth,
                    rows,
                )?,
            },
            hydrostatic_point: area_query::select_expansion(a, &self.hydrostatic_point, rows)?,
            coordinates: area_query::select_coordinates(a, &self.coordinates, Some(rows))?,
        })
    }
}
