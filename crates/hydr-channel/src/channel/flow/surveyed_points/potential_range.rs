use super::analytic_bottom::AnalyticSource;
use super::source_terms::SourceRows;
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig,
        error::{Arithmetic, HydraulicError, HydraulicResult, checked},
        work::WorkLedger,
    },
    section::{
        AreaDirection, AreaQuery, BoundaryAreas, GatheredBands, HydraulicGeometry,
        PrismaticAreaPointRecovery, SupportedAreaQuery, source_difference,
    },
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Independent leaf results retain arithmetic uncertainty separately from physical restrictions.
pub(super) struct PotentialValue<R: Runtime<DType = DType>> {
    pub(super) retained: CertifiedExpansion<R>,
    pub(super) physical: Interval<R>,
    /// Only eligible rows carry analytic certificates. Other rows remain unavailable.
    pub(super) analytic: Option<AnalyticSource<R>>,
}

pub(super) struct PotentialEvaluation<'a> {
    pub(super) config: &'a PrismaticRiemannConfig,
    pub(super) panels: usize,
    pub(super) work: &'a WorkLedger,
}

/// Construction checks one model/owner graph. Leaves perform only their requested integral.
pub(super) struct PotentialContext<'a, R: Runtime<DType = DType>> {
    pub(super) geometry: &'a HydraulicGeometry<R>,
    pub(super) bands: &'a GatheredBands<R>,
    pub(super) source: &'a PrismaticAreaPointRecovery<R>,
    pub(super) query: &'a SupportedAreaQuery<R>,
    pub(super) boundaries: &'a BoundaryAreas<R>,
    pub(super) evaluation: PotentialEvaluation<'a>,
    /// Parent rows of a fixed paired side whose source terms persist over target callbacks.
    pub(super) source_rows: Option<SourceRows<'a, R>>,
}

impl<'a, R> PotentialContext<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn new(
        geometry: &'a HydraulicGeometry<R>,
        bands: &'a GatheredBands<R>,
        source: &'a PrismaticAreaPointRecovery<R>,
        query: &'a SupportedAreaQuery<R>,
        boundaries: &'a BoundaryAreas<R>,
        evaluation: PotentialEvaluation<'a>,
    ) -> HydraulicResult<Self> {
        admit_evaluation(geometry, bands, &evaluation)?;
        checked(geometry.client(), "point potential authority", |a| {
            source_difference(a, bands, query, source, boundaries)?;
            Ok(())
        })?;
        Ok(Self {
            geometry,
            bands,
            source,
            query,
            boundaries,
            evaluation,
            source_rows: None,
        })
    }

    /// This context with its rows bound to the parent rows of a fixed paired side.
    pub(super) fn with_source_rows(mut self, rows: Option<SourceRows<'a, R>>) -> Self {
        self.source_rows = rows;
        self
    }

    pub(super) fn source_query(&self) -> HydraulicResult<SupportedAreaQuery<R>> {
        source_query(
            self.geometry,
            self.bands,
            self.source,
            &self.query.query.original_rows,
        )
    }
}

pub(super) fn admit_evaluation<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    evaluation: &PotentialEvaluation<'_>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    evaluation.config.check()?;
    if !evaluation.panels.is_power_of_two()
        || evaluation.panels > evaluation.config.max_simpson_panels
    {
        return Err(HydraulicError::Geometry(
            crate::channel::section::CrossSectionError::InvalidInput {
                section: None,
                context: format!(
                    "potential panels={} exceed the admitted positive power-of-two ceiling={}. Supply admitted panels",
                    evaluation.panels, evaluation.config.max_simpson_panels,
                ),
            },
        ));
    }
    if !geometry.same_model(bands) {
        return Err(HydraulicError::Geometry(crate::channel::section::CrossSectionError::InvalidInput { section: None, context:
                "potential geometry and bands belong to different models. Supply one checked graph".into() }));
    }
    Ok(())
}

/// Original rows already belong to the checked source graph and remain diagnostic identities.
pub(super) fn source_query<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    source: &PrismaticAreaPointRecovery<R>,
    original_rows: &Tensor<R>,
) -> HydraulicResult<SupportedAreaQuery<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(geometry.client(), "point source potential chart", |a| {
        let zero = Tensor::zeros(source.owners().shape(), a.dtype(), a.client().device())?;
        let scale = a.expansion_point(&zero)?;
        let parameter = a.point(&zero)?;
        let mut query =
            AreaQuery::source_affine(a, bands, source, AreaDirection::Add, &scale, &parameter)?;
        let rows = Tensor::empty(original_rows.shape(), DType::I64, a.client().device())?;
        a.client().copy_into(&rows, original_rows)?;
        query.original_rows = rows;
        query.prove_support(a, bands)
    })
}

pub(super) fn zero_value<R>(
    a: &mut Arithmetic<'_, R>,
    count: usize,
) -> HydraulicResult<PotentialValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = Tensor::zeros(&[count], DType::F64, a.client().device())?;
    Ok(PotentialValue {
        retained: a.expansion_point(&zero)?,
        physical: a.point(&zero)?,
        analytic: None,
    })
}

pub(super) fn select_expansion<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    Ok(a.expansion_bounds(
        &c.index_select(value.high(), 0, rows)?,
        &c.index_select(value.low(), 0, rows)?,
        &c.index_select(value.abs_error(), 0, rows)?,
    )?)
}

pub(super) fn scatter<R>(
    a: &mut Arithmetic<'_, R>,
    dense: PotentialValue<R>,
    rows: &Tensor<R>,
    value: &PotentialValue<R>,
) -> HydraulicResult<PotentialValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    Ok(PotentialValue {
        retained: a.expansion_bounds(
            &c.index_put(dense.retained.high(), 0, rows, value.retained.high())?,
            &c.index_put(dense.retained.low(), 0, rows, value.retained.low())?,
            &c.index_put(
                dense.retained.abs_error(),
                0,
                rows,
                value.retained.abs_error(),
            )?,
        )?,
        physical: a.bounds(
            &c.index_put(dense.physical.lower(), 0, rows, value.physical.lower())?,
            &c.index_put(dense.physical.upper(), 0, rows, value.physical.upper())?,
        )?,
        analytic: dense.analytic,
    })
}

pub(super) fn from_interval<R>(
    a: &mut Arithmetic<'_, R>,
    physical: Interval<R>,
) -> HydraulicResult<PotentialValue<R>>
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
    Ok(PotentialValue {
        retained: a.expansion_from_interval(&physical)?,
        physical,
        analytic: None,
    })
}
