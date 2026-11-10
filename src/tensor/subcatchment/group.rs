//! Immutable topological role maps and compact native-soil correspondence.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::runoff::{
    ReservoirElement, ReservoirError, Subarea, SubcatchmentConfig, SubcatchmentDefinition,
    SubcatchmentError,
};
use crate::soil::{ModifiedGreenAmptParameters, SoilError, admission::AdjustmentReduction};
use crate::tensor::reservoir_storage::TensorReservoirContext;

#[derive(Debug)]
pub(crate) struct TensorSubareaRoute<R: Runtime<DType = DType>> {
    pub source_indices: Tensor<R>,
    pub destination_indices: Tensor<R>,
}

#[derive(Debug)]
pub(crate) struct TensorSubareaGroup<R: Runtime<DType = DType>> {
    pub reservoir: TensorReservoirContext<R>,
    pub slots: Vec<usize>,
    pub indices: Tensor<R>,
    pub soil_slots: Vec<usize>,
    pub soil_indices: Tensor<R>,
    pub soil_parameters: Vec<ModifiedGreenAmptParameters>,
    pub area_bound: Tensor<R>,
    pub route_fraction: Tensor<R>,
    pub routes: [Option<TensorSubareaRoute<R>>; 3],
    pub adjustment_reduction: AdjustmentReduction,
}

pub(super) fn index_tensor<R>(
    device: &R::Device,
    values: &[usize],
) -> Result<Tensor<R>, SubcatchmentError>
where
    R: Runtime<DType = DType>,
{
    let values: Vec<i64> = values
        .iter()
        .map(|&v| i64::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| SubcatchmentError::input("component index exceeds I64 tensor indexing"))?;
    Ok(Tensor::<R>::from_slice(&values, &[values.len()], device)?)
}

impl<R> TensorSubareaGroup<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(super) fn new(
        client: R::Client,
        slots: Vec<usize>,
        definitions: &[SubcatchmentDefinition],
        config: &SubcatchmentConfig,
    ) -> Result<Option<Self>, SubcatchmentError> {
        let device = client.device().clone();
        if slots.is_empty() {
            return Ok(None);
        }
        let mut elements = Vec::with_capacity(slots.len());
        let mut soil_slots = Vec::new();
        let mut soil_parameters = Vec::new();
        let mut area_bound = Vec::with_capacity(slots.len());
        let mut fractions = Vec::with_capacity(slots.len());
        let mut sources: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
        let mut destinations: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
        for (lane, &slot) in slots.iter().enumerate() {
            let catchment = slot / 3;
            let area = Subarea::from_index(slot % 3)
                .ok_or_else(|| SubcatchmentError::input("invalid cached physical subarea"))?;
            let definition = definitions.get(catchment).ok_or_else(|| {
                SubcatchmentError::input("cached subarea exceeds definition support")
            })?;
            let p = &definition.parameters;
            let parameters = p.closure(area).ok_or_else(|| {
                SubcatchmentError::input("compact group includes absent support")
                    .at(catchment, Some(area))
            })?;
            elements.push(ReservoirElement {
                id: format!("{}/{:?}", definition.id, area),
                parameters,
            });
            area_bound.push(p.area_bound(area));
            let fraction = if p.receiver().is_some() {
                p.donor_fraction(area)
            } else {
                0.0
            };
            fractions.push(fraction);
            if area == Subarea::Pervious {
                soil_slots.push(lane);
                soil_parameters.push(p.soil_parameters().ok_or_else(|| {
                    SubcatchmentError::input("positive pervious group lacks soil parameters")
                        .at(catchment, Some(area))
                })?);
            }
            if fraction > 0.0 {
                let receiver = p.receiver().ok_or_else(|| {
                    SubcatchmentError::input("positive routed donor lacks a receiver")
                })?;
                sources[area.index()].push(lane);
                destinations[area.index()].push(catchment * 3 + receiver.index());
            }
        }
        let reservoir =
            TensorReservoirContext::<R>::new(client, elements, config.reservoir.clone())
                .map_err(|e| map_reservoir(&slots, e))?;
        let indices = index_tensor::<R>(&device, &slots)?;
        let soil_indices = index_tensor::<R>(&device, &soil_slots)?;
        let mut routes: [Option<TensorSubareaRoute<R>>; 3] = std::array::from_fn(|_| None);
        for i in 0..3 {
            if !sources[i].is_empty() {
                routes[i] = Some(TensorSubareaRoute {
                    source_indices: index_tensor::<R>(&device, &sources[i])?,
                    destination_indices: index_tensor::<R>(&device, &destinations[i])?,
                });
            }
        }
        let adjustment_reduction = AdjustmentReduction::new(slots.len())?;
        Ok(Some(Self {
            area_bound: reservoir.tensor(&area_bound)?,
            route_fraction: reservoir.tensor(&fractions)?,
            reservoir,
            slots,
            indices,
            soil_slots,
            soil_indices,
            soil_parameters,
            routes,
            adjustment_reduction,
        }))
    }
    pub(crate) fn map_reservoir_error(&self, error: ReservoirError) -> SubcatchmentError {
        map_reservoir(&self.slots, error)
    }
    pub(crate) fn map_soil_error(&self, error: SoilError) -> SubcatchmentError {
        let slot = error
            .cell()
            .and_then(|i| self.soil_slots.get(i))
            .and_then(|&i| self.slots.get(i))
            .copied();
        let error = SubcatchmentError::from(error);
        if let Some(slot) = slot {
            error.at(slot / 3, Subarea::from_index(slot % 3))
        } else {
            error
        }
    }
}

fn map_reservoir(slots: &[usize], error: ReservoirError) -> SubcatchmentError {
    let slot = error.element().and_then(|i| slots.get(i)).copied();
    let error = SubcatchmentError::from(error);
    if let Some(slot) = slot {
        error.at(slot / 3, Subarea::from_index(slot % 3))
    } else {
        error
    }
}
