//! One-time projections of nested receiver, drainage, and partition tensors.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::{
    SubcatchmentBridgeReceipt, SubcatchmentDrainageReceipt, SubcatchmentError,
    SubcatchmentPartitionReceipt,
};
use crate::runoff::numerics::subcatchment::TensorSubareaStep;
use crate::runoff::tensor::subcatchment::TensorSubareaGroup;

pub(super) fn read<R>(tensor: &Tensor<R>, count: usize) -> Result<Vec<f64>, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let values = tensor.try_to_vec::<f64>()?;
    if values.len() != count {
        return Err(SubcatchmentError::numerical(
            "projected tensor support differs from immutable phase support",
        ));
    }
    Ok(values)
}

pub(super) fn bridges<R>(
    step: &TensorSubareaStep<R>,
    n: usize,
) -> Result<Vec<SubcatchmentBridgeReceipt>, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let b = &step.bridge;
    let fields = [
        &step.merged.volume,
        &step.merged.bound,
        &b.received_depth,
        &b.received_rate,
        &b.total_rate,
        &b.input_depth,
        &b.available_depth,
        &b.rain_depth,
        &b.rain_only_available,
        &b.depth_increment,
        &b.represented_received_volume,
        &b.residual,
        &b.residual_bound,
        &b.uncertainty,
    ];
    let a = fields
        .into_iter()
        .map(|f| read(f, n))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((0..n)
        .map(|i| SubcatchmentBridgeReceipt {
            merged_volume_m3: a[0][i],
            merge_bound_m3: a[1][i],
            received_depth_m: a[2][i],
            received_rate_m_per_s: a[3][i],
            total_rate_m_per_s: a[4][i],
            input_depth_m: a[5][i],
            available_depth_m: a[6][i],
            rain_depth_m: a[7][i],
            rain_only_available_m: a[8][i],
            depth_increment_m: a[9][i],
            represented_received_volume_m3: a[10][i],
            residual_m3: a[11][i],
            residual_bound_m3: a[12][i],
            uncertainty_m3: a[13][i],
        })
        .collect())
}

pub(super) fn drainage<R>(
    group: &TensorSubareaGroup<R>,
    step: &TensorSubareaStep<R>,
) -> Result<Vec<SubcatchmentDrainageReceipt>, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let d = &step.drainage;
    let n = group.slots.len();
    let fields = [
        &d.lower,
        &d.upper,
        &d.depth,
        &d.export_depth,
        &d.storage_after,
        &d.export_volume,
        &d.endpoint_rate,
        &d.endpoint_discharge,
        &d.law_depth,
        &d.law_volume,
        &d.constitutive_residual,
        &d.constitutive_scale,
    ];
    let a = fields
        .into_iter()
        .map(|f| read(f, n))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(group
        .reservoir
        .elements
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let finite = !e.parameters.is_instantaneous();
            SubcatchmentDrainageReceipt {
                root_lower_m: a[0][i],
                root_upper_m: a[1][i],
                depth_m: a[2][i],
                export_depth_m: a[3][i],
                storage_after_m3: a[4][i],
                export_volume_m3: a[5][i],
                endpoint_rate_m_per_s: finite.then_some(a[6][i]),
                endpoint_discharge_m3_per_s: finite.then_some(a[7][i]),
                law_depth_m: a[8][i],
                law_volume_m3: a[9][i],
                constitutive_residual_m3: a[10][i],
                constitutive_scale_m3: a[11][i],
            }
        })
        .collect())
}

pub(super) fn partitions<R>(
    group: &TensorSubareaGroup<R>,
    step: &TensorSubareaStep<R>,
) -> Result<Vec<SubcatchmentPartitionReceipt>, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let n = group.slots.len();
    let p = &step.partition;
    let fields = [
        &group.route_fraction,
        &p.routed,
        &p.outlet,
        &p.residual,
        &p.bound,
    ];
    let a = fields
        .into_iter()
        .map(|f| read(f, n))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((0..n)
        .map(|i| SubcatchmentPartitionReceipt {
            fraction: a[0][i],
            routed_volume_m3: a[1][i],
            outlet_volume_m3: a[2][i],
            residual_m3: a[3][i],
            bound_m3: a[4][i],
        })
        .collect())
}
