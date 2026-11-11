use super::super::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelLateralStateData,
    ChannelReach, ChannelReachDefinition, RectangularChannelCell,
    forcing::{ChannelTensorLateralInput, ChannelTensorWaterAmount},
};
use super::*;
use crate::forcing::WaterAmount;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
}

fn definition(directions: &[ChannelLateralDirection]) -> Arc<ChannelReachDefinition> {
    Arc::new(ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "explicit lateral arithmetic fixture".into(),
        geometry: crate::channel::ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "cell".into(),
                length_m: 4.0,
                width_m: 2.0,
                bed_elevation_m: 0.0,
                maximum_depth_m: 1e307,
                manning_n: 0.0,
            }],
        },
        lateral_bindings: directions
            .iter()
            .enumerate()
            .map(|(index, &direction)| ChannelLateralBinding {
                boundary_id: format!("source-{index}"),
                cell_id: "cell".into(),
                direction,
                ownership_assertion: "selected disjoint arithmetic source".into(),
            })
            .collect(),
        config: ChannelConfig::default(),
    })
}

fn forcing(inputs: Vec<ChannelLateralInput>) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s: 0.0,
        end_s: 2.0,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: inputs,
    }
}

fn inflow(index: usize, volume: f64, velocity: f64) -> ChannelLateralInput {
    ChannelLateralInput::Inflow {
        boundary_id: format!("source-{index}"),
        amount: WaterAmount::IntervalVolume { volume_m3: volume },
        longitudinal_velocity_m_per_s: velocity,
    }
}

fn native_case<R: Runtime<DType = DType>>(client: R::Client) -> TestResult
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let source = LateralContext::<R>::new(
        &client,
        definition(&[
            ChannelLateralDirection::Inflow,
            ChannelLateralDirection::Withdrawal,
        ]),
    )?;
    let values = Tensor::<R>::from_slice(
        &[f64::NAN, 0.125, f64::INFINITY, 0.5],
        &[2, 2],
        client.device(),
    )?
    .narrow(1, 1, 1)?
    .squeeze(Some(1));
    assert!(!values.is_contiguous());
    let incoming = values.narrow(0, 0, 1)?;
    let withdrawal = values.narrow(0, 1, 1)?;
    let velocity = Tensor::<R>::from_slice(&[-1.0], &[1], client.device())?;
    let bound = source.bind_native(
        &client,
        0.0,
        2.0,
        crate::channel::ChannelTensorBoundary::Wall,
        crate::channel::ChannelTensorBoundary::Wall,
        vec![
            ChannelTensorLateralInput::Withdrawal {
                boundary_id: "source-1".into(),
                amount: ChannelTensorWaterAmount::Rate {
                    rate_m3_per_s: withdrawal,
                },
            },
            ChannelTensorLateralInput::Inflow {
                boundary_id: "source-0".into(),
                amount: ChannelTensorWaterAmount::IntervalVolume {
                    volume_m3: incoming,
                },
                longitudinal_velocity_m_per_s: velocity,
            },
        ],
    )?;
    let projected = bound.project(&client)?;
    assert_eq!(projected.lateral_inputs[0].boundary_id(), "source-1");
    assert_eq!(
        projected.lateral_inputs[0].amount(),
        WaterAmount::Rate { rate_m3_per_s: 0.5 }
    );
    assert_eq!(
        projected.lateral_inputs[1].amount(),
        WaterAmount::IntervalVolume { volume_m3: 0.125 }
    );
    client.copy_into(
        &values,
        &Tensor::<R>::from_slice(&[f64::NAN, f64::NAN], &[2], client.device())?,
    )?;
    assert_eq!(
        bound.lateral.original[1]
            .amount()
            .tensor()
            .item::<f64>()?
            .to_bits(),
        0.125_f64.to_bits()
    );
    let make = |value| Tensor::<R>::from_slice(&[value], &[1], client.device());
    let previous = source.initial_state(&client)?;
    let prepared = source.prepare(
        &client,
        &bound.lateral,
        &previous,
        &make(8.0)?,
        &make(3.0)?,
        1.0,
    )?;
    let application = source.apply_lateral_and_transport(
        &client,
        &make(8.0)?,
        &make(24.0)?,
        &make(0.0)?,
        &make(0.0)?,
        &prepared,
    )?;
    assert_eq!(application.volume.item::<f64>()?, 7.625);
    assert_eq!(application.momentum.item::<f64>()?, 22.375);
    let receipts =
        source.project_receipts(&client, &bound.lateral, &previous, &prepared.receipt)?;
    assert_eq!(receipts[0].volume_m3, 0.125);
    assert_eq!(receipts[0].momentum_m4_per_s, -0.125);
    assert_eq!(receipts[1].volume_m3, 0.5);
    assert_eq!(receipts[1].momentum_m4_per_s, -1.5);
    assert_eq!(
        previous.cumulative_volume.try_to_vec::<f64>()?,
        vec![0.0, 0.0]
    );
    let reach = ChannelReach::<R>::new(
        client.clone(),
        (*source.definition).clone(),
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![8.0],
            longitudinal_momentum_m4_per_s: vec![24.0],
        },
    )?;
    let endpoint = ChannelBoundary::Exterior {
        depth_m: 1.0,
        velocity_m_per_s: 3.0,
    };
    let native = reach.context.lateral.bind_native(
        &client,
        0.0,
        0.01,
        crate::channel::ChannelTensorBoundary::from_host(&client, &endpoint)?,
        crate::channel::ChannelTensorBoundary::from_host(&client, &endpoint)?,
        bound.lateral.original.clone(),
    )?;
    let advanced = reach.step_bound(0.01, &native)?.next;
    let state = advanced.state()?;
    assert!((state[0].volume_m3 - 8.12).abs() <= 1e-12);
    assert!((state[0].longitudinal_momentum_m4_per_s - 23.86).abs() <= 1e-12);
    let data = advanced.snapshot_data()?;
    let last = data
        .last_interval
        .as_ref()
        .ok_or("missing native final interval")?;
    assert_eq!(last.diagnostics.retries, 0);
    assert_eq!(last.diagnostics.end_s.to_bits(), 0.01_f64.to_bits());
    assert!(
        last.forcing
            .lateral_inputs
            .iter()
            .zip(&projected.lateral_inputs)
            .all(|(a, b)| a.same_bits(b))
    );
    let cold = reach.import_snapshot(&data)?;
    assert_eq!(cold.snapshot_data()?, data);
    Ok(())
}

#[test]
fn native_cpu_sources_retain_strides_input_order_and_momentum() -> TestResult {
    native_case::<CpuRuntime>(client())
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_sources_retain_strides_input_order_and_momentum() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    native_case::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[test]
fn integrated_operands_never_enter_unused_rate_multiplication() -> TestResult {
    let client = client();
    let source =
        LateralContext::<CpuRuntime>::new(&client, definition(&[ChannelLateralDirection::Inflow]))?;
    let bound = source.bind_host(&client, &forcing(vec![inflow(0, f64::MAX, 0.0)]))?;
    let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
    let prepared = source.prepare(
        &client,
        &bound.lateral,
        &source.initial_state(&client)?,
        &make(1.0)?,
        &make(0.0)?,
        2.0,
    )?;
    assert_eq!(
        prepared.receipt.volume.item::<f64>()?.to_bits(),
        f64::MAX.to_bits()
    );
    assert_eq!(prepared.incoming_volume.item::<f64>()?, f64::MAX);
    assert_eq!(prepared.incoming_momentum.item::<f64>()?, 0.0);
    Ok(())
}

#[test]
fn positive_water_absorption_rejects_grouping_and_receiving_storage() -> TestResult {
    let client = client();
    let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
    let source = LateralContext::<CpuRuntime>::new(
        &client,
        definition(&[ChannelLateralDirection::Inflow; 2]),
    )?;
    let bound = source.bind_host(
        &client,
        &forcing(vec![inflow(0, 2.0_f64.powi(53), 0.0), inflow(1, 1.0, 0.0)]),
    )?;
    assert!(matches!(
        source.prepare(
            &client,
            &bound.lateral,
            &source.initial_state(&client)?,
            &make(1.0)?,
            &make(0.0)?,
            1.0
        ),
        Err(super::super::ChannelError::Numerical { .. })
    ));
    let source =
        LateralContext::<CpuRuntime>::new(&client, definition(&[ChannelLateralDirection::Inflow]))?;
    let bound = source.bind_host(&client, &forcing(vec![inflow(0, 1.0, 0.0)]))?;
    let prepared = source.prepare(
        &client,
        &bound.lateral,
        &source.initial_state(&client)?,
        &make(2.0_f64.powi(53))?,
        &make(0.0)?,
        1.0,
    )?;
    assert!(matches!(
        source.apply_lateral_and_transport(
            &client,
            &make(2.0_f64.powi(53))?,
            &make(0.0)?,
            &make(0.0)?,
            &make(0.0)?,
            &prepared
        ),
        Err(super::super::ChannelError::Numerical { .. })
    ));
    Ok(())
}

#[test]
fn signed_momentum_cancellation_retains_an_absolute_bound() -> TestResult {
    let client = client();
    let source = LateralContext::<CpuRuntime>::new(
        &client,
        definition(&[ChannelLateralDirection::Inflow; 3]),
    )?;
    let inputs = [2.0_f64.powi(53), 1.0, -2.0_f64.powi(53)]
        .into_iter()
        .enumerate()
        .map(|(i, u)| inflow(i, 1.0, u))
        .collect();
    let bound = source.bind_host(&client, &forcing(inputs))?;
    let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
    let prepared = source.prepare(
        &client,
        &bound.lateral,
        &source.initial_state(&client)?,
        &make(1.0)?,
        &make(0.0)?,
        1.0,
    )?;
    assert_eq!(prepared.incoming_volume.item::<f64>()?, 3.0);
    assert_eq!(prepared.incoming_momentum.item::<f64>()?, 0.0);
    assert!(prepared.incoming_momentum_bound.item::<f64>()? >= 1.0);
    Ok(())
}

#[test]
fn counter_addition_retains_low_components_and_dry_zero_velocity() -> TestResult {
    let client = client();
    let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], client.device());
    let source =
        LateralContext::<CpuRuntime>::new(&client, definition(&[ChannelLateralDirection::Inflow]))?;
    let previous = source.restore_state(
        &client,
        &ChannelLateralStateData {
            cumulative_volume_m3: vec![2.0_f64.powi(53)],
            counter_roundoff_bound_m3: vec![0.0],
            cumulative_assembly_bound_m3: 0.0,
        },
    )?;
    let bound = source.bind_host(&client, &forcing(vec![inflow(0, 1.0, 0.0)]))?;
    let prepared = source.prepare(
        &client,
        &bound.lateral,
        &previous,
        &make(1.0)?,
        &make(0.0)?,
        1.0,
    )?;
    assert_eq!(prepared.receipt.counter_addition_low.item::<f64>()?, 1.0);
    assert_eq!(
        prepared.receipt.next_counter.item::<f64>()?,
        2.0_f64.powi(53)
    );
    assert!(prepared.receipt.next_counter_bound.item::<f64>()? >= 1.0);
    let source = LateralContext::<CpuRuntime>::new(
        &client,
        definition(&[ChannelLateralDirection::Withdrawal]),
    )?;
    let bound = source.bind_host(
        &client,
        &forcing(vec![ChannelLateralInput::Withdrawal {
            boundary_id: "source-0".into(),
            amount: WaterAmount::IntervalVolume { volume_m3: 0.0 },
        }]),
    )?;
    let previous = source.initial_state(&client)?;
    let prepared = source.prepare(
        &client,
        &bound.lateral,
        &previous,
        &make(0.0)?,
        &make(0.0)?,
        1.0,
    )?;
    let receipts =
        source.project_receipts(&client, &bound.lateral, &previous, &prepared.receipt)?;
    assert_eq!(receipts[0].applied_longitudinal_velocity_m_per_s, None);
    Ok(())
}
