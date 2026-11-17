use super::*;
use crate::storage::{LevelPoolCurve, LevelPoolElement, LevelPoolKnot};
use numr::dtype::DType;
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

type TestResult = Result<(), Box<dyn std::error::Error>>;

pub(super) fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), Some(64)))
}
pub(super) fn pools(
    capacity: f64,
) -> Result<Vec<LevelPoolElement>, crate::storage::LevelPoolError> {
    (0..2)
        .map(|index| {
            Ok(LevelPoolElement {
                id: format!("pool-{index}"),
                curve: LevelPoolCurve::new(
                    vec![
                        LevelPoolKnot {
                            volume_m3: 0.0,
                            stage_m: 0.0,
                        },
                        LevelPoolKnot {
                            volume_m3: capacity,
                            stage_m: capacity,
                        },
                    ],
                    "analytic datum".into(),
                    "unit area".into(),
                )?,
                connected_pool_assertion: "one connected level pool".into(),
                storage_support_identity: format!("storage-{index}"),
            })
        })
        .collect()
}
pub(super) fn binding(boundary: &str) -> CoupledPoolInflowBinding {
    CoupledPoolInflowBinding {
        boundary_id: boundary.into(),
        pool_id: "pool-0".into(),
        input_id: format!("input-{boundary}"),
        physical_support_id: "shared receiving aperture".into(),
        boundary_assertion: "prescribed external flow enters pool-0".into(),
        ownership_assertion:
            "independent selected input excludes represented water and local rainfall".into(),
    }
}
fn rate(boundary: &str, value: f64) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: boundary.into(),
        input_id: format!("input-{boundary}"),
        value: CoupledPoolInflowValue::Rate {
            rate_m3_per_s: value,
        },
    }
}
fn ledger(value: f64, bound: f64) -> CoupledPoolInflowLedger {
    CoupledPoolInflowLedger {
        boundary_id: "boundary".into(),
        input_id: "input-boundary".into(),
        cumulative_volume_m3: value,
        roundoff_bound_m3: bound,
    }
}

#[test]
fn empty_and_explicit_zero_inflows_preserve_incoming_bits() -> TestResult {
    let client = client();
    for bindings in [vec![], vec![binding("boundary")]] {
        let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &bindings)?;
        let rates = if bindings.is_empty() {
            vec![]
        } else {
            vec![rate("boundary", 0.0)]
        };
        let rates = context.bind_inputs(&client, &rates)?;
        let original = context.zero_state(&client)?;
        let prepared = context.prepare(&client, &rates, &original, 1.0, 1)?;
        let incoming = Tensor::<CpuRuntime>::from_slice(&[-0.0f64, 1.0], &[2], client.device())?;
        let merged = context.merge_incoming(&client, &incoming, &prepared)?;
        assert_eq!(
            merged
                .volume
                .try_to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            vec![(-0.0f64).to_bits(), 1.0f64.to_bits()]
        );
        assert_eq!(merged.roundoff_bound_m3, 0.0);
        assert_eq!(prepared.summary.volume_m3, 0.0);
        assert_eq!(prepared.summary.reduction_bound_m3, 0.0);
        assert_eq!(
            context.project_state(&client, &original)?,
            context.project_state(&client, &prepared.next)?
        );
    }
    Ok(())
}

#[test]
fn original_interval_volume_survives_cumulative_counter_absorption() -> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
    let original = context.import_state(&client, &[ledger(1.0, 0.0)], 1)?;
    let value = f64::EPSILON / 4.0;
    let rates = context.bind_inputs(&client, &[rate("boundary", value)])?;
    let prepared = context.prepare(&client, &rates, &original, 1.0, 2)?;
    assert_eq!(prepared.summary.volume_m3, value);
    assert!(prepared.summary.counter_roundoff_bound_m3 >= value);
    let receipt = context
        .project_step(&client, &prepared.step)?
        .pop()
        .ok_or("missing external receipt")?;
    assert_eq!(receipt.volume_m3, value);
    assert_eq!(receipt.addition_witness_m3, value);
    assert_eq!(receipt.cumulative_before_m3, 1.0);
    assert_eq!(receipt.cumulative_after_m3, 1.0);
    assert!(receipt.bound_after_m3 >= value);
    let incoming = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0], &[2], client.device())?;
    assert!(
        context
            .merge_incoming(&client, &incoming, &prepared)
            .is_err()
    );
    assert_eq!(
        context.project_state(&client, &original)?,
        vec![ledger(1.0, 0.0)]
    );
    Ok(())
}

#[test]
fn grouped_boundaries_reject_absorbed_positive_volumes() -> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(
        &client,
        &pools(4.0)?,
        &[binding("large"), binding("small")],
    )?;
    let rates = context.bind_inputs(
        &client,
        &[rate("small", f64::EPSILON / 4.0), rate("large", 1.0)],
    )?;
    let original = context.zero_state(&client)?;
    assert!(context.prepare(&client, &rates, &original, 1.0, 1).is_err());
    assert!(
        context
            .project_state(&client, &original)?
            .iter()
            .all(|entry| entry.cumulative_volume_m3 == 0.0 && entry.roundoff_bound_m3 == 0.0)
    );
    Ok(())
}

#[test]
fn finite_capacity_ceiling_remains_active_beyond_exact_integer_counts() -> TestResult {
    if usize::BITS <= 53 {
        return Ok(());
    }
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
    let steps = usize::MAX;
    let finite_ceiling = 4.0 * steps as f64;
    assert!(
        context
            .import_state(&client, &[ledger(finite_ceiling, 0.0)], steps)
            .is_ok()
    );
    assert!(
        context
            .import_state(&client, &[ledger(2.0 * finite_ceiling, 0.0)], steps)
            .is_err()
    );
    assert!(
        context
            .import_state(&client, &[ledger(1.0, 0.0)], 0)
            .is_err()
    );
    assert!(
        context
            .import_state(&client, &[ledger(0.0, 1e-20)], 1)
            .is_err()
    );
    assert!(
        context
            .import_state(&client, &[ledger(-0.0, 0.0)], 0)
            .is_err()
    );
    Ok(())
}

#[test]
fn theoretical_capacity_overflow_keeps_finite_counters_admissible() -> TestResult {
    let client = client();
    let context =
        InflowContext::<CpuRuntime>::new(&client, &pools(1e308)?, &[binding("boundary")])?;
    assert!(
        context
            .import_state(&client, &[ledger(f64::MAX, 0.0)], 2)
            .is_ok()
    );
    assert!(
        context
            .import_state(&client, &[ledger(f64::INFINITY, 0.0)], 2)
            .is_err()
    );
    Ok(())
}

#[test]
fn original_products_reject_positive_underflow_and_overflow() -> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
    let original = context.zero_state(&client)?;
    for (value, duration) in [(f64::from_bits(1), 0.25), (f64::MAX, 2.0)] {
        let rates = context.bind_inputs(&client, &[rate("boundary", value)])?;
        assert!(
            context
                .prepare(&client, &rates, &original, duration, 1)
                .is_err()
        );
    }
    let rates = context.bind_inputs(&client, &[rate("boundary", 0.25)])?;
    let prepared = context.prepare(&client, &rates, &original, 2.0, 1)?;
    assert_eq!(prepared.incoming.dtype(), DType::F64);
    assert_eq!(prepared.incoming.try_to_vec::<f64>()?, vec![0.5, 0.0]);
    Ok(())
}

fn volume(boundary: &str, value: f64) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: boundary.into(),
        input_id: format!("input-{boundary}"),
        value: CoupledPoolInflowValue::IntervalVolume { volume_m3: value },
    }
}

#[test]
fn authoritative_subnormal_volume_never_passes_through_a_rate() -> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
    let original = context.zero_state(&client)?;
    let quantum = f64::from_bits(1);
    assert_eq!(quantum / 2.0, 0.0);
    let inputs = context.bind_inputs(&client, &[volume("boundary", quantum)])?;
    assert!(inputs.has_interval_volumes());
    let prepared = context.prepare(&client, &inputs, &original, 2.0, 1)?;
    assert_eq!(
        prepared
            .incoming
            .try_to_vec::<f64>()?
            .first()
            .map(|value| value.to_bits()),
        Some(quantum.to_bits())
    );
    let receipt = context
        .project_step(&client, &prepared.step)?
        .pop()
        .ok_or("missing interval-volume receipt")?;
    assert!(
        receipt
            .input
            .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: quantum })
    );
    assert_eq!(receipt.volume_m3.to_bits(), quantum.to_bits());
    assert_eq!(receipt.cumulative_after_m3.to_bits(), quantum.to_bits());
    Ok(())
}

#[test]
fn integrated_lanes_are_masked_before_unused_rate_multiplication() -> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
    let original = context.zero_state(&client)?;
    let inputs = context.bind_inputs(&client, &[volume("boundary", 3.0)])?;
    let prepared = context.prepare(&client, &inputs, &original, f64::MAX, 1)?;
    assert_eq!(prepared.incoming.try_to_vec::<f64>()?, vec![3.0, 0.0]);
    let rate_inputs = context.bind_inputs(&client, &[rate("boundary", 3.0)])?;
    assert!(!rate_inputs.has_interval_volumes());
    assert!(
        context
            .prepare(&client, &rate_inputs, &original, f64::MAX, 1)
            .is_err()
    );
    Ok(())
}

#[test]
fn mixed_inputs_preserve_original_variants_and_mask_underflow_checks() -> TestResult {
    let client = client();
    let mut receiving_rate = binding("rate");
    receiving_rate.pool_id = "pool-1".into();
    let context = InflowContext::<CpuRuntime>::new(
        &client,
        &pools(4.0)?,
        &[binding("volume"), receiving_rate],
    )?;
    let original = context.zero_state(&client)?;
    let quantum = f64::from_bits(1);
    let inputs = context.bind_inputs(&client, &[rate("rate", 0.25), volume("volume", quantum)])?;
    let prepared = context.prepare(&client, &inputs, &original, 2.0, 1)?;
    assert_eq!(prepared.incoming.try_to_vec::<f64>()?, vec![quantum, 0.5]);
    let receipts = context.project_step(&client, &prepared.step)?;
    let first = receipts.first().ok_or("missing integrated receipt")?;
    assert!(
        first
            .input
            .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: quantum })
    );
    let mut altered = first.clone();
    altered.input = CoupledPoolInflowValue::Rate {
        rate_m3_per_s: quantum,
    };
    assert!(!first.same_bits(&altered));
    let last = receipts.last().ok_or("missing rate receipt")?;
    assert!(last.input.same_bits(&CoupledPoolInflowValue::Rate {
        rate_m3_per_s: 0.25
    }));
    let underflow =
        context.bind_inputs(&client, &[volume("volume", quantum), rate("rate", quantum)])?;
    assert!(
        context
            .prepare(&client, &underflow, &original, 0.25, 1)
            .is_err()
    );
    Ok(())
}

#[test]
fn zero_integrated_input_keeps_its_kind_and_original_operand_bits() -> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[binding("boundary")])?;
    let original = context.zero_state(&client)?;
    let inputs = context.bind_inputs(&client, &[volume("boundary", -0.0)])?;
    assert!(inputs.has_interval_volumes());
    let prepared = context.prepare(&client, &inputs, &original, 1.0, 1)?;
    let receipt = context
        .project_step(&client, &prepared.step)?
        .pop()
        .ok_or("missing zero-volume receipt")?;
    assert!(
        receipt
            .input
            .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: -0.0 })
    );
    assert!(
        !receipt
            .input
            .same_bits(&CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.0 })
    );
    assert!(!receipt.input.same_bits(&CoupledPoolInflowValue::Rate {
        rate_m3_per_s: -0.0
    }));
    assert_eq!(receipt.volume_m3.to_bits(), (-0.0f64).to_bits());
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            context
                .bind_inputs(&client, &[volume("boundary", invalid)])
                .is_err()
        );
    }
    assert_eq!(
        context.project_state(&client, &original)?,
        vec![ledger(0.0, 0.0)]
    );
    Ok(())
}
