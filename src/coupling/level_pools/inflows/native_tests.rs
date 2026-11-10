use super::super::CoupledLevelPoolForcingInterval;
use super::super::forcing::BoundPoolForcing;
use super::tests::{binding, client, pools};
use super::*;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime, ParallelismConfig};
use numr::runtime::{Device, RuntimeClient};
use numr::tensor::Tensor;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn inputs() -> Vec<CoupledPoolInflow> {
    vec![
        CoupledPoolInflow {
            boundary_id: "second".into(),
            input_id: "input-second".into(),
            value: CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.75 },
        },
        CoupledPoolInflow {
            boundary_id: "first".into(),
            input_id: "input-first".into(),
            value: CoupledPoolInflowValue::Rate {
                rate_m3_per_s: 0.25,
            },
        },
    ]
}
fn bindings() -> Vec<CoupledPoolInflowBinding> {
    let first = binding("first");
    let mut second = binding("second");
    second.pool_id = "pool-1".into();
    vec![first, second]
}

#[test]
fn native_exact_binding_preserves_authoritative_operands_and_matches_literal_preparation()
-> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &bindings())?;
    let original = context.zero_state(&client)?;
    let quantum = 3.0 * f64::from_bits(1);
    let values = Tensor::<CpuRuntime>::from_slice(&[quantum, 0.75], &[2], client.device())?;
    let native = context.bind_interval_volumes(&client, &values)?;
    let literal = context.bind_inputs(
        &client,
        &[
            CoupledPoolInflow {
                boundary_id: "first".into(),
                input_id: "input-first".into(),
                value: CoupledPoolInflowValue::IntervalVolume { volume_m3: quantum },
            },
            CoupledPoolInflow {
                boundary_id: "second".into(),
                input_id: "input-second".into(),
                value: CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.75 },
            },
        ],
    )?;
    let prepared = context.prepare(&client, &native, &original, 2.0, 1)?;
    let expected = context.prepare(&client, &literal, &original, 2.0, 1)?;
    assert!(prepared.step.all_interval_volumes());
    assert_eq!(
        prepared
            .incoming
            .try_to_vec::<f64>()?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![quantum.to_bits(), 0.75f64.to_bits()]
    );
    assert!(prepared.incoming.device().is_same(client.device()));
    assert_eq!(
        context.project_step(&client, &prepared.step)?,
        context.project_step(&client, &expected.step)?
    );
    assert_eq!(
        context.project_state(&client, &prepared.next)?,
        context.project_state(&client, &expected.next)?
    );
    assert_eq!(
        context
            .project_state(&client, &original)?
            .iter()
            .map(|entry| entry.cumulative_volume_m3)
            .collect::<Vec<_>>(),
        vec![0.0, 0.0]
    );
    Ok(())
}

#[test]
fn native_binding_rejects_metadata_and_original_value_errors() -> TestResult {
    let client = client();
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &bindings())?;
    for shape in [vec![], vec![1], vec![2, 1], vec![3]] {
        let values = Tensor::<CpuRuntime>::zeros(&shape, DType::F64, client.device())?;
        assert!(context.bind_interval_volumes(&client, &values).is_err());
    }
    for dtype in [DType::F32, DType::I64] {
        let values = Tensor::<CpuRuntime>::zeros(&[2], dtype, client.device())?;
        assert!(context.bind_interval_volumes(&client, &values).is_err());
    }
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let values = Tensor::<CpuRuntime>::from_slice(&[0.0, value], &[2], client.device())?;
        assert!(context.bind_interval_volumes(&client, &values).is_err());
    }
    let empty = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[])?;
    let values = Tensor::<CpuRuntime>::zeros(&[0], DType::F64, client.device())?;
    assert!(empty.bind_interval_volumes(&client, &values).is_err());
    assert!(empty.bind_inputs(&client, &[]).is_ok());
    Ok(())
}

#[test]
fn bound_forcing_projection_preserves_public_record_order_and_destination_context() -> TestResult {
    let client = client();
    let destination_client = CpuClient::new(client.device().clone())
        .with_parallelism(ParallelismConfig::new(Some(1), Some(128)));
    let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &bindings())?;
    let destination =
        InflowContext::<CpuRuntime>::new(&destination_client, &pools(4.0)?, &bindings())?;
    let interval = CoupledLevelPoolForcingInterval {
        start_s: 0.0,
        end_s: 2.0,
        sources: None,
        inflows: inputs(),
    };
    let forcing = BoundPoolForcing {
        start_s: interval.start_s,
        end_s: interval.end_s,
        sources: None,
        inflows: context.bind_inputs(&client, &interval.inflows)?,
    };
    assert!(forcing.project(&client, &context)?.same_bits(&interval));
    let rebound = forcing.restore(&destination_client, &destination)?;
    assert!(
        rebound
            .project(&destination_client, &destination)?
            .same_bits(&interval)
    );
    let original = destination.zero_state(&destination_client)?;
    let mixed = destination.prepare(&destination_client, &rebound.inflows, &original, 2.0, 1)?;
    assert!(!mixed.step.all_interval_volumes());
    assert_eq!(destination_client.parallelism().chunk_size, Some(128));
    Ok(())
}
