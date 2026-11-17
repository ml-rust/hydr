use super::{advance, check_step};
use crate::runoff::tensor::reservoir_storage::{TensorReservoirContext, TensorReservoirState};
use crate::runoff::{
    ReservoirConfig, ReservoirElement, ReservoirError, ReservoirParameters, ReservoirState,
};
use numr::runtime::cpu::CpuRuntime;

fn element(id: &str, n: f64, width: f64) -> Result<ReservoirElement, ReservoirError> {
    Ok(ReservoirElement {
        id: id.into(),
        parameters: ReservoirParameters::from_native(1024.0, 1024.0, width, 1.0 / 64.0, n, 0.125)?,
    })
}

#[test]
fn mixed_branches_retain_independent_tensor_receipts() -> Result<(), ReservoirError> {
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![
            element("finite", 1.0 / 32.0, 32.0)?,
            element("closed", 1.0 / 32.0, 0.0)?,
            element("instant", 0.0, 32.0)?,
        ],
        ReservoirConfig::default(),
    )?;
    let state = TensorReservoirState::<CpuRuntime>::new(
        &context,
        &[ReservoirState {
            depth_m: 0.25 * 0.3048,
        }; 3],
    )?;
    let step = advance(&context, &state, 1.0, &context.tensor(&[0.0; 3])?)?;
    check_step(&context, &step, 1.0)?;
    let depths = step.depth.try_to_vec::<f64>()?;
    let exports = step.export_volume.try_to_vec::<f64>()?;
    assert!(depths[0] > 0.125 * 0.3048 && depths[0] < 0.25 * 0.3048);
    assert_eq!(depths[1], 0.25 * 0.3048);
    assert_eq!(depths[2], 0.125 * 0.3048);
    assert!(exports[0] > 0.0);
    assert_eq!(exports[1], 0.0);
    assert!(exports[2] > 0.0);
    assert_eq!(step.constitutive_residual.try_to_vec::<f64>()?[2], 0.0);
    assert_eq!(step.constitutive_scale.try_to_vec::<f64>()?[2], 0.0);
    assert_eq!(step.endpoint_rate.try_to_vec::<f64>()?[2], 0.0);
    assert_eq!(state.depth.try_to_vec::<f64>()?, vec![0.25 * 0.3048; 3]);
    Ok(())
}

#[test]
fn imported_receipt_rechecks_export_and_law_on_its_element() -> Result<(), ReservoirError> {
    let elements = (0..33)
        .map(|index| element(&format!("element-{index}"), 1.0 / 32.0, 32.0))
        .collect::<Result<Vec<_>, _>>()?;
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        elements,
        ReservoirConfig::default(),
    )?;
    let state = TensorReservoirState::<CpuRuntime>::new(
        &context,
        &vec![
            ReservoirState {
                depth_m: 0.25 * 0.3048
            };
            33
        ],
    )?;
    let step = advance(&context, &state, 1.0, &context.tensor(&[0.0; 33])?)?;
    check_step(&context, &step, 1.0)?;
    for law_field in [false, true] {
        let mut altered = step.clone();
        let mut values = if law_field {
            altered.law_volume.try_to_vec::<f64>()?
        } else {
            altered.export_volume.try_to_vec::<f64>()?
        };
        values[32] += 1.0;
        if law_field {
            altered.law_volume = context.tensor(&values)?;
        } else {
            altered.export_volume = context.tensor(&values)?;
        }
        let error = check_step(&context, &altered, 1.0).err().ok_or_else(|| {
            ReservoirError::numerical("receipt algebra must reject altered physical volumes")
        })?;
        assert_eq!(error.element(), Some(32));
    }
    Ok(())
}

#[test]
fn exhausted_batched_root_work_keeps_antecedent_tensor() -> Result<(), ReservoirError> {
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![element("finite", 1.0 / 32.0, 32.0)?],
        ReservoirConfig {
            root_iterations: 1,
            ..ReservoirConfig::default()
        },
    )?;
    let state = TensorReservoirState::<CpuRuntime>::new(
        &context,
        &[ReservoirState {
            depth_m: 0.25 * 0.3048,
        }],
    )?;
    let before = state.depth.item::<f64>()?.to_bits();
    let error = advance(&context, &state, 64.0, &context.tensor(&[0.0])?)
        .err()
        .ok_or_else(|| ReservoirError::numerical("one round cannot resolve the implicit root"))?;
    assert!(matches!(
        error,
        ReservoirError::Numerical { element: None, .. }
    ));
    assert!(error.to_string().contains("solvr bisect_batched"));
    assert!(error.to_string().contains("max_expansions=0"));
    assert_eq!(state.depth.item::<f64>()?.to_bits(), before);
    Ok(())
}

#[test]
fn positive_implicit_excess_cannot_collapse_into_depression() -> Result<(), ReservoirError> {
    let depression = 1e6f64;
    let parameters = ReservoirParameters::from_si(1.0, 1.0, 1.0, 1.0, 1e-8, depression)?;
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![ReservoirElement {
            id: "stiff".into(),
            parameters,
        }],
        ReservoirConfig::default(),
    )?;
    let state = TensorReservoirState::<CpuRuntime>::new(
        &context,
        &[ReservoirState {
            depth_m: f64::from_bits(depression.to_bits() + 1),
        }],
    )?;
    let error = advance(&context, &state, 1.0, &context.tensor(&[0.0])?)
        .err()
        .ok_or_else(|| {
            ReservoirError::numerical("represented implicit excess must remain positive")
        })?;
    assert_eq!(error.element(), Some(0));
    assert!(error.to_string().contains("represented implicit excess"));
    Ok(())
}

#[test]
fn closed_and_instantaneous_lanes_skip_event_and_root_work() -> Result<(), ReservoirError> {
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![
            element("closed", 1.0 / 32.0, 0.0)?,
            element("instant", 0.0, 32.0)?,
        ],
        ReservoirConfig {
            root_iterations: 1,
            ..ReservoirConfig::default()
        },
    )?;
    let state =
        TensorReservoirState::<CpuRuntime>::new(&context, &[ReservoirState { depth_m: 0.0 }; 2])?;
    let step = advance(
        &context,
        &state,
        256.0,
        &context.tensor(&[0.3048 / 1024.0; 2])?,
    )?;
    assert_eq!(step.root_iterations, 0);
    assert_eq!(step.active_duration.try_to_vec::<f64>()?, vec![0.0; 2]);
    assert_eq!(step.root_upper.try_to_vec::<f64>()?, vec![0.0; 2]);
    check_step(&context, &step, 256.0)?;
    Ok(())
}

#[test]
fn positive_supply_absorbed_by_storage_rounding_rejects() -> Result<(), ReservoirError> {
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![element("closed", 1.0 / 32.0, 0.0)?],
        ReservoirConfig::default(),
    )?;
    let state =
        TensorReservoirState::<CpuRuntime>::new(&context, &[ReservoirState { depth_m: 1e16 }])?;
    let error = advance(&context, &state, 1.0, &context.tensor(&[1e-20])?)
        .err()
        .ok_or_else(|| {
            ReservoirError::numerical("positive incoming storage must remain represented")
        })?;
    assert_eq!(error.element(), Some(0));
    assert!(error.to_string().contains("absorbed by storage rounding"));
    Ok(())
}

#[test]
fn provided_budget_below_initial_storage_uses_no_synthetic_source() -> Result<(), ReservoirError> {
    use super::drainage::{DrainageRoot, admit_drainage, drain, prepare_budget};
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![element("loss-budget", 1.0 / 32.0, 32.0)?],
        ReservoirConfig::default(),
    )?;
    let initial_depth = 0.5 * 0.3048;
    let available = 0.25 * 0.3048;
    assert!(available < initial_depth);
    let budget = prepare_budget(&context, context.tensor(&[available])?, 1.0, |_| {
        context.tensor(&[1.0])
    })?;
    let result = drain(&budget)?;
    assert_eq!(
        budget.available_depth().item::<f64>()?.to_bits(),
        available.to_bits()
    );
    assert!(result.depth.item::<f64>()? < available);
    assert!(result.export_volume.item::<f64>()? > 0.0);
    let checked = admit_drainage(
        &budget,
        DrainageRoot {
            lower: result.lower.clone(),
            upper: result.upper.clone(),
            depth: result.depth.clone(),
            iterations: result.iterations,
        },
    )?;
    assert_eq!(
        checked.export_volume.item::<f64>()?.to_bits(),
        result.export_volume.item::<f64>()?.to_bits()
    );
    Ok(())
}

#[test]
fn provided_inactive_budgets_need_no_event_rate_or_root_work() -> Result<(), ReservoirError> {
    use super::drainage::{drain, prepare_budget};
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![
            element("closed", 1.0 / 32.0, 0.0)?,
            element("instant", 0.0, 32.0)?,
            element("unfilled", 1.0 / 32.0, 32.0)?,
        ],
        ReservoirConfig {
            root_iterations: 1,
            ..ReservoirConfig::default()
        },
    )?;
    let available = [0.25 * 0.3048, 0.25 * 0.3048, 0.0625 * 0.3048];
    let budget = prepare_budget(&context, context.tensor(&available)?, 1.0, |_| {
        context.tensor(&[0.0; 3])
    })?;
    let result = drain(&budget)?;
    assert_eq!(result.iterations, 0);
    assert_eq!(
        result.depth.try_to_vec::<f64>()?,
        vec![available[0], 0.125 * 0.3048, available[2]]
    );
    assert_eq!(budget.active_duration().try_to_vec::<f64>()?, vec![0.0; 3]);
    Ok(())
}

#[test]
fn provided_budget_rejects_invalid_storage_and_active_duration() -> Result<(), ReservoirError> {
    use super::drainage::prepare_budget;
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![element("finite", 1.0 / 32.0, 32.0)?],
        ReservoirConfig::default(),
    )?;
    for duration in [0.0, -1.0, 2.0, f64::NAN, f64::INFINITY] {
        let error = prepare_budget(&context, context.tensor(&[0.25 * 0.3048])?, 1.0, |_| {
            context.tensor(&[duration])
        })
        .err()
        .ok_or_else(|| ReservoirError::numerical("invalid active drainage duration must reject"))?;
        assert_eq!(error.element(), Some(0));
    }
    for available in [-1.0, f64::NAN, f64::INFINITY] {
        let error = prepare_budget(&context, context.tensor(&[available])?, 1.0, |_| {
            context.tensor(&[0.0])
        })
        .err()
        .ok_or_else(|| ReservoirError::numerical("invalid authoritative storage must reject"))?;
        assert_eq!(error.element(), Some(0));
    }
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = prepare_budget(&context, context.tensor(&[0.25 * 0.3048])?, dt, |_| {
            context.tensor(&[1.0])
        })
        .err()
        .ok_or_else(|| ReservoirError::numerical("invalid drainage interval must reject"))?;
        assert!(matches!(error, ReservoirError::InvalidInput { .. }));
    }
    Ok(())
}

#[test]
fn provided_budget_retains_strict_reconstructed_excess_admission() -> Result<(), ReservoirError> {
    use super::drainage::{drain, prepare_budget};
    let depression = 1e6f64;
    let context = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![ReservoirElement {
            id: "stiff-budget".into(),
            parameters: ReservoirParameters::from_si(1.0, 1.0, 1.0, 1.0, 1e-8, depression)?,
        }],
        ReservoirConfig::default(),
    )?;
    let available = f64::from_bits(depression.to_bits() + 1);
    let budget = prepare_budget(&context, context.tensor(&[available])?, 1.0, |_| {
        context.tensor(&[1.0])
    })?;
    let error = drain(&budget)
        .err()
        .ok_or_else(|| ReservoirError::numerical("collapsed implicit excess must reject"))?;
    assert_eq!(error.element(), Some(0));
    assert!(error.to_string().contains("represented implicit excess"));
    Ok(())
}

#[test]
fn foreign_root_witness_uses_the_budget_construction_context() -> Result<(), ReservoirError> {
    use super::drainage::{DrainageRoot, admit_drainage, drain, prepare_budget};
    let context_a = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![element("same-identity", 1.0 / 32.0, 32.0)?],
        ReservoirConfig::default(),
    )?;
    let context_b = TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![element("same-identity", 1.0 / 32.0, 64.0)?],
        ReservoirConfig::default(),
    )?;
    let available = 0.25 * 0.3048;
    let budget_a = prepare_budget(&context_a, context_a.tensor(&[available])?, 1.0, |_| {
        context_a.tensor(&[1.0])
    })?;
    let budget_b = prepare_budget(&context_b, context_b.tensor(&[available])?, 1.0, |_| {
        context_b.tensor(&[1.0])
    })?;
    let from_a = drain(&budget_a)?;
    let error = admit_drainage(
        &budget_b,
        DrainageRoot {
            lower: from_a.lower,
            upper: from_a.upper,
            depth: from_a.depth,
            iterations: from_a.iterations,
        },
    )
    .err()
    .ok_or_else(|| {
        ReservoirError::numerical(
            "a physically different context must reject the foreign root witness",
        )
    })?;
    assert_eq!(error.element(), Some(0));
    assert!(error.to_string().contains("root signs"));
    let from_b = drain(&budget_b)?;
    assert!(from_b.export_volume.item::<f64>()? > 0.0);
    Ok(())
}

fn cpu_client() -> numr::runtime::cpu::CpuClient {
    numr::runtime::cpu::CpuClient::new(numr::runtime::cpu::CpuDevice::new())
        .with_parallelism(numr::runtime::cpu::ParallelismConfig::new(Some(1), None))
}
