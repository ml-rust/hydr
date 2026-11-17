//! Exact counter capacity boundaries on the configured tensor runtime.

use super::arithmetic::count;
use crate::structures::*;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

#[test]
fn exact_counter_capacity_preserves_last_increment_and_rejects_absorption()
-> Result<(), Box<dyn std::error::Error>> {
    let text = || "Explicit analytical fixture".to_string();
    let element = StructureElement {
        id: "gate".into(),
        endpoint_a: "a".into(),
        endpoint_b: "b".into(),
        datum: text(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            1.0,
            0.0,
            1.0,
            1.0,
            9.81,
            GateApplicability {
                coefficient_provenance: text(),
                hydrostatic_quiescent_pool_assertion: text(),
                free_submerged_aperture_applicability: text(),
                unobstructed_aperture_assertion: text(),
                symmetric_reversal: true,
            },
        )?),
    };
    let batch = StructureBatch::<CpuRuntime>::new(
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
        vec![element],
    )?
    .with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: "actuator".into(),
            structure_id: "gate".into(),
            opening_rate_m_per_s: 1.0,
            closing_rate_m_per_s: 1.0,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 0.0,
                maximum_relative_opening_error: 0.0,
                maximum_arrival_window_s: 0.0,
            },
            rate_provenance: text(),
            ideal_response_applicability: text(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: "actuator".into(),
            opening_m: 0.0,
            target_opening_m: 0.0,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: Vec::new(),
            end_s: 1.0,
        },
    )?;
    let ctx = &batch.context;
    let maximum = (1u64 << 53) as f64;
    assert_eq!(
        count(ctx, &ctx.constant(maximum - 1.0)?, &ctx.one)?.item::<f64>()?,
        maximum
    );
    assert_eq!(
        count(ctx, &ctx.constant(maximum)?, &ctx.zero)?.item::<f64>()?,
        maximum
    );
    assert!(matches!(
        count(ctx, &ctx.constant(maximum)?, &ctx.one),
        Err(StructureError::Numerical { .. })
    ));
    Ok(())
}
