//! Weir, orifice and gate discharge on configured runtime clients over a dense head grid.

#[allow(dead_code)]
#[path = "../gate_laws/fixture.rs"]
mod gates;
#[allow(dead_code)]
#[path = "../structure_laws/fixture.rs"]
mod laws;

use super::{Result, runtime};
use hydr::structures::{
    DirectionPolicy, GateParameters, StructureBatch, StructureElement, StructureHeads,
    StructureHydraulicOperands, StructureLaw, StructureOperatingPoint, StructureOperation,
    StructureRegime,
};

/// Log discharge of the hydr regression input for correctly rounded `exp`.
const REGRESSION_LOG_DISCHARGE: f64 = 0.15604306081525554;

/// Weir coefficient and length whose correctly rounded logarithms sum to that log discharge.
///
/// With a unit head above the crest, the free-weir log discharge is exactly this sum.
const REGRESSION_WEIR: (f64, f64) = (1.168876534782721, 0.9999999999999993);

/// Upstream stages per law, in metres. Each grid holds more lanes than one vector width.
const STAGES: usize = 64;

/// Gate opening of every gate lane, in metres. The sill sits at 0 m.
const OPENING_M: f64 = 0.5;

/// Every lane of the case: its element, heads and operation.
struct Lane {
    element: StructureElement,
    heads: StructureHeads,
    operation: StructureOperation,
}

fn lane(id: String, law: StructureLaw, high: f64, low: f64) -> Lane {
    let operation = match &law {
        StructureLaw::HydrostaticRectangularGate(_) => gates::opening(&id, OPENING_M),
        _ => StructureOperation {
            structure_id: id.clone(),
            operating_point: StructureOperatingPoint::Passive,
        },
    };
    Lane {
        element: laws::element(&id, law, DirectionPolicy::Bidirectional),
        heads: laws::heads(&id, high, low),
        operation,
    }
}

/// Free and submerged weir, orifice, and free, mixed and submerged gate lanes over dense heads.
///
/// The last lane is the regression weir, whose log discharge is `REGRESSION_LOG_DISCHARGE`.
fn dense_case() -> Result<Vec<Lane>> {
    let gate = GateParameters::new(1.0, 0.0, 1.0, 0.6, 9.81, gates::applicability())?;
    let mut lanes = Vec::new();
    for i in 0..STAGES {
        let step = i as f64;
        let weir_head = 0.01 + 0.03 * step;
        lanes.push(lane(
            format!("weir-free-{i}"),
            laws::weir(1.7, 2.0, 0.0)?,
            weir_head,
            -1.0,
        ));
        for fraction in [0.1, 0.5, 0.9, 0.999] {
            lanes.push(lane(
                format!("weir-submerged-{i}-{fraction}"),
                laws::weir(1.7, 2.0, 0.0)?,
                weir_head,
                weir_head * fraction,
            ));
        }
        for gap in [1e-6, 0.01, 0.3, 0.9] {
            let high = 0.05 * step;
            lanes.push(lane(
                format!("orifice-{i}-{gap}"),
                laws::orifice(0.6, 2.0, 9.81)?,
                high,
                high - gap,
            ));
        }
        let gate_head = 0.6 + 0.025 * step;
        for (regime, low) in [
            ("free", -0.5),
            ("mixed", 0.25),
            ("submerged", OPENING_M + (gate_head - OPENING_M) * 0.5),
        ] {
            lanes.push(lane(
                format!("gate-{regime}-{i}"),
                StructureLaw::HydrostaticRectangularGate(gate.clone()),
                gate_head,
                low,
            ));
        }
    }
    let (coefficient, length) = REGRESSION_WEIR;
    lanes.push(lane(
        "weir-regression".into(),
        laws::weir(coefficient, length, 0.0)?,
        1.0,
        -1.0,
    ));
    Ok(lanes)
}

/// Proposes the dense case on one client and returns every output.
macro_rules! structure_law_outputs {
    ($runtime:ty, $client:expr) => {{
        let lanes = dense_case()?;
        let heads: Vec<_> = lanes.iter().map(|l| l.heads.clone()).collect();
        let operations: Vec<_> = lanes.iter().map(|l| l.operation.clone()).collect();
        let batch = StructureBatch::<$runtime>::new(
            $client,
            lanes.into_iter().map(|l| l.element).collect(),
        )?;
        batch
            .propose_with_operations(&heads, &operations)?
            .outputs()?
    }};
}

#[test]
fn selected_cpu_client_evaluates_structure_laws_across_dense_heads() -> Result {
    let outputs = structure_law_outputs!(runtime::Runtime, runtime::client());
    let regimes: Vec<_> = outputs.iter().map(|o| o.regime).collect();
    for regime in [
        StructureRegime::FreeWeirOverflow,
        StructureRegime::SubmergedWeirOverflow,
        StructureRegime::SubmergedOrifice,
        StructureRegime::FreeGateAperture,
        StructureRegime::MixedGateAperture,
        StructureRegime::SubmergedGateAperture,
    ] {
        assert!(regimes.contains(&regime), "no lane reaches {regime:?}");
    }
    let last = outputs.last().ok_or("missing regression output")?;
    let Some(StructureHydraulicOperands::FreeWeir {
        log_discharge_magnitude,
        ..
    }) = last.hydraulic_operands
    else {
        return Err(format!("regression lane is not a free weir: {last:?}").into());
    };
    assert_eq!(
        log_discharge_magnitude.to_bits(),
        REGRESSION_LOG_DISCHARGE.to_bits()
    );
    assert_eq!(
        last.discharge_m3_per_s.to_bits(),
        numr::algorithm::correctly_rounded::exp(REGRESSION_LOG_DISCHARGE).to_bits()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_structure_laws_bit_for_bit_across_dense_heads() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = structure_law_outputs!(runtime::Runtime, runtime::client());
    let cuda = structure_law_outputs!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    super::same_bits(&cpu, &cuda)
}
