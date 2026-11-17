use super::*;
use crate::structures::{
    DirectionPolicy, EndpointHead, GateApplicability, GateParameters, HeadConvention, HeadSupport,
    PumpApplicability, PumpCurve, PumpCurveKnot, PumpParameters, PumpSpeedPolicy, StructureHeads,
    StructureLaw, StructureOperatingPoint,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn fixture() -> Result<(CpuDevice, CouplingStructures<CpuRuntime>), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let gate = GateParameters::new(
        1.0,
        0.0,
        1.0,
        1.0,
        0.5,
        GateApplicability {
            coefficient_provenance: "Analytical unit coefficient".into(),
            hydrostatic_quiescent_pool_assertion: "Level analytical pools".into(),
            free_submerged_aperture_applicability: "Declared analytical aperture envelope".into(),
            unobstructed_aperture_assertion: "Complete unobstructed aperture".into(),
            symmetric_reversal: true,
        },
    )?;
    let pump = PumpParameters::new(
        PumpCurve::new(
            vec![
                PumpCurveKnot {
                    head_m: -4.0,
                    discharge_m3_per_s: 7.0,
                },
                PumpCurveKnot {
                    head_m: 3.0,
                    discharge_m3_per_s: 0.0,
                },
            ],
            "Analytical linear reference-speed curve".into(),
        )?,
        0.0,
        PumpSpeedPolicy::ReferenceOnly,
        PumpApplicability {
            endpoint_stage_head_assertion: "Common-datum quiescent analytical pools".into(),
            connection_losses_and_velocity_head_account: "Curve includes declared connections"
                .into(),
            stopped_isolation_assertion: "Explicit ideal isolation when stopped".into(),
            operating_envelope: "Wet suction and receiving pools within supplied curve".into(),
        },
    )?;
    let elements = vec![
        StructureElement {
            id: "gate".into(),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: "analytical".into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::HydrostaticRectangularGate(gate),
        },
        StructureElement {
            id: "pump".into(),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: "analytical".into(),
            direction: DirectionPolicy::AtoBOnly,
            law: StructureLaw::Pump(pump),
        },
    ];
    Ok((
        device,
        CouplingStructures::new(client, elements, operations())?,
    ))
}

fn operations() -> Vec<StructureOperation> {
    vec![
        StructureOperation {
            structure_id: "gate".into(),
            operating_point: StructureOperatingPoint::GateOpening { opening_m: 0.5 },
        },
        StructureOperation {
            structure_id: "pump".into(),
            operating_point: StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
        },
    ]
}

struct Fields {
    stage_a: Tensor<CpuRuntime>,
    stage_b: Tensor<CpuRuntime>,
    bed: Tensor<CpuRuntime>,
    wet: Tensor<CpuRuntime>,
}

impl Fields {
    fn new(device: &CpuDevice) -> numr::error::Result<Self> {
        Ok(Self {
            stage_a: Tensor::from_slice(&[4.0_f64; 2], &[2], device)?,
            stage_b: Tensor::from_slice(&[2.0_f64; 2], &[2], device)?,
            bed: Tensor::from_slice(&[0.0_f64; 2], &[2], device)?,
            wet: Tensor::from_slice(&[1.0_f64; 2], &[2], device)?,
        })
    }

    fn evaluate(
        &self,
        structures: &CouplingStructures<CpuRuntime>,
    ) -> Result<TensorStructureResult<CpuRuntime>, StructureError> {
        structures.evaluate_heads(
            &self.stage_a,
            &self.stage_b,
            &self.bed,
            &self.bed,
            &self.wet,
            &self.wet,
        )
    }
}

#[test]
fn cached_manual_operations_match_original_proposals() -> TestResult {
    let (device, structures) = fixture()?;
    let fields = Fields::new(&device)?;
    let operations = operations();
    let heads = structures
        .batch
        .elements()
        .iter()
        .map(|element| StructureHeads {
            structure_id: element.id.clone(),
            endpoint_a: EndpointHead {
                id: "a".into(),
                support: HeadSupport::Wet { stage_m: 4.0 },
            },
            endpoint_b: EndpointHead {
                id: "b".into(),
                support: HeadSupport::Wet { stage_m: 2.0 },
            },
            datum: "analytical".into(),
            convention: HeadConvention::QuiescentStage,
        })
        .collect::<Vec<_>>();
    let expected = structures
        .batch
        .propose_with_operations(&heads, &operations)?
        .outputs()?;
    let result = fields.evaluate(&structures)?;
    let original = result.discharge.try_to_vec::<f64>()?;
    for (actual, expected) in original.iter().zip(&expected) {
        assert_eq!(actual.to_bits(), expected.discharge_m3_per_s.to_bits());
    }
    let held = structures
        .held_volumes(&result, 0.25)?
        .try_to_vec::<f64>()?;
    for (actual, discharge) in held.iter().zip(&original) {
        assert_eq!(actual.to_bits(), (discharge * 0.25).to_bits());
    }

    let mut changed = Fields::new(&device)?;
    changed.stage_a = Tensor::from_slice(&[3.0_f64, 5.0], &[2], &device)?;
    changed.evaluate(&structures)?;
    assert_eq!(result.discharge.try_to_vec::<f64>()?, original);
    assert_eq!(
        fields
            .evaluate(&structures)?
            .discharge
            .try_to_vec::<f64>()?,
        original
    );

    let mut dry_heads = heads;
    dry_heads[0].endpoint_b.support = HeadSupport::Dry {
        bed_elevation_m: 0.0,
    };
    let expected_dry = structures
        .batch
        .propose_with_operations(&dry_heads, &operations)?
        .outputs()?;
    let dry_stage = Tensor::from_slice(&[f64::MAX, 2.0], &[2], &device)?;
    let dry_flags = Tensor::from_slice(&[0.0_f64, 1.0], &[2], &device)?;
    let dry_result = structures.evaluate_heads(
        &fields.stage_a,
        &dry_stage,
        &fields.bed,
        &fields.bed,
        &fields.wet,
        &dry_flags,
    )?;
    for (actual, expected) in dry_result
        .discharge
        .try_to_vec::<f64>()?
        .iter()
        .zip(&expected_dry)
    {
        assert_eq!(actual.to_bits(), expected.discharge_m3_per_s.to_bits());
    }
    Ok(())
}

#[test]
fn tensor_heads_require_complete_f64_binary_support() -> TestResult {
    let (device, structures) = fixture()?;
    let mut fields = Fields::new(&device)?;
    fields.stage_a = Tensor::from_slice(&[4.0_f64], &[1], &device)?;
    assert_eq!(
        fields
            .evaluate(&structures)
            .err()
            .and_then(|e| e.input_kind()),
        Some(StructureInputErrorKind::Identity)
    );
    fields.stage_a = Tensor::from_slice(&[4.0_f32; 2], &[2], &device)?;
    assert_eq!(
        fields
            .evaluate(&structures)
            .err()
            .and_then(|e| e.input_kind()),
        Some(StructureInputErrorKind::Definition)
    );
    fields.stage_a = Tensor::from_slice(&[4.0_f64; 2], &[2], &device)?;
    for value in [0.5_f64, -1.0, f64::NAN, f64::INFINITY] {
        fields.wet = Tensor::from_slice(&[value; 2], &[2], &device)?;
        assert_eq!(
            fields
                .evaluate(&structures)
                .err()
                .and_then(|e| e.input_kind()),
            Some(StructureInputErrorKind::Definition)
        );
    }
    Fields::new(&device)?.evaluate(&structures)?;
    Ok(())
}

#[test]
fn nonfinite_known_heads_and_invalid_durations_reject() -> TestResult {
    let (device, structures) = fixture()?;
    let mut fields = Fields::new(&device)?;
    fields.stage_a = Tensor::from_slice(&[f64::INFINITY, 4.0], &[2], &device)?;
    assert_eq!(
        fields
            .evaluate(&structures)
            .err()
            .and_then(|e| e.input_kind()),
        Some(StructureInputErrorKind::NonfiniteInput)
    );
    let mut fields = Fields::new(&device)?;
    fields.bed = Tensor::from_slice(&[0.0_f64, f64::NAN], &[2], &device)?;
    assert_eq!(
        fields
            .evaluate(&structures)
            .err()
            .and_then(|e| e.input_kind()),
        Some(StructureInputErrorKind::NonfiniteInput)
    );
    let result = Fields::new(&device)?.evaluate(&structures)?;
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            structures
                .held_volumes(&result, dt)
                .err()
                .and_then(|e| e.input_kind()),
            Some(StructureInputErrorKind::Definition)
        );
    }
    Ok(())
}
