//! Shared complete-law point evaluation in passive, pump, gate, then culvert order.
use numr::{dtype::DType, runtime::Runtime};

use numr::ops::{ConditionalOps, ScalarOps};
use numr::tensor::Tensor;

use crate::structures::StructureError;
use crate::structures::tensor::{
    TensorStructureContext, TensorStructureInput, TensorStructureResult, codes,
};

pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
) -> Result<TensorStructureResult<R>, StructureError>
where
    R::Client: numr::ops::CumulativeOps<R>
        + numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>
        + solvr::interpolate::Interp1dAlgorithms<R>,
{
    let passive = super::evaluate(context, input)?;
    let result = super::pump::evaluate(context, input, passive)?;
    let result = super::gate::evaluate(context, input, result)?;
    let result = match &context.culvert {
        Some(culvert) => culvert.evaluate(context, input, result)?,
        None => result,
    };
    match &context.closed {
        Some(closed) => close(context, closed, result),
        None => Ok(result),
    }
}

/// Selects exact zero discharge on closed rows, ahead of every other regime.
///
/// Every other row keeps its bits. A closed row reports regime `Closed`, status `PhysicalZero`,
/// no direction, and zero head sensitivity.
fn close<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    closed: &Tensor<R>,
    mut result: TensorStructureResult<R>,
) -> Result<TensorStructureResult<R>, StructureError>
where
    R::Client: numr::ops::ConditionalOps<R> + numr::ops::ScalarOps<R>,
{
    let c = &context.client;
    let zero = &context.zero;
    result.discharge = c.where_cond(closed, zero, &result.discharge)?;
    result.regime = c.where_cond(
        closed,
        &c.mul_scalar(&context.one, codes::CLOSED)?,
        &result.regime,
    )?;
    result.status = c.where_cond(
        closed,
        &c.mul_scalar(&context.one, codes::PHYSICAL_ZERO)?,
        &result.status,
    )?;
    result.direction = c.where_cond(closed, zero, &result.direction)?;
    result.head_sensitivity = c.where_cond(closed, zero, &result.head_sensitivity)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use crate::structures::coupling::CouplingStructures;
    use crate::structures::numerics::sensitivity::fixture::{TestResult, column};
    use crate::structures::tensor::codes;
    use crate::structures::{
        DirectionPolicy, GateApplicability, GateParameters, OrificeApplicability,
        StructureCondition, StructureElement, StructureLaw, StructureOperatingPoint,
        StructureOperation, SubmergedOrificeParameters, SuppressedRectangularWeirParameters,
        WeirApplicability,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};

    fn laws() -> TestResult<[(StructureLaw, StructureOperatingPoint); 3]> {
        let text = || "Analytical closed fixture".to_string();
        Ok([
            (
                StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
                    2.0,
                    0.0,
                    1.7,
                    WeirApplicability {
                        coefficient_provenance: text(),
                        free_nappe_and_approach_assertion: text(),
                        submergence_correction_assertion: text(),
                        symmetric_reversal: true,
                    },
                )?),
                StructureOperatingPoint::Passive,
            ),
            (
                StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
                    1.0,
                    -1.0,
                    0.0,
                    0.6,
                    9.81,
                    OrificeApplicability {
                        coefficient_provenance: text(),
                        fully_submerged_applicability: text(),
                        symmetric_reversal: true,
                    },
                )?),
                StructureOperatingPoint::Passive,
            ),
            (
                StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                    1.0,
                    0.0,
                    1.0,
                    0.6,
                    9.81,
                    GateApplicability {
                        coefficient_provenance: text(),
                        hydrostatic_quiescent_pool_assertion: text(),
                        free_submerged_aperture_applicability: text(),
                        unobstructed_aperture_assertion: text(),
                        symmetric_reversal: true,
                    },
                )?),
                StructureOperatingPoint::GateOpening { opening_m: 0.5 },
            ),
        ])
    }

    #[test]
    fn closed_rows_carry_zero_discharge_and_sensitivity_beside_open_rows() -> TestResult {
        let mut elements = Vec::new();
        let mut operations = Vec::new();
        for (index, (law, operating_point)) in laws()?.into_iter().enumerate() {
            for closed in [false, true] {
                let id = format!("s{index}{closed}");
                elements.push(StructureElement {
                    id: id.clone(),
                    endpoint_a: "a".into(),
                    endpoint_b: "b".into(),
                    datum: "analytical".into(),
                    direction: DirectionPolicy::Bidirectional,
                    law: law.clone(),
                    condition: if closed {
                        StructureCondition::Closed {
                            evidence: "Inspection record".into(),
                        }
                    } else {
                        StructureCondition::AsBuilt
                    },
                });
                operations.push(StructureOperation {
                    structure_id: id,
                    operating_point,
                });
            }
        }
        let n = elements.len();
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let structures = CouplingStructures::new(client, elements, operations)?;
        let bed = column(vec![-2.0; n])?;
        let wet = column(vec![1.0; n])?;
        let result = structures.evaluate_heads(
            &column(vec![2.0; n])?,
            &column(vec![1.0; n])?,
            &bed,
            &bed,
            &wet,
            &wet,
        )?;
        let discharge = result.discharge.try_to_vec::<f64>()?;
        let sensitivity = result.head_sensitivity.try_to_vec::<f64>()?;
        let regime = result.regime.try_to_vec::<f64>()?;
        let status = result.status.try_to_vec::<f64>()?;
        let direction = result.direction.try_to_vec::<f64>()?;
        let volumes = structures.held_volumes(&result, 3.0)?.try_to_vec::<f64>()?;
        let lanes = discharge
            .iter()
            .zip(&sensitivity)
            .zip(&regime)
            .zip(&status)
            .zip(&direction)
            .zip(&volumes)
            .enumerate();
        for (lane, (((((q, s), r), status), d), v)) in lanes {
            if !lane.is_multiple_of(2) {
                for value in [q, s, d, v] {
                    assert_eq!(value.to_bits(), 0.0_f64.to_bits(), "lane {lane}");
                }
                assert_eq!(*r, codes::CLOSED, "lane {lane}");
                assert_eq!(*status, codes::PHYSICAL_ZERO, "lane {lane}");
            } else {
                assert!(*q > 0.0 && *s > 0.0, "lane {lane}: q={q}, s={s}");
                assert_eq!(*status, codes::ACTIVE, "lane {lane}");
            }
        }
        Ok(())
    }
}
