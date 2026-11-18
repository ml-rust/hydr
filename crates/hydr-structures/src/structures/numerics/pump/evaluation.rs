//! Fixed-orientation running/shutoff/isolation proposals with immutable operating metadata.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::ScalarOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::structures::tensor::{
    TensorStructureContext, TensorStructureInput, TensorStructureResult, codes,
    pump::TensorPumpResult,
};

pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
    mut result: TensorStructureResult<R>,
) -> Result<TensorStructureResult<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>
        + solvr::interpolate::Interp1dAlgorithms<R>,
{
    let Some(pump) = &context.pump else {
        return Ok(result);
    };
    let operation = input.pump.as_ref().ok_or_else(|| {
        StructureError::numerical("checked pump proposal lacks its explicit operating tensors")
    })?;
    let c = &context.client;
    let (donor, receiver) = super::admission::heads(context, pump, input, operation)?;
    let normalized = super::normalization::evaluate(context, &donor, &receiver, operation)?;
    let super::curve::Curve {
        reference,
        zero_region,
        slope,
    } = super::curve::evaluate(context, pump, operation, &normalized)?;
    let positive = c.sub(&operation.running, &zero_region)?;
    let used_speed = c.where_cond(&positive, &operation.speed, &context.one)?;
    let used_reference = c.where_cond(&positive, &reference, &context.zero)?;
    let magnitude = c.mul(&used_speed, &used_reference)?;
    context.finite(
        &magnitude,
        "active final pump discharge overflows or underflows to zero",
    )?;
    context.reject(
        &c.mul(&positive, &c.le(&magnitude, &context.zero)?)?,
        "active final pump discharge overflows or underflows to zero",
    )?;
    let discharge = c.where_cond(
        &positive,
        &c.mul(&pump.orientation, &magnitude)?,
        &context.zero,
    )?;
    let status = c.add(
        &c.mul_scalar(&positive, codes::PUMP_RUNNING)?,
        &c.add(
            &c.mul_scalar(&zero_region, codes::PUMP_SHUTOFF)?,
            &c.mul_scalar(&operation.isolated, codes::PUMP_STOPPED)?,
        )?,
    )?;
    let regime = c.add(
        &c.mul_scalar(&operation.running, codes::PUMP_CURVE)?,
        &c.mul_scalar(&operation.isolated, codes::PUMP_ISOLATED)?,
    )?;
    // `Q = N Q_ref(Δ / N²)` with lift `Δ`, so `|dQ/dΔ| / Q = |dQ_ref/dh| / (N Q)`.
    // The donor head moves `Δ` one for one. The receiver head does too, except where a free
    // outlet above it sets the lift. There the receiver moves nothing, so the bound still holds.
    let sensitivity = c.where_cond(
        &positive,
        &c.div(
            &c.div(&slope, &used_speed)?,
            &c.where_cond(&positive, &magnitude, &context.one)?,
        )?,
        &context.zero,
    )?;
    result.head_sensitivity = c.where_cond(&pump.mask, &sensitivity, &result.head_sensitivity)?;
    result.discharge = c.where_cond(&pump.mask, &discharge, &result.discharge)?;
    result.status = c.where_cond(&pump.mask, &status, &result.status)?;
    result.regime = c.where_cond(&pump.mask, &regime, &result.regime)?;
    result.direction = c.where_cond(
        &pump.mask,
        &c.where_cond(&positive, &pump.orientation, &context.zero)?,
        &result.direction,
    )?;
    result.pump = Some(TensorPumpResult {
        head_high: normalized.high,
        head_low: normalized.low,
        reference_head: normalized.query,
        error_bound: normalized.bound,
        lower: normalized.lower,
        upper: normalized.upper,
        reference_discharge: reference,
        division_exact: normalized.division_exact,
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::super::sensitivity::fixture::{
        Lane, MARGIN, STEP, TestResult, check_bound, evaluate, grid, structures,
    };
    use crate::structures::{
        DirectionPolicy, PumpApplicability, PumpCurve, PumpCurveKnot, PumpFreeOutlet,
        PumpParameters, PumpSpeedPolicy, StructureLaw, StructureOperatingPoint,
    };

    /// Curve over `[-4, 3]` m: two falling segments, a flat one, and a fall to shutoff.
    fn pump() -> TestResult<StructureLaw> {
        let knots = [(-4.0, 7.0), (-1.0, 6.0), (1.0, 3.0), (2.0, 3.0), (3.0, 0.0)];
        Ok(StructureLaw::Pump(PumpParameters::new(
            PumpCurve::new(
                knots
                    .iter()
                    .map(|&(head_m, discharge_m3_per_s)| PumpCurveKnot {
                        head_m,
                        discharge_m3_per_s,
                    })
                    .collect(),
                "Analytical piecewise-linear curve".into(),
            )?,
            -100.0,
            PumpSpeedPolicy::DeclaredAffinity {
                minimum_ratio: 0.5,
                maximum_ratio: 2.0,
                applicability: "Analytical affinity scaling".into(),
            },
            PumpApplicability {
                endpoint_stage_head_assertion: "Common-datum quiescent pools".into(),
                connection_losses_and_velocity_head_account: "Curve includes connections".into(),
                stopped_isolation_assertion: "Ideal isolation when stopped".into(),
                operating_envelope: "Wet pools within the curve".into(),
            },
        )?))
    }

    #[test]
    fn pump_sensitivity_bounds_every_curve_segment_and_knot() -> TestResult {
        let knots = [-1.0, 1.0, 2.0];
        let mut cases = Vec::new();
        let mut speeds = Vec::new();
        for speed in [1.0, 0.5, 2.0, 0.8] {
            let square = speed * speed;
            // Normalized lifts inside the curve domain, then each interior knot exactly.
            let mut lifts: Vec<(f64, f64)> = grid(-3.99, 2.99, 60)
                .into_iter()
                .filter_map(|q| {
                    let ends = (q + 4.0).min(3.0 - q);
                    let distance = knots.iter().map(|k| (q - k).abs()).fold(ends, f64::min);
                    (distance >= MARGIN).then_some((q, distance))
                })
                .collect();
            // A power-of-two speed keeps a knot lift exact after normalization.
            if speed != 0.8 {
                lifts.extend(knots.iter().map(|&k| (k, 1.0)));
            }
            for (q, distance) in lifts {
                let donor = 2.5;
                cases.push(Lane {
                    stage: [donor, donor + q * square],
                    wet: [true, true],
                    step: STEP * distance * square,
                });
                speeds.push(speed);
            }
        }
        let law = pump()?;
        let structures = structures(
            |index| {
                (
                    law.clone(),
                    DirectionPolicy::AtoBOnly,
                    StructureOperatingPoint::PumpRunning {
                        speed_ratio: speeds[index],
                    },
                )
            },
            cases.len(),
        )?;
        let flowing = check_bound(&structures, &cases)?;
        assert_eq!(flowing, cases.len());
        Ok(())
    }

    /// The analytical pump with a free outlet at `elevation_m`.
    fn outlet_pump(elevation_m: f64) -> TestResult<StructureLaw> {
        let StructureLaw::Pump(parameters) = pump()? else {
            return Err("analytical law is not a pump".into());
        };
        Ok(StructureLaw::Pump(parameters.with_free_outlet(
            PumpFreeOutlet {
                elevation_m,
                evidence: "Analytical free outlet".into(),
            },
        )?))
    }

    #[test]
    fn outlet_sensitivity_bounds_the_donor_and_the_receiver_below_the_outlet_moves_nothing()
    -> TestResult {
        let outlet = 4.0;
        let knots = [-1.0, 1.0, 2.0];
        let mut cases = Vec::new();
        let mut speeds = Vec::new();
        for speed in [1.0, 0.5] {
            let square = speed * speed;
            for q in grid(-3.99, 2.99, 40) {
                let ends = (q + 4.0).min(3.0 - q);
                let distance = knots.iter().map(|k| (q - k).abs()).fold(ends, f64::min);
                if distance < MARGIN {
                    continue;
                }
                let donor = outlet - q * square;
                // A wet receiver one metre below the outlet, then a dry one on the fixture bed.
                for wet in [true, false] {
                    cases.push(Lane {
                        stage: [donor, outlet - 1.0],
                        wet: [true, wet],
                        step: STEP * distance * square,
                    });
                    speeds.push(speed);
                }
            }
        }
        let law = outlet_pump(outlet)?;
        let structures = structures(
            |index| {
                (
                    law.clone(),
                    DirectionPolicy::AtoBOnly,
                    StructureOperatingPoint::PumpRunning {
                        speed_ratio: speeds[index],
                    },
                )
            },
            cases.len(),
        )?;
        assert_eq!(check_bound(&structures, &cases)?, cases.len());
        let centre = evaluate(&structures, &cases, None)?;
        for sign in [1.0, -1.0] {
            let moved = evaluate(&structures, &cases, Some((1, sign)))?;
            for (index, (&a, &b)) in centre.discharge.iter().zip(&moved.discharge).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "lane {index}");
            }
        }
        Ok(())
    }

    #[test]
    fn pump_sensitivity_is_zero_at_shutoff_and_isolation() -> TestResult {
        let law = pump()?;
        let structures = structures(
            |index| {
                let operation = if index == 0 {
                    StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
                } else {
                    StructureOperatingPoint::PumpStoppedIsolated
                };
                (law.clone(), DirectionPolicy::AtoBOnly, operation)
            },
            2,
        )?;
        // Lift 3 m sits on the shutoff knot.
        let lane = Lane {
            stage: [1.0, 4.0],
            wet: [true, true],
            step: 0.0,
        };
        let sample = evaluate(&structures, &[lane, lane], None)?;
        assert_eq!(sample.discharge, [0.0, 0.0]);
        assert_eq!(sample.sensitivity, [0.0, 0.0]);
        Ok(())
    }
}
