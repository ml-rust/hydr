//! Piecewise-linear and step evaluation of structure geometry schedules.

use super::knot::{
    DiscreteGeometryKnot, ScalarGeometryParameter, StructureGeometryKnot, StructureGeometryTrack,
};
use super::schedule::StructureGeometrySchedule;
use crate::structures::blockage::effective;
use crate::structures::{
    BlockageFraction, CulvertBlockage, CulvertSediment, GateParameters, StructureBlockage,
    StructureCondition, StructureElement, StructureError, StructureLaw, SubmergedOrificeParameters,
    SuppressedRectangularWeirParameters,
};

/// Evaluates a scalar track at `time_s`. When `time_s` equals a knot time, `after` selects
/// `knot.right` (`true`) or `knot.left` (`false`).
pub(crate) fn eval_scalar(knots: &[StructureGeometryKnot], time_s: f64, after: bool) -> f64 {
    let first = knots[0];
    if time_s < first.time_s || (time_s == first.time_s && !after) {
        return first.left;
    }
    let last = knots[knots.len() - 1];
    if time_s > last.time_s || (time_s == last.time_s && after) {
        return last.right;
    }
    let idx = knots.partition_point(|k| k.time_s < time_s);
    if let Some(knot) = knots.get(idx)
        && knot.time_s == time_s
    {
        return if after { knot.right } else { knot.left };
    }
    let left_knot = knots[idx - 1];
    let right_knot = knots[idx];
    let span = right_knot.time_s - left_knot.time_s;
    let frac = (time_s - left_knot.time_s) / span;
    left_knot.right + frac * (right_knot.left - left_knot.right)
}

/// Evaluates a discrete step track at `time_s`.
pub(crate) fn eval_discrete<T: Copy>(
    knots: &[DiscreteGeometryKnot<T>],
    time_s: f64,
    after: bool,
) -> T {
    let idx = if after {
        knots.partition_point(|k| k.time_s <= time_s)
    } else {
        knots.partition_point(|k| k.time_s < time_s)
    };
    if idx == 0 {
        knots[0].left
    } else {
        knots[idx - 1].right
    }
}

/// Bit-encoded value of `track` at `time_s` (`after = false` for `t⁻`, `true` for `t⁺`).
pub fn track_value_bits(track: &StructureGeometryTrack, time_s: f64, after: bool) -> u64 {
    match track {
        StructureGeometryTrack::Scalar { knots, .. }
        | StructureGeometryTrack::CulvertSediment { knots, .. } => {
            eval_scalar(knots, time_s, after).to_bits()
        }
        StructureGeometryTrack::CulvertClosedBarrels { knots } => {
            u64::from(eval_discrete(knots, time_s, after))
        }
        StructureGeometryTrack::Closed { knots } => u64::from(eval_discrete(knots, time_s, after)),
    }
}

/// Evaluates `element` at midpoint of `[start_s, end_s]` under `schedules` (which must lie inside
/// one knot-free interval `(start_s, end_s)`).
pub fn evaluate_element_interval(
    element: &StructureElement,
    schedules: &[&StructureGeometrySchedule],
    start_s: f64,
    end_s: f64,
) -> Result<StructureElement, StructureError> {
    if schedules.is_empty() {
        return Ok(element.clone());
    }
    let mid = start_s + 0.5 * (end_s - start_s);
    apply_at(element, schedules, mid, true)
}

/// Evaluates `element` at `time_s` (`after = true` for right limit, `false` for left limit).
pub(crate) fn apply_at(
    element: &StructureElement,
    schedules: &[&StructureGeometrySchedule],
    time_s: f64,
    after: bool,
) -> Result<StructureElement, StructureError> {
    let mut out = element.clone();
    // Apply law parameter changes first.
    match &element.law {
        StructureLaw::SuppressedRectangularWeir(p) => {
            let mut crest = p.crest_m();
            let mut length = p.crest_length_m();
            for s in schedules {
                if let StructureGeometryTrack::Scalar { parameter, knots } = &s.track {
                    match parameter {
                        ScalarGeometryParameter::WeirCrestElevation => {
                            crest = eval_scalar(knots, time_s, after);
                        }
                        ScalarGeometryParameter::WeirCrestLength => {
                            length = eval_scalar(knots, time_s, after);
                        }
                        _ => {}
                    }
                }
            }
            out.law =
                StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
                    length,
                    crest,
                    p.discharge_coefficient(),
                    p.applicability().clone(),
                )?);
        }
        StructureLaw::SubmergedOrifice(p) => {
            let mut area = p.area_m2();
            let mut sill = p.sill_m();
            let mut crown = p.crown_m();
            for s in schedules {
                if let StructureGeometryTrack::Scalar { parameter, knots } = &s.track {
                    match parameter {
                        ScalarGeometryParameter::OrificeArea => {
                            area = eval_scalar(knots, time_s, after);
                        }
                        ScalarGeometryParameter::OrificeSillElevation => {
                            sill = eval_scalar(knots, time_s, after);
                        }
                        ScalarGeometryParameter::OrificeCrownElevation => {
                            crown = eval_scalar(knots, time_s, after);
                        }
                        _ => {}
                    }
                }
            }
            out.law = StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
                area,
                sill,
                crown,
                p.discharge_coefficient(),
                p.gravity_m_per_s2(),
                p.applicability().clone(),
            )?);
        }
        StructureLaw::HydrostaticRectangularGate(p) => {
            let mut width = p.width_m();
            let mut sill = p.sill_m();
            let mut max_open = p.maximum_opening_m();
            for s in schedules {
                if let StructureGeometryTrack::Scalar { parameter, knots } = &s.track {
                    match parameter {
                        ScalarGeometryParameter::GateWidth => {
                            width = eval_scalar(knots, time_s, after);
                        }
                        ScalarGeometryParameter::GateSillElevation => {
                            sill = eval_scalar(knots, time_s, after);
                        }
                        ScalarGeometryParameter::GateMaximumOpening => {
                            max_open = eval_scalar(knots, time_s, after);
                        }
                        _ => {}
                    }
                }
            }
            out.law = StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                width,
                sill,
                max_open,
                p.discharge_coefficient(),
                p.gravity_m_per_s2(),
                p.applicability().clone(),
            )?);
        }
        StructureLaw::Culvert(_) | StructureLaw::Pump(_) => {}
    }

    // Apply closure / blockage condition changes.
    for s in schedules {
        if let StructureGeometryTrack::Closed { knots } = &s.track {
            if eval_discrete(knots, time_s, after) {
                out.condition = StructureCondition::Closed {
                    evidence: s.evidence.clone(),
                };
                return Ok(out);
            } else if matches!(out.condition, StructureCondition::Closed { .. }) {
                out.condition = StructureCondition::AsBuilt;
            }
        }
    }
    apply_blockage(&mut out, schedules, time_s, after)?;
    Ok(out)
}

fn apply_blockage(
    out: &mut StructureElement,
    schedules: &[&StructureGeometrySchedule],
    time_s: f64,
    after: bool,
) -> Result<(), StructureError> {
    let mut evidence: Option<&str> = None;
    match &out.law {
        StructureLaw::SuppressedRectangularWeir(_) => {
            for s in schedules {
                if let StructureGeometryTrack::Scalar {
                    parameter: ScalarGeometryParameter::WeirBlockage,
                    knots,
                } = &s.track
                {
                    let share = BlockageFraction::new(eval_scalar(knots, time_s, after))?;
                    out.condition = StructureCondition::Blocked(StructureBlockage::weir(
                        share,
                        s.evidence.clone(),
                    )?);
                }
            }
        }
        StructureLaw::SubmergedOrifice(_) => {
            for s in schedules {
                if let StructureGeometryTrack::Scalar {
                    parameter: ScalarGeometryParameter::OrificeBlockage,
                    knots,
                } = &s.track
                {
                    let share = BlockageFraction::new(eval_scalar(knots, time_s, after))?;
                    out.condition = StructureCondition::Blocked(StructureBlockage::orifice(
                        share,
                        s.evidence.clone(),
                    )?);
                }
            }
        }
        StructureLaw::HydrostaticRectangularGate(_) => {
            let (mut width, mut range) = match &out.condition {
                StructureCondition::Blocked(b) => match b.geometry() {
                    crate::structures::BlockageGeometry::Gate {
                        width,
                        opening_range,
                    } => (*width, *opening_range),
                    _ => (BlockageFraction::NONE, BlockageFraction::NONE),
                },
                _ => (BlockageFraction::NONE, BlockageFraction::NONE),
            };
            let mut changed = false;
            for s in schedules {
                if let StructureGeometryTrack::Scalar { parameter, knots } = &s.track {
                    match parameter {
                        ScalarGeometryParameter::GateWidthBlockage => {
                            width = BlockageFraction::new(eval_scalar(knots, time_s, after))?;
                            evidence = Some(s.evidence.as_str());
                            changed = true;
                        }
                        ScalarGeometryParameter::GateOpeningRangeBlockage => {
                            range = BlockageFraction::new(eval_scalar(knots, time_s, after))?;
                            evidence = Some(s.evidence.as_str());
                            changed = true;
                        }
                        _ => {}
                    }
                }
            }
            if changed {
                let ev = evidence.unwrap_or("Scheduled gate blockage").to_string();
                out.condition =
                    StructureCondition::Blocked(StructureBlockage::gate(width, range, ev)?);
            }
        }
        StructureLaw::Culvert(_) => {
            let mut cb = effective::culvert_blockage(&out.condition)
                .copied()
                .unwrap_or(CulvertBlockage::NONE);
            let mut changed = false;
            for s in schedules {
                match &s.track {
                    StructureGeometryTrack::Scalar {
                        parameter: ScalarGeometryParameter::CulvertFaceABlockage,
                        knots,
                    } => {
                        cb.face_a = BlockageFraction::new(eval_scalar(knots, time_s, after))?;
                        evidence = Some(s.evidence.as_str());
                        changed = true;
                    }
                    StructureGeometryTrack::Scalar {
                        parameter: ScalarGeometryParameter::CulvertFaceBBlockage,
                        knots,
                    } => {
                        cb.face_b = BlockageFraction::new(eval_scalar(knots, time_s, after))?;
                        evidence = Some(s.evidence.as_str());
                        changed = true;
                    }
                    StructureGeometryTrack::CulvertSediment {
                        bed_manning_n,
                        knots,
                    } => {
                        let depth = eval_scalar(knots, time_s, after);
                        cb.sediment = if depth > 0.0 {
                            Some(CulvertSediment {
                                depth_m: depth,
                                bed_manning_n: *bed_manning_n,
                            })
                        } else {
                            None
                        };
                        evidence = Some(s.evidence.as_str());
                        changed = true;
                    }
                    StructureGeometryTrack::CulvertClosedBarrels { knots } => {
                        cb.closed_barrels = eval_discrete(knots, time_s, after);
                        evidence = Some(s.evidence.as_str());
                        changed = true;
                    }
                    _ => {}
                }
            }
            if changed {
                let ev = evidence.unwrap_or("Scheduled culvert blockage").to_string();
                out.condition = StructureCondition::Blocked(StructureBlockage::culvert(cb, ev)?);
            }
        }
        StructureLaw::Pump(_) => {}
    }
    Ok(())
}
