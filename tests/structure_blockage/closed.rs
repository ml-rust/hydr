//! Closed structures give exact zero ahead of every regime, and mixed layouts keep each lane's
//! single-row bits.

use super::fixture::{Result, blocked, closed, element, laws, operation, packet, proposal, wet};
use hydr::structures::{
    DirectionPolicy, HeadSupport, HeldHeadInterval, StructureCondition, StructureElement,
    StructureHeads, StructureLaw, StructureOutput, StructureRegime, StructureStatus,
};

/// Head classes each law admits: unequal, reversed and equal stages, plus a dry donor where the
/// law admits one.
fn classes(law: &StructureLaw) -> Vec<[HeadSupport; 2]> {
    let mut classes = vec![
        [wet(3.0), wet(1.5)],
        [wet(1.5), wet(3.0)],
        [wet(2.0), wet(2.0)],
    ];
    if !matches!(law, StructureLaw::SubmergedOrifice(_)) {
        classes.push([
            HeadSupport::Dry {
                bed_elevation_m: 0.0,
            },
            wet(2.0),
        ]);
    }
    classes
}

fn check_closed(output: &StructureOutput) {
    assert_eq!(
        output.discharge_m3_per_s.to_bits(),
        0.0_f64.to_bits(),
        "{output:?}"
    );
    assert_eq!(output.regime, StructureRegime::Closed, "{output:?}");
    assert_eq!(output.status, StructureStatus::PhysicalZero, "{output:?}");
    assert_eq!(output.proposed_direction, None, "{output:?}");
    assert_eq!(output.hydraulic_operands, None, "{output:?}");
}

#[test]
fn closed_laws_give_exact_zero_under_every_head_class() -> Result {
    for law in laws()? {
        for [a, b] in classes(&law) {
            let element = element("s", law.clone(), closed());
            let proposal = proposal(vec![element], &[packet("s", a, b)])?;
            for output in proposal.outputs()? {
                check_closed(&output);
            }
            for held in proposal
                .held_volumes(HeldHeadInterval {
                    start: 0.0,
                    end: 2.0,
                })?
                .outputs()?
            {
                assert_eq!(held.volume_m3.to_bits(), 0.0_f64.to_bits(), "{held:?}");
            }
        }
    }
    Ok(())
}

#[test]
fn closed_takes_precedence_over_gate_closure_and_flap_blocking() -> Result {
    let [weir, _, gate] = laws()?;
    let shut = element("g", gate, closed());
    let heads = [packet("g", wet(3.0), wet(1.5))];
    let operations = [operation(&shut, 0.0)];
    let outputs = super::fixture::batch(vec![shut])?
        .propose_with_operations(&heads, &operations)?
        .outputs()?;
    outputs.iter().for_each(check_closed);
    let mut flap = element("w", weir, closed());
    flap.direction = DirectionPolicy::AtoBOnly;
    let outputs = proposal(vec![flap], &[packet("w", wet(1.5), wet(3.0))])?.outputs()?;
    outputs.iter().for_each(check_closed);
    Ok(())
}

/// Lane `index` of a mixed layout: law, condition and stages cycle independently.
fn lane(index: usize) -> Result<(StructureElement, StructureHeads)> {
    let laws = laws()?;
    let law = &laws[index % 3];
    let condition = match (index / 3) % 3 {
        0 => StructureCondition::AsBuilt,
        1 => blocked(law, 0.3)?,
        _ => closed(),
    };
    let id = format!("lane-{index}");
    let high = 2.0 + 0.125 * (index % 7) as f64;
    let heads = if index.is_multiple_of(2) {
        packet(&id, wet(high), wet(1.5))
    } else {
        packet(&id, wet(1.5), wet(high))
    };
    Ok((element(&id, law.clone(), condition), heads))
}

#[test]
fn mixed_layouts_match_single_rows() -> Result {
    for width in [1, 31, 32, 33, 64, 65] {
        let (elements, heads): (Vec<_>, Vec<_>) = (0..width)
            .map(lane)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        let batch = proposal(elements.clone(), &heads)?.outputs()?;
        assert_eq!(batch.len(), width);
        for (mut output, (element, head)) in batch.into_iter().zip(elements.into_iter().zip(heads))
        {
            let single = proposal(vec![element], &[head])?.outputs()?;
            output.element = 0;
            assert_eq!(
                format!("{single:?}"),
                format!("{:?}", [output]),
                "width={width}"
            );
        }
    }
    Ok(())
}
