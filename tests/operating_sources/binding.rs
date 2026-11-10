use super::fixture::*;
use hydr::structures::*;

fn manual(source: &str, ids: &[&str]) -> OperatingSourceBinding<'static> {
    OperatingSourceBinding::Manual {
        source_id: source.into(),
        structure_ids: ids.iter().map(|id| (*id).into()).collect(),
    }
}

#[test]
fn every_original_element_requires_one_explicit_ordered_writer() -> Result {
    let context = StructureBatch::new(vec![gate("a")?, passive("weir")?, gate("b")?])?;
    for sources in [
        vec![],
        vec![manual("one", &["a", "weir"])],
        vec![manual("one", &["a", "weir", "b"]), manual("two", &["b"])],
        vec![manual("one", &["a", "weir", "b"]), manual("one", &["a"])],
        vec![manual("one", &["a", "b", "weir"])],
        vec![manual("one", &["a", "a", "weir", "b"])],
        vec![manual("one", &["a", "weir", "foreign"])],
        vec![manual("", &["a", "weir", "b"])],
        vec![manual("empty", &[]), manual("one", &["a", "weir", "b"])],
    ] {
        kind(
            context.with_operating_sources(sources),
            StructureInputErrorKind::Identity,
        )?;
    }
    let registry = context.with_operating_sources(vec![
        manual("first", &["weir"]),
        manual("second", &["a", "b"]),
    ])?;
    assert_eq!(registry.elements(), context.elements());
    assert!(registry.support().is_none());
    let proposal = registry.propose_at(
        0.0,
        &heads(&context),
        &[
            opening("a", 0.25),
            operation("weir", StructureOperatingPoint::Passive),
            opening("b", 0.75),
        ],
        &[],
    )?;
    let outputs = proposal.outputs()?;
    assert_eq!(outputs[0].source.source_id, "second");
    assert_eq!(outputs[1].source.source_id, "first");
    assert_eq!(outputs[2].source.source_id, "second");
    Ok(())
}

#[test]
fn complete_foreign_context_rejects_even_when_bound_gate_ids_match() -> Result {
    let context = StructureBatch::new(vec![gate("a")?, gate("b")?])?;
    for variant in 0..4 {
        let mut elements = context.elements().to_vec();
        match variant {
            0 => elements[1].endpoint_a.push_str(" altered"),
            1 => elements[1].datum.push_str(" altered"),
            2 => elements[1].direction = DirectionPolicy::AtoBOnly,
            3 => elements.reverse(),
            _ => return Err("Unknown immutable-context mutation".into()),
        }
        let foreign = StructureBatch::new(elements)?;
        let prescribed = trajectory(&foreign, "a", &[(0.0, 0.25), (2.0, 0.5)])?;
        kind(
            context.with_operating_sources(vec![
                OperatingSourceBinding::Prescribed {
                    source_id: "path".into(),
                    trajectory: &prescribed,
                },
                manual("manual", &["b"]),
            ]),
            StructureInputErrorKind::Definition,
        )?;
        let actual = actuator(&foreign, "a", 0.25, 1.0, (0.25, 0.5), 0.0, 2.0)?;
        kind(
            context.with_operating_sources(vec![
                OperatingSourceBinding::Actuated {
                    source_id: "actual".into(),
                    actuator: &actual,
                },
                manual("manual", &["b"]),
            ]),
            StructureInputErrorKind::Definition,
        )?;
    }
    Ok(())
}

#[test]
fn manual_operations_cover_the_original_complement_order_without_overrides() -> Result {
    let fixture = Mixed::new()?;
    let heads = heads(&fixture.context);
    let manual = fixture.manual();
    let frames = fixture.frames();
    kind(
        fixture
            .registry
            .propose_at(0.0, &heads, &manual[..3], &frames),
        StructureInputErrorKind::OperatingPoint,
    )?;
    let mut extra = manual.clone();
    extra.push(opening("prescribed-a", 0.25));
    kind(
        fixture.registry.propose_at(0.0, &heads, &extra, &frames),
        StructureInputErrorKind::OperatingPoint,
    )?;
    let mut reordered = manual.clone();
    reordered.swap(0, 3);
    kind(
        fixture
            .registry
            .propose_at(0.0, &heads, &reordered, &frames),
        StructureInputErrorKind::Identity,
    )?;
    let mut overridden = manual.clone();
    overridden[0] = opening("actuated-a", 0.25);
    kind(
        fixture
            .registry
            .propose_at(0.0, &heads, &overridden, &frames),
        StructureInputErrorKind::Identity,
    )?;
    let proposal = fixture.registry.propose_at(0.0, &heads, &manual, &frames)?;
    explicit_match(&fixture.context, &heads, &proposal)
}

#[test]
fn all_manual_points_accept_signed_finite_times_without_inventing_future_support() -> Result {
    let context = StructureBatch::new(vec![gate("a")?, passive("weir")?, pump("pump")?])?;
    let registry =
        context.with_operating_sources(vec![manual("inputs", &["a", "weir", "pump"])])?;
    assert!(registry.support().is_none());
    let heads = heads(&context);
    let operations = vec![
        opening("a", 0.25),
        operation("weir", StructureOperatingPoint::Passive),
        operation("pump", StructureOperatingPoint::PumpStoppedIsolated),
    ];
    for time in [-1e100, -1.0, -0.0, 0.0, 1e100] {
        let proposal = registry.propose_at(time, &heads, &operations, &[])?;
        assert_eq!(proposal.time_s().to_bits(), time.to_bits());
        for output in proposal.outputs()? {
            assert_eq!(output.source.time_s.to_bits(), time.to_bits());
            assert!(matches!(
                output.source.detail,
                OperatingSourceDetail::Manual
            ));
        }
        assert!(registry.next_event_after(time, &[])?.is_none());
        explicit_match(&context, &heads, &proposal)?;
    }
    for time in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        kind(
            registry.propose_at(time, &heads, &operations, &[]),
            StructureInputErrorKind::NonfiniteInput,
        )?;
        kind(
            registry.next_event_after(time, &[]),
            StructureInputErrorKind::NonfiniteInput,
        )?;
    }
    Ok(())
}
