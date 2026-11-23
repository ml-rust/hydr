//! Culvert structure law with blockage, sediment, open-bottom floor roughness, and closure.

use super::outlet_rows::{GRAVITY, ROWS as OUTLET_ROWS, SECTIONS as OUTLET_SECTIONS};
use super::outlets::{Lane as OutletLane, barrel as outlet_barrel, headwater as outlet_hw, lane};
use super::runtime::{Runtime, client};
use super::sections::Outcome;
use hydr::structures::{
    BlockageFraction, CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertBlockage,
    CulvertEnd, CulvertEnds, CulvertInlet, CulvertInletBatch, CulvertInletRow, CulvertParameters,
    CulvertSediment, DirectionPolicy, EndpointHead, HeadConvention, HeadSupport, HeldHeadInterval,
    StructureBatch, StructureBlockage, StructureCondition, StructureElement, StructureHeads,
    StructureHydraulicOperands, StructureLaw, StructureOutput, StructureRegime, StructureStatus,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

pub(super) const DATUM: &str = "analytical";
pub(super) const EVIDENCE: &str = "Synthetic blockage inspection";

pub(super) fn applicability() -> CulvertApplicability {
    CulvertApplicability {
        coefficient_provenance: "HDS-5 Table A.1".into(),
        barrel_condition_assertion: "Checked synthetic barrel".into(),
    }
}

pub(super) fn wet(stage_m: f64) -> HeadSupport {
    HeadSupport::Wet { stage_m }
}

pub(super) fn dry(bed_elevation_m: f64) -> HeadSupport {
    HeadSupport::Dry { bed_elevation_m }
}

pub(super) fn packet(id: &str, a: HeadSupport, b: HeadSupport) -> StructureHeads {
    StructureHeads {
        structure_id: id.into(),
        endpoint_a: EndpointHead {
            id: format!("a-{id}"),
            support: a,
        },
        endpoint_b: EndpointHead {
            id: format!("b-{id}"),
            support: b,
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }
}

pub(super) fn element(
    id: &str,
    direction: DirectionPolicy,
    params: CulvertParameters,
    condition: StructureCondition,
) -> StructureElement {
    StructureElement {
        id: id.into(),
        endpoint_a: format!("a-{id}"),
        endpoint_b: format!("b-{id}"),
        datum: DATUM.into(),
        direction,
        law: StructureLaw::Culvert(params),
        condition,
    }
}

pub(super) fn propose(
    elements: Vec<StructureElement>,
    heads: &[StructureHeads],
) -> Outcome<Vec<StructureOutput>> {
    let batch = StructureBatch::<Runtime>::with_culverts(client(), elements)?;
    Ok(batch.propose(heads)?.outputs()?)
}

pub(super) fn blocked(blockage: CulvertBlockage) -> Outcome<StructureCondition> {
    Ok(StructureCondition::Blocked(StructureBlockage::culvert(
        blockage,
        EVIDENCE.into(),
    )?))
}

pub(super) fn circle_params(
    inverts: [f64; 2],
    count: u32,
    ends: CulvertEnds,
) -> Outcome<CulvertParameters> {
    Ok(CulvertParameters::new(
        CulvertBarrel::circular(1.2)?,
        CulvertAlignment {
            length_m: 40.0,
            invert_a_m: inverts[0],
            invert_b_m: inverts[1],
        },
        0.012,
        count,
        ends,
        9.81,
        applicability(),
    )?)
}

#[test]
fn zero_blockage_reproduces_as_built_bits_over_head_grid() -> Outcome<()> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let params = circle_params(
        [0.1, 0.0],
        2,
        CulvertEnds {
            a: Some(end),
            b: Some(end),
        },
    )?;
    let grid = [
        (wet(0.05), wet(0.0)),
        (wet(0.6), wet(0.6)),
        (wet(0.8), dry(0.0)),
        (wet(1.5), wet(0.4)),
        (wet(2.4), wet(1.6)),
        (wet(0.5), wet(1.9)),
        (dry(0.0), wet(2.2)),
        (dry(0.0), dry(-0.1)),
    ];
    let run = |cond: StructureCondition| -> Outcome<String> {
        let mut elements = Vec::new();
        let mut heads = Vec::new();
        for (k, (a, b)) in grid.iter().enumerate() {
            let id = format!("c{k}");
            elements.push(element(
                &id,
                DirectionPolicy::Bidirectional,
                params.clone(),
                cond.clone(),
            ));
            heads.push(packet(&id, a.clone(), b.clone()));
        }
        Ok(format!("{:?}", propose(elements, &heads)?))
    };
    let as_built = run(StructureCondition::AsBuilt)?;
    let zero = run(blocked(CulvertBlockage::NONE)?)?;
    assert_eq!(zero, as_built);
    Ok(())
}

#[test]
fn closed_culvert_returns_exact_zero_under_every_head_class() -> Outcome<()> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let params = circle_params(
        [0.1, 0.0],
        2,
        CulvertEnds {
            a: Some(end),
            b: Some(end),
        },
    )?;
    let cond = StructureCondition::Closed {
        evidence: "Bulkhead".into(),
    };
    let cases = [
        (wet(2.5), wet(0.2)),
        (wet(0.2), wet(2.5)),
        (wet(1.0), wet(1.0)),
        (wet(0.0), dry(-0.1)),
        (dry(0.0), wet(2.0)),
        (dry(0.0), dry(-0.1)),
    ];
    let mut elements = Vec::new();
    let mut heads = Vec::new();
    for (k, (a, b)) in cases.iter().enumerate() {
        let id = format!("c{k}");
        elements.push(element(
            &id,
            DirectionPolicy::Bidirectional,
            params.clone(),
            cond.clone(),
        ));
        heads.push(packet(&id, a.clone(), b.clone()));
    }
    let batch = StructureBatch::<Runtime>::with_culverts(client(), elements)?;
    let proposal = batch.propose(&heads)?;
    let outputs = proposal.outputs()?;
    let volumes = proposal
        .held_volumes(HeldHeadInterval {
            start: 0.0,
            end: 5.0,
        })?
        .outputs()?;
    for (output, volume) in outputs.iter().zip(&volumes) {
        assert_eq!(output.discharge_m3_per_s.to_bits(), 0.0_f64.to_bits());
        assert_eq!(volume.volume_m3.to_bits(), 0.0_f64.to_bits());
        assert_eq!(output.regime, StructureRegime::Closed);
        assert_eq!(output.status, StructureStatus::PhysicalZero);
        assert_eq!(output.proposed_direction, None);
        assert_eq!(output.hydraulic_operands, None);
    }
    Ok(())
}

#[test]
fn discharge_never_rises_with_face_fraction_closed_barrels_or_rough_sediment() -> Outcome<()> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let params = circle_params(
        [0.08, 0.0],
        4,
        CulvertEnds {
            a: Some(end),
            b: None,
        },
    )?;
    let heads = [packet("c", wet(1.8), wet(0.6))];

    // Face fraction sweep.
    let mut prev: Option<f64> = None;
    for share in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] {
        let out = propose(
            vec![element(
                "c",
                DirectionPolicy::AtoBOnly,
                params.clone(),
                blocked(CulvertBlockage {
                    face_a: BlockageFraction::new(share)?,
                    ..CulvertBlockage::NONE
                })?,
            )],
            &heads,
        )?;
        let q = out[0].discharge_m3_per_s;
        if let Some(before) = prev {
            assert!(q <= before, "face={share}: {q} > {before}");
        }
        prev = Some(q);
    }

    // Closed barrels k sweep: linear in N - k.
    let mut base_q = 0.0;
    for k in 0..4u32 {
        let out = propose(
            vec![element(
                "c",
                DirectionPolicy::AtoBOnly,
                params.clone(),
                blocked(CulvertBlockage {
                    closed_barrels: k,
                    ..CulvertBlockage::NONE
                })?,
            )],
            &heads,
        )?;
        let q = out[0].discharge_m3_per_s;
        if k == 0 {
            base_q = q;
        } else {
            let expected = base_q * f64::from(4 - k) / 4.0;
            assert!(
                (q - expected).abs() <= 1e-12 * expected,
                "k={k}: {q} vs {expected}"
            );
        }
    }

    // Sediment sweep for rho = 2.5 >= 1 on circle: no certified reversal.
    let mut prev_hi: Option<f64> = None;
    for frac in [0.0, 0.05, 0.2, 0.4, 0.5, 0.75] {
        let sed = (frac > 0.0).then_some(CulvertSediment {
            depth_m: 1.2 * frac,
            bed_manning_n: 0.030,
        });
        let out = propose(
            vec![element(
                "c",
                DirectionPolicy::AtoBOnly,
                params.clone(),
                blocked(CulvertBlockage {
                    sediment: sed,
                    ..CulvertBlockage::NONE
                })?,
            )],
            &heads,
        )?;
        let Some(StructureHydraulicOperands::Culvert {
            barrel_discharge_m3_per_s: (lo, hi),
            ..
        }) = out[0].hydraulic_operands
        else {
            return Err("expected active culvert operands".into());
        };
        if let Some(before) = prev_hi {
            assert!(lo <= before, "h/D={frac}: [{lo}, {hi}] rose above {before}");
        }
        prev_hi = Some(hi);
    }

    // Smooth bed (rho = 0.5) in a rough pipe (n = 0.026): record direction under drowned full flow.
    let rough = CulvertParameters::new(
        CulvertBarrel::circular(1.2)?,
        CulvertAlignment {
            length_m: 80.0,
            invert_a_m: 0.0,
            invert_b_m: 0.0,
        },
        0.026,
        1,
        CulvertEnds {
            a: Some(end),
            b: None,
        },
        9.81,
        applicability(),
    )?;
    let drowned_heads = [packet("c", wet(3.0), wet(2.0))];
    let q_clean = propose(
        vec![element(
            "c",
            DirectionPolicy::AtoBOnly,
            rough.clone(),
            StructureCondition::AsBuilt,
        )],
        &drowned_heads,
    )?[0]
        .discharge_m3_per_s;
    let q_smooth = propose(
        vec![element(
            "c",
            DirectionPolicy::AtoBOnly,
            rough,
            blocked(CulvertBlockage {
                sediment: Some(CulvertSediment {
                    depth_m: 0.05,
                    bed_manning_n: 0.013,
                }),
                ..CulvertBlockage::NONE
            })?,
        )],
        &drowned_heads,
    )?[0]
        .discharge_m3_per_s;
    // First-order full-flow conveyance rises for a thin smooth bed (rho < 1).
    assert!(
        q_smooth > q_clean,
        "expected smooth bed to raise Q: {q_smooth} <= {q_clean}"
    );
    Ok(())
}

#[test]
fn round_trip_holds_per_floored_shape_and_face_blockage() -> Outcome<()> {
    let cases = [
        (0, 1, 0.0),
        (1, 1, 0.5),
        (2, 9, 0.0),
        (3, 9, 0.25),
        (4, 30, 0.0),
        (5, 34, 0.5),
        (6, 36, 0.5),
        (7, 45, 0.0),
        (8, 45, 0.25),
    ];
    for (section_idx, code, face) in cases {
        let entry = &OUTLET_SECTIONS[section_idx];
        let barrel_with_floor = outlet_barrel(section_idx)?;
        let floor = barrel_with_floor.floor().ok_or("missing floor")?;
        let row = OUTLET_ROWS
            .iter()
            .find(|r| r.section == section_idx && r.name == "m1" && r.face_fraction == 0)
            .ok_or("missing m1 row")?;
        let wall_n = f64::from_bits(row.wall_manning);
        let q_target = f64::from_bits(row.discharge);
        let tw = f64::from_bits(row.tailwater);
        let length = f64::from_bits(row.length);
        let slope = f64::from_bits(row.slope);
        let ke = f64::from_bits(row.ke);
        let f_block = BlockageFraction::new(face)?;
        let inlet_batch = CulvertInletBatch::<Runtime>::new(
            client(),
            vec![CulvertInletRow {
                barrel: barrel_with_floor,
                inlet: CulvertInlet::hds5(code)?,
                slope,
                gravity_m_per_s2: f64::from_bits(GRAVITY),
                face_blockage: f_block,
            }],
        )?;
        let q_tensor = Tensor::<Runtime>::from_slice(&[q_target], &[1], client().device())?;
        let hw_in = inlet_batch.headwater(&q_tensor)?.project()?[0].headwater_m;
        let out_lane = OutletLane {
            row: hydr::structures::CulvertOutletRow {
                face_blockage: f_block,
                ..lane(row)?.row
            },
            discharge: q_target,
            tailwater: tw,
        };
        let hw_out = outlet_hw(&[out_lane])?[0].headwater_m;
        let hw = 0.5 * (hw_in.0.max(hw_out.0) + hw_in.1.max(hw_out.1));
        let invert_a = slope * length;
        let invert_b = 0.0;
        let bed_a = invert_a + floor.sediment_depth_m;
        let bed_b = invert_b + floor.sediment_depth_m;
        let mut params = CulvertParameters::new(
            barrel_with_floor.base(),
            CulvertAlignment {
                length_m: length,
                invert_a_m: invert_a,
                invert_b_m: invert_b,
            },
            wall_n,
            1,
            CulvertEnds {
                a: Some(CulvertEnd::new(CulvertInlet::hds5(code)?, ke)?),
                b: None,
            },
            f64::from_bits(GRAVITY),
            applicability(),
        )?;
        let cond = if floor.sediment_depth_m == 0.0 {
            params = params.with_floor_manning_n(wall_n * floor.roughness_ratio)?;
            if face == 0.0 {
                StructureCondition::AsBuilt
            } else {
                blocked(CulvertBlockage {
                    face_a: f_block,
                    ..CulvertBlockage::NONE
                })?
            }
        } else {
            blocked(CulvertBlockage {
                face_a: f_block,
                sediment: Some(CulvertSediment {
                    depth_m: floor.sediment_depth_m,
                    bed_manning_n: wall_n * floor.roughness_ratio,
                }),
                ..CulvertBlockage::NONE
            })?
        };
        let out = propose(
            vec![element("c", DirectionPolicy::AtoBOnly, params, cond)],
            &[packet("c", wet(bed_a + hw), wet(bed_b + tw))],
        )?;
        let q_found = out[0].discharge_m3_per_s;
        assert!(
            (q_found - q_target).abs() <= 1e-6 * q_target,
            "{} face={face}: Q(HW(Q))={q_found} vs {q_target}",
            entry.name
        );
    }
    Ok(())
}
