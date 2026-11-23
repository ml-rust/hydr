//! Bed-referenced heads, reversal, open-bottom floor roughness, rejections, and geometry reports.

use super::laws::{applicability, blocked, circle_params, element, packet, propose, wet};
use super::runtime::{Runtime, client};
use super::sections::Outcome;
use hydr::structures::{
    BlockageFraction, CulvertAlignment, CulvertBarrel, CulvertBlockage, CulvertEnd, CulvertEnds,
    CulvertInlet, CulvertParameters, CulvertSediment, DirectionPolicy, FlowDirection,
    StructureBatch, StructureCondition, StructureConditionKind, StructureError, StructureGeometry,
    StructureHydraulicOperands, StructureRegime,
};

fn err_msg<T>(res: Result<T, StructureError>) -> String {
    res.err().map(|e| e.to_string()).unwrap_or_default()
}

#[test]
fn bed_referenced_heads_equal_heads_dry_donor_and_reversal() -> Outcome<()> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let sed = CulvertSediment {
        depth_m: 0.3,
        bed_manning_n: 0.030,
    };
    let params = circle_params(
        [0.1, 0.0],
        1,
        CulvertEnds {
            a: Some(end),
            b: Some(end),
        },
    )?;
    let cond = blocked(CulvertBlockage {
        sediment: Some(sed),
        ..CulvertBlockage::NONE
    })?;
    // Bed A = 0.4, Bed B = 0.3. Donor at or below bed gives exact zero and CulvertBelowInvert.
    let dry_donor = propose(
        vec![element(
            "c",
            DirectionPolicy::Bidirectional,
            params.clone(),
            cond.clone(),
        )],
        &[packet("c", wet(0.4), wet(0.25))],
    )?;
    assert_eq!(dry_donor[0].discharge_m3_per_s.to_bits(), 0.0_f64.to_bits());
    assert_eq!(dry_donor[0].regime, StructureRegime::CulvertBelowInvert);

    // Equal stages above both beds give exact zero and EqualHeads.
    let equal = propose(
        vec![element(
            "c",
            DirectionPolicy::Bidirectional,
            params.clone(),
            cond.clone(),
        )],
        &[packet("c", wet(0.9), wet(0.9))],
    )?;
    assert_eq!(equal[0].discharge_m3_per_s.to_bits(), 0.0_f64.to_bits());
    assert_eq!(equal[0].regime, StructureRegime::EqualHeads);

    // Active operands measure headwater and tailwater from each end's bed.
    let active = propose(
        vec![element("c", DirectionPolicy::Bidirectional, params, cond)],
        &[packet("c", wet(1.4), wet(0.8))],
    )?;
    assert_eq!(
        active[0].hydraulic_operands.as_ref().map(|op| match op {
            StructureHydraulicOperands::Culvert {
                headwater_m,
                tailwater_m,
                ..
            } => (*headwater_m, *tailwater_m),
            _ => (f64::NAN, f64::NAN),
        }),
        Some((1.4 - (0.1 + 0.3), 0.8 - 0.3))
    );

    // Swapped heads negate discharge with the other end's face fraction (AtoB vs BtoA).
    let f = BlockageFraction::new(0.4)?;
    let fwd = circle_params(
        [0.0, 0.0],
        1,
        CulvertEnds {
            a: Some(end),
            b: None,
        },
    )?;
    let rev = circle_params(
        [0.0, 0.0],
        1,
        CulvertEnds {
            a: None,
            b: Some(end),
        },
    )?;
    let q_fwd = propose(
        vec![element(
            "c",
            DirectionPolicy::AtoBOnly,
            fwd,
            blocked(CulvertBlockage {
                face_a: f,
                sediment: Some(sed),
                ..CulvertBlockage::NONE
            })?,
        )],
        &[packet("c", wet(1.5), wet(0.7))],
    )?[0]
        .clone();
    let q_rev = propose(
        vec![element(
            "c",
            DirectionPolicy::BtoAOnly,
            rev,
            blocked(CulvertBlockage {
                face_b: f,
                sediment: Some(sed),
                ..CulvertBlockage::NONE
            })?,
        )],
        &[packet("c", wet(0.7), wet(1.5))],
    )?[0]
        .clone();
    assert_eq!(q_fwd.proposed_direction, Some(FlowDirection::AtoB));
    assert_eq!(q_rev.proposed_direction, Some(FlowDirection::BtoA));
    assert_eq!(
        q_fwd.discharge_m3_per_s.to_bits(),
        (-q_rev.discharge_m3_per_s).to_bits()
    );
    Ok(())
}

#[test]
fn open_bottom_floor_manning_matches_zero_depth_floor_and_sediment_replaces_it() -> Outcome<()> {
    for (barrel, code) in [
        (CulvertBarrel::rectangular_box(2.0, 1.5)?, 9),
        (CulvertBarrel::arch(3.048, 1.6002)?, 45),
    ] {
        let base = CulvertParameters::new(
            barrel,
            CulvertAlignment {
                length_m: 40.0,
                invert_a_m: 0.04,
                invert_b_m: 0.0,
            },
            0.012,
            1,
            CulvertEnds {
                a: Some(CulvertEnd::new(CulvertInlet::hds5(code)?, 0.5)?),
                b: None,
            },
            9.81,
            applicability(),
        )?;
        let open_floor = base.clone().with_floor_manning_n(0.030)?;
        let heads = [packet("c", wet(1.4), wet(0.9))];
        let q_plain = propose(
            vec![element(
                "c",
                DirectionPolicy::AtoBOnly,
                base.clone(),
                StructureCondition::AsBuilt,
            )],
            &heads,
        )?[0]
            .discharge_m3_per_s;
        let q_rough_floor = propose(
            vec![element(
                "c",
                DirectionPolicy::AtoBOnly,
                open_floor.clone(),
                StructureCondition::AsBuilt,
            )],
            &heads,
        )?[0]
            .discharge_m3_per_s;
        assert!(q_rough_floor < q_plain, "rougher floor must lower outlet Q");
        let sed = blocked(CulvertBlockage {
            sediment: Some(CulvertSediment {
                depth_m: 0.3,
                bed_manning_n: 0.024,
            }),
            ..CulvertBlockage::NONE
        })?;
        let q_a = propose(
            vec![element("c", DirectionPolicy::AtoBOnly, base, sed.clone())],
            &heads,
        )?[0]
            .discharge_m3_per_s;
        let q_b = propose(
            vec![element("c", DirectionPolicy::AtoBOnly, open_floor, sed)],
            &heads,
        )?[0]
            .discharge_m3_per_s;
        assert_eq!(q_a.to_bits(), q_b.to_bits());
    }
    Ok(())
}

#[test]
fn culvert_blockage_rejections_name_the_rule_and_action() -> Outcome<()> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let two_way = circle_params(
        [0.0, 0.0],
        2,
        CulvertEnds {
            a: Some(end),
            b: Some(end),
        },
    )?;
    let one_way = circle_params(
        [0.0, 0.0],
        2,
        CulvertEnds {
            a: Some(end),
            b: None,
        },
    )?;
    let adverse = circle_params(
        [0.0, 0.4],
        1,
        CulvertEnds {
            a: Some(end),
            b: None,
        },
    )?;

    let m = err_msg(one_way.clone().with_floor_manning_n(0.03));
    assert!(
        m.contains("only a box or flat-floor arch barrel has its own floor"),
        "{m}"
    );

    let batch_err = |dir, params, cond| {
        err_msg(StructureBatch::<Runtime>::with_culverts(
            client(),
            vec![element("c", dir, params, cond)],
        ))
    };

    let m = batch_err(
        DirectionPolicy::Bidirectional,
        two_way,
        blocked(CulvertBlockage {
            face_a: BlockageFraction::new(0.2)?,
            ..CulvertBlockage::NONE
        })?,
    );
    assert!(m.contains("exit-face blockage"), "{m}");

    let m = batch_err(
        DirectionPolicy::AtoBOnly,
        one_way.clone(),
        blocked(CulvertBlockage {
            closed_barrels: 2,
            ..CulvertBlockage::NONE
        })?,
    );
    assert!(m.contains("StructureCondition::Closed"), "{m}");

    let m = batch_err(
        DirectionPolicy::AtoBOnly,
        one_way.clone(),
        blocked(CulvertBlockage {
            sediment: Some(CulvertSediment {
                depth_m: 1.2,
                bed_manning_n: 0.03,
            }),
            ..CulvertBlockage::NONE
        })?,
    );
    assert!(m.contains("the barrel is full"), "{m}");

    let m = batch_err(
        DirectionPolicy::AtoBOnly,
        adverse,
        blocked(CulvertBlockage {
            sediment: Some(CulvertSediment {
                depth_m: 0.5,
                bed_manning_n: 0.03,
            }),
            ..CulvertBlockage::NONE
        })?,
    );
    assert!(m.contains("sediment_depth_m=0.5"), "{m}");

    let m = batch_err(
        DirectionPolicy::AtoBOnly,
        one_way,
        blocked(CulvertBlockage {
            sediment: Some(CulvertSediment {
                depth_m: 0.0,
                bed_manning_n: 0.03,
            }),
            ..CulvertBlockage::NONE
        })?,
    );
    assert!(m.contains("a circular barrel has no floor"), "{m}");
    Ok(())
}

#[test]
fn geometry_report_describes_base_and_effective_culvert_geometry() -> Outcome<()> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let params = circle_params(
        [0.1, 0.0],
        3,
        CulvertEnds {
            a: Some(end),
            b: None,
        },
    )?;
    let batch = StructureBatch::<Runtime>::with_culverts(
        client(),
        vec![
            element(
                "c0",
                DirectionPolicy::AtoBOnly,
                params.clone(),
                blocked(CulvertBlockage {
                    face_a: BlockageFraction::new(0.25)?,
                    face_b: BlockageFraction::NONE,
                    sediment: Some(CulvertSediment {
                        depth_m: 0.3,
                        bed_manning_n: 0.030,
                    }),
                    closed_barrels: 1,
                })?,
            ),
            element(
                "c1",
                DirectionPolicy::AtoBOnly,
                params,
                StructureCondition::Closed {
                    evidence: "Sealed".into(),
                },
            ),
        ],
    )?;
    let reports = batch.geometry();
    assert_eq!(reports[0].condition, StructureConditionKind::Blocked);
    assert_eq!(reports[1].condition, StructureConditionKind::Closed);
    assert_eq!(reports[1].effective, None);
    let StructureGeometry::Culvert {
        rise_m,
        full_area_m2,
        bed_a_m,
        bed_b_m,
        barrel_count,
        face_open_share,
        floor_width_m,
        roughness_ratio,
        ..
    } = reports[0].effective.as_ref().ok_or("missing effective")?
    else {
        return Err("expected Culvert geometry".into());
    };
    let open_rise = 1.2_f64 - 0.3_f64;
    let admitted_bed = 1.2_f64 - open_rise;
    assert_eq!(*rise_m, open_rise);
    assert!(full_area_m2.0 > 0.0 && full_area_m2.0 <= full_area_m2.1);
    assert_eq!((*bed_a_m, *bed_b_m), (0.1 + admitted_bed, admitted_bed));
    assert_eq!(*barrel_count, 2);
    assert_eq!(*face_open_share, [0.75, 1.0]);
    assert!(floor_width_m.0 > 0.0 && floor_width_m.0 <= floor_width_m.1);
    assert_eq!(*roughness_ratio, Some(0.030 / 0.012));
    Ok(())
}
