//! Capability declaration of `SurfaceChannelDrainageBatch` checked against admission.

use hydr::capability::{ComponentId, UnsupportedReason};
use hydr::coupling::{
    DrainageControlAuthorityTable, DrainageControlClaim, SurfaceDrainageError,
    SurfaceDrainageInlet, surface_channel_drainage_capabilities,
};
use hydr::drainage::SwmmControlAuthority;

const SAMPLE_INP: &str = r#"[CONTROLS]
RULE R1
IF NODE SU1 DEPTH > 1.5
THEN ORIFICE OR1 SETTING = 1.0
ELSE ORIFICE OR1 SETTING = 0.25
"#;

fn sample_inlet(id: &str, cell: usize, node: &str) -> SurfaceDrainageInlet {
    SurfaceDrainageInlet {
        id: id.to_string(),
        surface_cell: cell,
        drainage_node_id: node.to_string(),
        rim_elevation_m: 10.0,
        weir_crest_length_m: 2.0,
        weir_cd: 1.7,
        orifice_area_m2: 0.5,
        orifice_cd: 0.62,
        allow_surcharge_to_surface: true,
    }
}

#[test]
fn surface_channel_drainage_declaration_matches_admission() -> Result<(), Box<dyn std::error::Error>>
{
    let caps = surface_channel_drainage_capabilities();
    assert_eq!(caps.component, ComponentId::SurfaceChannelDrainage);

    let rejected_ids: Vec<&str> = caps
        .unsupported
        .iter()
        .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission)
        .map(|u| u.id)
        .collect();
    assert_eq!(
        rejected_ids,
        ["duplicate-parameter-authority", "datum-mismatch"]
    );

    let links = vec!["OR1".to_string(), "P1".to_string()];
    let conflict = DrainageControlAuthorityTable::build(
        SAMPLE_INP,
        &links,
        &[DrainageControlClaim {
            link_id: "OR1".to_string(),
            authority: SwmmControlAuthority::HydrActuatorImposedPosition,
        }],
        &[sample_inlet("IN1", 0, "J1")],
    );
    assert!(matches!(
        conflict,
        Err(SurfaceDrainageError::Input(ref msg)) if msg.contains("duplicate-parameter-authority")
    ));
    Ok(())
}
