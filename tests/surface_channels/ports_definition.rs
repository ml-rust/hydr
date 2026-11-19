//! Port admission and endpoint policy rejections name the capability entry and the input.

use crate::fixture::{Result, contains};
use crate::ports_fixture::{
    ALPHA, BETA, REACH_WIDTH, Scenario, forcing, four_ports, port, rectangular, rejection, west,
    x_face, y_face,
};
use hydr::channel::{
    ChannelBoundary, ChannelEnd, ChannelGeometry, ChannelReachDefinition, ChannelSurveyedConfig,
    CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell, TriangularChannelCell,
};
use hydr::coupling::{
    SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE, SurfaceChannelDefinition, SurfaceChannelError,
    SurfaceChannelPort, SurfaceChannelSetup,
};
use hydr::solver::{BoundaryConditions, BoundaryFace};

/// Bed of every `beta` cell in [`Scenario::rest`], in m.
const BETA_BED: f64 = 0.25;

/// Setup of the rest case with `inactive` surface cells and no ports.
fn setup(inactive: &[usize]) -> Result<SurfaceChannelSetup> {
    let mut case = Scenario::rest(1.0, 1.1, 0.5, Vec::new());
    case.inactive = inactive.to_vec();
    Ok(case.parts()?.0)
}

fn alpha_left(faces: Vec<BoundaryFace>) -> SurfaceChannelPort {
    port("p", ALPHA, ChannelEnd::Left, faces)
}

#[test]
fn ports_admit_beside_links_and_alone() -> Result {
    let mut ported = setup(&[])?;
    ported.ports = four_ports();
    SurfaceChannelDefinition::new(ported.clone())?;
    ported.links.clear();
    let definition = SurfaceChannelDefinition::new(ported)?;
    assert!(definition.surface_cells().is_empty());
    assert_eq!(definition.links().len(), 0);
    let mut empty = setup(&[])?;
    empty.links.clear();
    contains(
        &rejection(empty, Vec::new())?,
        &[
            "links is empty and ports is empty",
            "supply at least one surface-channel link or port",
        ],
    )
}

#[test]
fn a_triangular_end_rejects_as_a_section_mismatch() -> Result {
    let mut setup = setup(&[])?;
    setup.reaches[1].definition = ChannelReachDefinition {
        geometry: ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: BETA_BED,
            cells: vec![TriangularChannelCell {
                id: "cell-0".into(),
                length_m: 2.0,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        },
        ..rectangular(1, BETA_BED)
    };
    let ports = vec![port(
        "p",
        BETA,
        ChannelEnd::Right,
        vec![y_face(0, 0), y_face(1, 0)],
    )];
    contains(
        &rejection(setup, ports)?,
        &[
            "port-section-mismatch",
            "port=\"p\"",
            "reach=\"beta\"",
            "geometry=triangular",
            "use a rectangular reach",
        ],
    )
}

/// Setup whose `beta` reach is one surveyed cell with the profile `points` of (station, elevation).
fn surveyed_beta(points: [(f64, f64); 4]) -> Result<SurfaceChannelSetup> {
    let mut setup = setup(&[])?;
    setup.reaches[1].definition = ChannelReachDefinition {
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: "section".into(),
                vertical_datum: setup.reaches[1].definition.vertical_datum.clone(),
                points: points
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 4.0,
                profile_provenance: "Analytical rectangular survey".into(),
            },
            cells: vec![SurveyedChannelCell {
                id: "cell-0".into(),
                length_m: 2.0,
                manning_n: 0.0,
            }],
            numerical: ChannelSurveyedConfig::default(),
        },
        ..rectangular(1, BETA_BED)
    };
    Ok(setup)
}

fn beta_left() -> Vec<SurfaceChannelPort> {
    vec![port(
        "p",
        BETA,
        ChannelEnd::Left,
        vec![y_face(0, 0), y_face(1, 0)],
    )]
}

#[test]
fn a_surveyed_rectangle_end_of_the_face_width_admits() -> Result {
    let mut setup = surveyed_beta([(0.0, 4.0), (0.0, 0.0), (4.0, 0.0), (4.0, 4.0)])?;
    setup.ports = beta_left();
    SurfaceChannelDefinition::new(setup)?;
    Ok(())
}

#[test]
fn a_surveyed_rectangle_off_the_face_width_rejects() -> Result {
    let setup = surveyed_beta([(0.0, 4.0), (0.0, 0.0), (3.0, 0.0), (3.0, 4.0)])?;
    contains(
        &rejection(setup, beta_left())?,
        &[
            "port-width-mismatch",
            "port=\"p\"",
            "face width=4",
            "reach=\"beta\" end=left cell=\"cell-0\" width_m=3",
        ],
    )
}

#[test]
fn a_surveyed_end_that_is_not_a_rectangle_rejects_as_a_section_mismatch() -> Result {
    let setup = surveyed_beta([(0.0, 4.0), (1.0, 0.0), (3.0, 0.0), (4.0, 4.0)])?;
    contains(
        &rejection(setup, beta_left())?,
        &[
            "port-section-mismatch",
            "port=\"p\"",
            "reach=\"beta\"",
            "geometry=surveyed",
            "a port needs a rectangular surveyed section",
        ],
    )
}

#[test]
fn the_width_tolerance_is_sixteen_epsilon_of_the_end_width() -> Result {
    assert_eq!(
        SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE.to_bits(),
        (16.0 * f64::EPSILON).to_bits()
    );
    let set_width = |ulps: u64| -> Result<SurfaceChannelSetup> {
        let mut setup = setup(&[])?;
        let ChannelGeometry::Rectangular { cells } = &mut setup.reaches[0].definition.geometry
        else {
            return Err("alpha is rectangular".into());
        };
        cells[0].width_m = f64::from_bits(REACH_WIDTH.to_bits() + ulps);
        Ok(setup)
    };
    let faces = || vec![x_face(0, 0), x_face(0, 1)];
    // One ulp of 4 is 4 epsilon, so 16 ulps reach the 64 epsilon tolerance.
    let mut admitted = set_width(16)?;
    admitted.ports = vec![alpha_left(faces())];
    SurfaceChannelDefinition::new(admitted)?;
    contains(
        &rejection(set_width(17)?, vec![alpha_left(faces())])?,
        &[
            "port-width-mismatch",
            "port=\"p\"",
            "face width=4",
            "reach=\"alpha\" end=left cell=\"cell-0\"",
            "set the end cell width to the face width sum",
        ],
    )?;
    // One face of 2 m against a 4 m end.
    contains(
        &rejection(set_width(0)?, vec![alpha_left(vec![x_face(0, 0)])])?,
        &["port-width-mismatch", "face width=2", "width_m=4"],
    )
}

#[test]
fn faces_and_reach_ends_belong_to_one_port() -> Result {
    let setup = setup(&[])?;
    let mut shared = four_ports();
    shared[2].faces = vec![x_face(0, 1), x_face(0, 0)];
    contains(
        &rejection(setup.clone(), shared)?,
        &[
            "port-face-ownership",
            "port=\"p-beta-left\"",
            "is already bound to a port",
        ],
    )?;
    let mut twice = four_ports();
    twice[2].reach_id = ALPHA.into();
    twice[2].end = ChannelEnd::Left;
    contains(
        &rejection(setup.clone(), twice)?,
        &[
            "port-face-ownership",
            "reach=\"alpha\" end=left is already bound to port=\"p-alpha-left\"",
            "bind one port per reach end",
        ],
    )?;
    let mut unknown = four_ports();
    unknown[0].reach_id = "gamma".into();
    contains(
        &rejection(setup, unknown)?,
        &[
            "port-face-ownership",
            "reach_id=\"gamma\" names no setup reach",
        ],
    )
}

#[test]
fn each_bad_face_set_rejects() -> Result {
    let cases: [(&[usize], Vec<BoundaryFace>, &[&str]); 8] = [
        (&[], Vec::new(), &["has no faces"]),
        (&[], vec![x_face(9, 0)], &["face=", "lies outside the grid"]),
        (
            &[],
            vec![x_face(1, 0)],
            &["column: 1", "has two active cells"],
        ),
        (
            &[7],
            vec![x_face(4, 1)],
            &["column: 4", "has no active surface cell"],
        ),
        (
            &[],
            vec![x_face(0, 0), x_face(0, 0)],
            &["is already bound to a port"],
        ),
        (
            &[],
            vec![x_face(0, 0), y_face(0, 0)],
            &["bind faces of one axis on one face line"],
        ),
        (
            &[2, 5],
            vec![x_face(2, 0), x_face(2, 1)],
            &["bind faces with the surface on one side"],
        ),
        (
            &[],
            vec![y_face(0, 0), y_face(2, 0)],
            &["with a gap", "bind one contiguous face line"],
        ),
    ];
    for (inactive, faces, parts) in cases {
        let message = rejection(setup(inactive)?, vec![alpha_left(faces)])?;
        contains(&message, &["port-face-ownership", "port=\"p\""])?;
        contains(&message, parts)?;
    }
    Ok(())
}

#[test]
fn a_caller_open_face_rejects() -> Result {
    let mut setup = setup(&[])?;
    setup.surface.boundaries =
        BoundaryConditions::with_open_faces(&setup.surface.domain, vec![x_face(0, 1)])?;
    contains(
        &rejection(setup, vec![alpha_left(vec![x_face(0, 0), x_face(0, 1)])])?,
        &[
            "port-face-ownership",
            "is a caller open face",
            "remove it from the open faces or from the port",
        ],
    )
}

#[test]
fn port_identity_and_evidence_follow_the_link_rules() -> Result {
    let setup = setup(&[])?;
    let mut repeated = four_ports();
    repeated[1].id = repeated[0].id.clone();
    let mut as_link = four_ports();
    as_link[0].id = "bank".into();
    for ports in [repeated, as_link] {
        contains(
            &rejection(setup.clone(), ports)?,
            &["duplicate-links", "repeats a port or link identity"],
        )?;
    }
    let mut empty = four_ports();
    empty[0].id = " ".into();
    contains(
        &rejection(setup.clone(), empty)?,
        &["missing-link-evidence", "id=\" \""],
    )?;
    let mut control = four_ports();
    control[0].mapping_evidence = "bad\nevidence".into();
    contains(
        &rejection(setup, control)?,
        &["missing-link-evidence", "mapping_evidence="],
    )
}

#[test]
fn a_ported_reach_needs_the_surface_datum_and_gravity() -> Result {
    let mut other = setup(&[])?;
    other.reaches[1].definition.vertical_datum = "Other datum".into();
    contains(
        &rejection(other, four_ports())?,
        &["reach=\"beta\" vertical_datum=\"Other datum\""],
    )?;
    let mut other = setup(&[])?;
    other.reaches[1].definition.config.gravity = 9.8;
    contains(
        &rejection(other, four_ports())?,
        &["reach=\"beta\" gravity=9.8", "supply bit-identical gravity"],
    )
}

/// Input message of a rejected trial of `batch` under `forcing`.
fn trial_rejection(
    batch: &crate::ports_fixture::Batch,
    held: &hydr::coupling::SurfaceChannelForcingInterval,
) -> Result<String> {
    match batch.trial_to(0.1, held) {
        Ok(_) => Err("the trial admitted an invalid endpoint policy".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    }
}

#[test]
fn a_ported_end_takes_no_caller_policy_and_every_other_end_takes_one() -> Result {
    let batch = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]).build()?;
    let before = batch.snapshot_data()?;
    let mut owned = forcing(batch.setup(), 0.0, 0.1);
    owned.reaches[0].left = Some(ChannelBoundary::Wall);
    contains(
        &trial_rejection(&batch, &owned)?,
        &[
            "port-endpoint-policy",
            "reach=\"alpha\" left=Wall sits at port=\"p\"",
            "supply no caller policy at a ported end",
        ],
    )?;
    let mut missing = forcing(batch.setup(), 0.0, 0.1);
    missing.reaches[1].right = None;
    contains(
        &trial_rejection(&batch, &missing)?,
        &[
            "port-endpoint-policy",
            "reach=\"beta\" right has no port and no caller policy",
            "supply an endpoint policy",
        ],
    )?;
    assert_eq!(
        batch.snapshot_data()?,
        before,
        "a rejection changed the batch"
    );
    // The forcing with `None` exactly at the ported end admits.
    batch.trial_to(0.1, &forcing(batch.setup(), 0.0, 0.1))?;
    Ok(())
}
