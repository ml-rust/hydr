//! Mixed culvert and weir lanes on any runtime, for layout and runtime checks.
//!
//! Each lane is one element and one head packet. `Debug` prints each F64 in round-trip form, so
//! equal text means equal bits, signed zero included.

use super::support::{Culvert, DATUM, TestResult, dry, heads, wet};
use hydr::structures::{
    CulvertBarrel, CulvertClient, CulvertInlet, HeadSupport, StructureBatch, StructureElement,
    StructureLaw, SuppressedRectangularWeirParameters, WeirApplicability,
};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Batch widths of the layout checks.
pub const WIDTHS: [usize; 3] = [1, 33, 65];

/// One element and the supports it is evaluated at.
#[derive(Clone, Debug)]
pub struct Lane {
    pub element: StructureElement,
    pub a: HeadSupport,
    pub b: HeadSupport,
}

fn weir(id: &str) -> TestResult<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: format!("a-{id}"),
        endpoint_b: format!("b-{id}"),
        datum: DATUM.into(),
        direction: hydr::structures::DirectionPolicy::Bidirectional,
        law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            2.0,
            0.5,
            1.7,
            WeirApplicability {
                coefficient_provenance: "Analytical coefficient".into(),
                free_nappe_and_approach_assertion: "Analytical suppressed crest".into(),
                submergence_correction_assertion: "Adopted Villemonte correction".into(),
                symmetric_reversal: true,
            },
        )?),
        condition: hydr::structures::StructureCondition::AsBuilt,
    })
}

/// One evidence barrel per arc shape with an HDS-5 code of its shape. The pipe-arch takes a
/// plain and a mitered code.
pub fn arc_barrels() -> TestResult<[(CulvertBarrel, u8); 5]> {
    Ok([
        (
            CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?,
            30,
        ),
        (
            CulvertBarrel::vertical_ellipse(0.9652, 1.524, 0.419_475_92)?,
            34,
        ),
        (
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            36,
        ),
        (
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            37,
        ),
        (CulvertBarrel::arch(3.048, 1.6002)?, 46),
    ])
}

/// Arc culverts of every shape: inlet control into a dry bed, reversed outlet control, level,
/// blocked at an undeclared end, drowned, and dry at both ends.
fn arc_culverts() -> TestResult<Vec<(Culvert, HeadSupport, HeadSupport)>> {
    let [ellipse, tall, pipe_arch, mitered, arch] = arc_barrels()?;
    let culvert = |(barrel, code): (CulvertBarrel, u8), inverts| -> TestResult<Culvert> {
        Ok(Culvert::both(
            barrel,
            inverts,
            CulvertInlet::hds5(code)?,
            0.5,
        ))
    };
    let mut long = culvert(tall, [0.0, 0.0])?;
    long.length_m = 100.0;
    long.manning_n = 0.024;
    let mut flap = culvert(mitered, [0.2, 0.0])?;
    flap.end_b = None;
    let low = culvert(arch, [0.2, 0.0])?;
    Ok(vec![
        (culvert(ellipse, [0.3, 0.0])?, wet(1.8), dry(0.0)),
        (long, wet(2.2), wet(2.6)),
        (culvert(pipe_arch, [0.1, 0.0])?, wet(1.0), wet(1.0)),
        (flap, wet(0.6), wet(1.4)),
        (low, wet(2.4), wet(1.2)),
        (low, dry(0.1), dry(-0.1)),
    ])
}

/// Distinct lanes: inlet- and outlet-controlled, drowned, reversed, blocked, level and dry
/// culverts of every barrel shape between weirs.
pub fn layout_lanes() -> TestResult<Vec<Lane>> {
    let circle = CulvertBarrel::circular(1.2)?;
    let rectangle = CulvertBarrel::rectangular_box(2.0, 1.5)?;
    let mut flap = Culvert::both(circle, [0.3, 0.0], CulvertInlet::hds5(1)?, 0.5);
    flap.end_b = None;
    let steep = Culvert::both(circle, [0.3, 0.0], CulvertInlet::hds5(3)?, 0.5);
    let mut long = Culvert::both(rectangle, [0.0, 0.0], CulvertInlet::hds5(9)?, 0.5);
    long.length_m = 150.0;
    long.manning_n = 0.03;
    let mut twin = Culvert::both(rectangle, [0.1, 0.0], CulvertInlet::hds5(26)?, 0.4);
    twin.count = 2;
    let mut culverts = vec![
        (flap, wet(1.4), dry(0.0)),
        (steep, wet(2.5), wet(0.2)),
        (flap, wet(0.5), wet(1.0)),
        (long, wet(2.0), wet(2.6)),
        (twin, wet(0.9), wet(0.9)),
        (twin, dry(0.1), wet(3.5)),
        (steep, dry(0.2), dry(-0.1)),
        (long, wet(1.1), wet(0.6)),
    ];
    culverts.extend(arc_culverts()?);
    let mut lanes = Vec::new();
    for (index, (culvert, a, b)) in culverts.into_iter().enumerate() {
        lanes.push(Lane {
            element: culvert.element(&format!("c{index}"))?,
            a,
            b,
        });
        if index % 3 == 0 {
            lanes.push(Lane {
                element: weir(&format!("w{index}"))?,
                a: wet(1.2),
                b: wet(0.7),
            });
        }
    }
    Ok(lanes)
}

/// Lane indices of a batch of `width` lanes, starting at offset `width` and wrapping.
pub fn layout(width: usize, count: usize) -> Vec<usize> {
    (0..width).map(|j| (width + j) % count).collect()
}

/// Proposals of `lanes` in one batch, one `Debug` text per lane. Element identities repeat
/// across lanes, so each lane takes a unique identity first.
pub fn outputs<R>(client: &R::Client, lanes: &[Lane]) -> TestResult<Vec<String>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R> + solvr::interpolate::Interp1dAlgorithms<R>,
{
    let elements: Vec<StructureElement> = lanes
        .iter()
        .enumerate()
        .map(|(index, lane)| StructureElement {
            id: format!("lane{index}"),
            ..lane.element.clone()
        })
        .collect();
    let packets: Vec<_> = elements
        .iter()
        .zip(lanes)
        .map(|(element, lane)| heads(element, lane.a.clone(), lane.b.clone()))
        .collect();
    let batch = StructureBatch::<R>::with_culverts(client.clone(), elements)?;
    let outputs = batch.propose(&packets)?.outputs()?;
    Ok(outputs
        .iter()
        .map(|output| {
            format!(
                "{:?} {:?} {:?} {:?} {:?}",
                output.regime,
                output.status,
                output.proposed_direction,
                output.discharge_m3_per_s,
                output.hydraulic_operands
            )
        })
        .collect())
}
