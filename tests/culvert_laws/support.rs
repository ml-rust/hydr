//! Culvert structures, head packets and proposals shared by the culvert law tests.

use hydr::structures::{
    CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds, CulvertInlet,
    CulvertParameters, DirectionPolicy, EndpointHead, HeadConvention, HeadSupport, StructureBatch,
    StructureElement, StructureHeads, StructureLaw, StructureOutput,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Common datum of every fixture element.
pub const DATUM: &str = "Culvert fixture datum";

/// Gravity of every fixture culvert, the corpus value.
pub const GRAVITY: f64 = 9.81;

/// Sequential CPU client.
pub fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
}

/// One culvert: barrel, inverts, length, roughness, count and the inlet code and `ke` per end.
#[derive(Clone, Copy, Debug)]
pub struct Culvert {
    pub barrel: CulvertBarrel,
    pub invert_a_m: f64,
    pub invert_b_m: f64,
    pub length_m: f64,
    pub manning_n: f64,
    pub count: u32,
    pub end_a: Option<(CulvertInlet, f64)>,
    pub end_b: Option<(CulvertInlet, f64)>,
}

impl Culvert {
    /// A culvert with the same inlet and `ke` at both ends.
    pub fn both(barrel: CulvertBarrel, inverts: [f64; 2], inlet: CulvertInlet, ke: f64) -> Self {
        Self {
            barrel,
            invert_a_m: inverts[0],
            invert_b_m: inverts[1],
            length_m: 30.0,
            manning_n: 0.013,
            count: 1,
            end_a: Some((inlet, ke)),
            end_b: Some((inlet, ke)),
        }
    }

    /// The culvert law.
    pub fn law(&self) -> TestResult<StructureLaw> {
        let end = |side: Option<(CulvertInlet, f64)>| -> TestResult<Option<CulvertEnd>> {
            Ok(match side {
                Some((inlet, ke)) => Some(CulvertEnd::new(inlet, ke)?),
                None => None,
            })
        };
        Ok(StructureLaw::Culvert(CulvertParameters::new(
            self.barrel,
            CulvertAlignment {
                length_m: self.length_m,
                invert_a_m: self.invert_a_m,
                invert_b_m: self.invert_b_m,
            },
            self.manning_n,
            self.count,
            CulvertEnds {
                a: end(self.end_a)?,
                b: end(self.end_b)?,
            },
            GRAVITY,
            CulvertApplicability {
                coefficient_provenance: "HDS-5 Table A.1 fixture coefficients".into(),
                barrel_condition_assertion: "Clean synthetic barrel without storage".into(),
            },
        )?))
    }

    /// Direction policy that matches the declared ends.
    pub fn direction(&self) -> DirectionPolicy {
        match (self.end_a.is_some(), self.end_b.is_some()) {
            (true, false) => DirectionPolicy::AtoBOnly,
            (false, true) => DirectionPolicy::BtoAOnly,
            _ => DirectionPolicy::Bidirectional,
        }
    }

    /// Element `id` between `a-{id}` and `b-{id}`.
    pub fn element(&self, id: &str) -> TestResult<StructureElement> {
        Ok(StructureElement {
            id: id.into(),
            endpoint_a: format!("a-{id}"),
            endpoint_b: format!("b-{id}"),
            datum: DATUM.into(),
            direction: self.direction(),
            law: self.law()?,
        })
    }
}

/// Wet stage.
pub fn wet(stage_m: f64) -> HeadSupport {
    HeadSupport::Wet { stage_m }
}

/// Known dry bed.
pub fn dry(bed_elevation_m: f64) -> HeadSupport {
    HeadSupport::Dry { bed_elevation_m }
}

/// Head packet of `element` with supports `a` and `b`.
pub fn heads(element: &StructureElement, a: HeadSupport, b: HeadSupport) -> StructureHeads {
    StructureHeads {
        structure_id: element.id.clone(),
        endpoint_a: EndpointHead {
            id: element.endpoint_a.clone(),
            support: a,
        },
        endpoint_b: EndpointHead {
            id: element.endpoint_b.clone(),
            support: b,
        },
        datum: element.datum.clone(),
        convention: HeadConvention::QuiescentStage,
    }
}

/// Proposals of `elements` at `supports`, one pair per element, on the sequential CPU client.
pub fn propose(
    elements: &[StructureElement],
    supports: &[(HeadSupport, HeadSupport)],
) -> TestResult<Vec<StructureOutput>> {
    let batch = StructureBatch::<CpuRuntime>::with_culverts(client(), elements.to_vec())?;
    let packets: Vec<StructureHeads> = elements
        .iter()
        .zip(supports)
        .map(|(element, (a, b))| heads(element, a.clone(), b.clone()))
        .collect();
    Ok(batch.propose(&packets)?.outputs()?)
}

/// One element per culvert, named `c0`, `c1`, and so on.
pub fn elements(culverts: &[Culvert]) -> TestResult<Vec<StructureElement>> {
    culverts
        .iter()
        .enumerate()
        .map(|(index, culvert)| culvert.element(&format!("c{index}")))
        .collect()
}

/// `true` when `value` lies within `relative` of `reference`.
pub fn close(value: f64, reference: f64, relative: f64) -> bool {
    (value - reference).abs() <= relative * reference.abs()
}
