//! Culvert laws of the structure link tests.

use super::scenario::TestResult;
use crate::solver::SolverConfig;
use crate::structures::{
    CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds, CulvertInlet,
    CulvertParameters, StructureLaw,
};

/// Two-way circular concrete culvert of rise 0.6 m and length 10 m between `inverts`.
///
/// Each end has a square-edge headwall: HDS-5 code 1 and entrance loss `0.5`. Its gravity is
/// bit-identical to the surface default.
pub fn culvert_law(inverts: [f64; 2]) -> TestResult<StructureLaw> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    Ok(StructureLaw::Culvert(CulvertParameters::new(
        CulvertBarrel::circular(0.6)?,
        CulvertAlignment {
            length_m: 10.0,
            invert_a_m: inverts[0],
            invert_b_m: inverts[1],
        },
        0.013,
        1,
        CulvertEnds {
            a: Some(end),
            b: Some(end),
        },
        SolverConfig::default().gravity,
        CulvertApplicability {
            coefficient_provenance: "HDS-5 Table A.1 code 1".into(),
            barrel_condition_assertion: "Clean synthetic barrel".into(),
        },
    )?))
}
