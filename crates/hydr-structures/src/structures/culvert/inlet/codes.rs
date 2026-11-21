//! HDS-5 Table A.1 inlet-control coefficients for circular and concrete box barrels.
//!
//! The codes follow SWMM `culvert.c`, whose rows equal the HDS-5 table. Values are the
//! decimals HDS-5 prints.

use super::CulvertInletForm::{self, One, Two};
use crate::structures::CulvertBarrelShape::{self, Box, Circular};

/// One coded inlet: barrel shape, form, `K`, `M`, `c`, `Y` and the mitered flag.
pub(super) struct CodedInlet {
    pub(super) shape: CulvertBarrelShape,
    pub(super) form: CulvertInletForm,
    pub(super) k: f64,
    pub(super) m: f64,
    pub(super) c: f64,
    pub(super) y: f64,
    pub(super) mitered: bool,
}

const fn coded(
    shape: CulvertBarrelShape,
    form: CulvertInletForm,
    [k, m, c, y]: [f64; 4],
    mitered: bool,
) -> CodedInlet {
    CodedInlet {
        shape,
        form,
        k,
        m,
        c,
        y,
        mitered,
    }
}

/// Codes 1 to 26 in order. Codes 27 to 57 need other barrel shapes or tapered inlets.
pub(super) const CODES: [CodedInlet; 26] = [
    coded(Circular, One, [0.0098, 2.0, 0.0398, 0.67], false),
    coded(Circular, One, [0.0018, 2.0, 0.0292, 0.74], false),
    coded(Circular, One, [0.0045, 2.0, 0.0317, 0.69], false),
    coded(Circular, One, [0.0078, 2.0, 0.0379, 0.69], false),
    coded(Circular, One, [0.0210, 1.33, 0.0463, 0.75], true),
    coded(Circular, One, [0.0340, 1.50, 0.0553, 0.54], false),
    coded(Circular, One, [0.0018, 2.50, 0.0300, 0.74], false),
    coded(Circular, One, [0.0018, 2.50, 0.0243, 0.83], false),
    coded(Box, One, [0.026, 1.0, 0.0347, 0.81], false),
    coded(Box, One, [0.061, 0.75, 0.0400, 0.80], false),
    coded(Box, One, [0.061, 0.75, 0.0423, 0.82], false),
    coded(Box, Two, [0.510, 0.667, 0.0309, 0.80], false),
    coded(Box, Two, [0.486, 0.667, 0.0249, 0.83], false),
    coded(Box, Two, [0.515, 0.667, 0.0375, 0.79], false),
    coded(Box, Two, [0.495, 0.667, 0.0314, 0.82], false),
    coded(Box, Two, [0.486, 0.667, 0.0252, 0.865], false),
    coded(Box, Two, [0.545, 0.667, 0.04505, 0.73], false),
    coded(Box, Two, [0.533, 0.667, 0.0425, 0.705], false),
    coded(Box, Two, [0.522, 0.667, 0.0402, 0.68], false),
    coded(Box, Two, [0.498, 0.667, 0.0327, 0.75], false),
    coded(Box, Two, [0.497, 0.667, 0.0339, 0.803], false),
    coded(Box, Two, [0.493, 0.667, 0.0361, 0.806], false),
    coded(Box, Two, [0.495, 0.667, 0.0386, 0.71], false),
    coded(Box, Two, [0.497, 0.667, 0.0302, 0.835], false),
    coded(Box, Two, [0.495, 0.667, 0.0252, 0.881], false),
    coded(Box, Two, [0.493, 0.667, 0.0227, 0.887], false),
];
