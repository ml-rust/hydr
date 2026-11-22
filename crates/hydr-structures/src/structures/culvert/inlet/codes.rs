//! HDS-5 inlet-control coefficients by SWMM `culvert.c` code.
//!
//! The codes follow SWMM `culvert.c`, whose rows equal HDS-5 Tables A.1 and A.2. Values are
//! the decimals HDS-5 prints.
//!
//! | Codes | Barrel shape | Admitted |
//! | --- | --- | --- |
//! | 1 to 8 | circular | yes, code 5 mitered |
//! | 9 to 26 | concrete box | yes |
//! | 27 to 29 | corrugated metal box | no: no barrel shape models it |
//! | 30 to 32 | horizontal ellipse | yes |
//! | 33 to 35 | vertical ellipse | yes |
//! | 36 to 44 | pipe-arch | yes, code 37 mitered |
//! | 45 to 47 | flat-floor arch | yes, code 46 mitered |
//! | 48 to 57 | tapered inlets | no: they need throat and face geometry |
//!
//! A code admits by barrel shape class only. The corner radius in the HDS-5 chart labels of
//! codes 36 to 44 (18 in, 31 in) is not checked.
//!
//! HDS-5 disagrees with itself on chart 34 scale 3 (code 38). Section A.3.1 prints `Y = 0.53`
//! and Table A.2 prints `Y = 0.57`. This table takes Table A.2, as SWMM does.

use super::CulvertInletForm::{self, One, Two};
use crate::structures::CulvertBarrelShape::{
    self, Arch, Box, Circular, HorizontalEllipse, PipeArch, VerticalEllipse,
};

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

/// First code of [`ARC_CODES`].
const FIRST_ARC_CODE: usize = 30;

/// Codes 1 to 26 in order: circular and concrete box barrels, HDS-5 Table A.1.
const CODES: [CodedInlet; 26] = [
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

/// Codes 30 to 47 in order: tangent-arc barrels, HDS-5 Table A.2. All are form 1.
const ARC_CODES: [CodedInlet; 18] = [
    coded(HorizontalEllipse, One, [0.0100, 2.0, 0.0398, 0.67], false),
    coded(HorizontalEllipse, One, [0.0018, 2.5, 0.0292, 0.74], false),
    coded(HorizontalEllipse, One, [0.0045, 2.0, 0.0317, 0.69], false),
    coded(VerticalEllipse, One, [0.0100, 2.0, 0.0398, 0.67], false),
    coded(VerticalEllipse, One, [0.0018, 2.5, 0.0292, 0.74], false),
    coded(VerticalEllipse, One, [0.0095, 2.0, 0.0317, 0.69], false),
    coded(PipeArch, One, [0.0083, 2.0, 0.0379, 0.69], false),
    coded(PipeArch, One, [0.0300, 1.0, 0.0463, 0.75], true),
    coded(PipeArch, One, [0.0340, 1.5, 0.0496, 0.57], false),
    coded(PipeArch, One, [0.0300, 1.5, 0.0496, 0.57], false),
    coded(PipeArch, One, [0.0088, 2.0, 0.0368, 0.68], false),
    coded(PipeArch, One, [0.0030, 2.0, 0.0269, 0.77], false),
    coded(PipeArch, One, [0.0300, 1.5, 0.0496, 0.57], false),
    coded(PipeArch, One, [0.0088, 2.0, 0.0368, 0.68], false),
    coded(PipeArch, One, [0.0030, 2.0, 0.0269, 0.77], false),
    coded(Arch, One, [0.0083, 2.0, 0.0379, 0.69], false),
    coded(Arch, One, [0.0300, 1.0, 0.0473, 0.75], true),
    coded(Arch, One, [0.0340, 1.5, 0.0496, 0.57], false),
];

/// Why a code has no entry.
pub(super) enum Unadmitted {
    /// Codes 27 to 29: corrugated metal box inlets.
    MetalBox,
    /// Codes 48 to 57: tapered inlets.
    Tapered,
    /// Not an HDS-5 code.
    Unknown,
}

/// The entry of `code`, or why it has none.
pub(super) fn lookup(code: u8) -> Result<&'static CodedInlet, Unadmitted> {
    let index = usize::from(code);
    match index {
        1..=26 => CODES.get(index - 1).ok_or(Unadmitted::Unknown),
        27..=29 => Err(Unadmitted::MetalBox),
        30..=47 => ARC_CODES
            .get(index - FIRST_ARC_CODE)
            .ok_or(Unadmitted::Unknown),
        48..=57 => Err(Unadmitted::Tapered),
        _ => Err(Unadmitted::Unknown),
    }
}
