//! Arc, circle and box lanes for every barrel, inlet and outlet query, on any runtime.
//!
//! Arc lanes take frozen rows of the first evidence barrel of each shape. One circle or box lane
//! follows every third arc lane. The layouts and query runners live in `lane_outputs`.

use super::depth_rows::{CERTIFIED, CRITICAL, MANNING_N, NORMAL};
use super::inlet_rows::{self, InletRow, SLOPES};
use super::outlet_rows::{self, OutletRow, PROFILE_ROWS};
use super::reference_rows::{self, BARRELS, ROWS};
use hydr::structures::{
    CulvertBarrel, CulvertBarrelShape, CulvertInlet, CulvertInletRow, CulvertOutletRow,
};

pub type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Batch widths: one row, and one past each of 32 and 64 lanes.
pub const WIDTHS: [usize; 3] = [1, 33, 65];

/// One public barrel query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Query {
    /// `geometry` at `values[0]`.
    Geometry,
    /// `geometry_over` from `values[0]` to `values[1]`.
    Range,
    /// `full_geometry`, which reads no values.
    Full,
    /// `critical_depth` for discharge `values[0]` at the frozen gravity.
    Critical,
    /// `normal_depth` for discharge, Manning n and slope in `values`.
    Normal,
}

pub const QUERIES: [Query; 5] = [
    Query::Geometry,
    Query::Range,
    Query::Full,
    Query::Critical,
    Query::Normal,
];

/// One barrel row and the query inputs it reads.
#[derive(Clone, Copy, Debug)]
pub struct BarrelLane {
    pub barrel: CulvertBarrel,
    pub values: [f64; 3],
}

/// One inlet row and the discharge it is queried at.
#[derive(Clone, Copy, Debug)]
pub struct InletLane {
    pub row: CulvertInletRow,
    pub discharge: f64,
}

/// One outlet row with the discharge and tailwater it is queried at.
#[derive(Clone, Copy, Debug)]
pub struct OutletLane {
    pub row: CulvertOutletRow,
    pub discharge: f64,
    pub tailwater: f64,
}

/// The barrel of `shape` from fixture `inputs`: span, rise, then the given radii.
fn arc(shape: CulvertBarrelShape, name: &str, inputs: &[u64]) -> Outcome<CulvertBarrel> {
    let values: Vec<f64> = inputs.iter().map(|bits| f64::from_bits(*bits)).collect();
    Ok(match (shape, values.as_slice()) {
        (CulvertBarrelShape::HorizontalEllipse, &[b, d, r]) => {
            CulvertBarrel::horizontal_ellipse(b, d, r)?
        }
        (CulvertBarrelShape::VerticalEllipse, &[b, d, r]) => {
            CulvertBarrel::vertical_ellipse(b, d, r)?
        }
        (CulvertBarrelShape::PipeArch, &[b, d, c, r]) => CulvertBarrel::pipe_arch(b, d, c, r)?,
        (CulvertBarrelShape::Arch, &[b, d]) => CulvertBarrel::arch(b, d)?,
        _ => return Err(format!("fixture barrel {name}: wrong input count").into()),
    })
}

/// Defines `$name`, the barrel shape of the fixture shape `$fixture::Shape`. Each fixture
/// declares its own copy of the four arc shapes.
macro_rules! shape_of {
    ($(#[$doc:meta])* $name:ident, $fixture:ident) => {
        $(#[$doc])*
        pub fn $name(shape: $fixture::Shape) -> CulvertBarrelShape {
            match shape {
                $fixture::Shape::HorizontalEllipse => CulvertBarrelShape::HorizontalEllipse,
                $fixture::Shape::VerticalEllipse => CulvertBarrelShape::VerticalEllipse,
                $fixture::Shape::PipeArch => CulvertBarrelShape::PipeArch,
                $fixture::Shape::Arch => CulvertBarrelShape::Arch,
            }
        }
    };
}

shape_of!(
    /// Barrel shape of a geometry or depth fixture shape.
    geometry_shape,
    reference_rows
);
shape_of!(
    /// Barrel shape of an inlet fixture shape.
    inlet_shape,
    inlet_rows
);
shape_of!(
    /// Barrel shape of an outlet fixture shape.
    outlet_shape,
    outlet_rows
);

/// The barrel of a geometry or depth fixture entry.
pub fn barrel(entry: &reference_rows::Barrel) -> Outcome<CulvertBarrel> {
    arc(geometry_shape(entry.shape), entry.name, entry.inputs)
}

/// The barrel of inlet fixture barrel `index`.
pub fn inlet_barrel(index: usize) -> Outcome<CulvertBarrel> {
    let entry = inlet_rows::BARRELS
        .get(index)
        .ok_or_else(|| format!("no inlet fixture barrel {index}"))?;
    arc(inlet_shape(entry.shape), entry.name, entry.inputs)
}

/// The barrel of outlet fixture barrel `index`.
pub fn outlet_barrel(index: usize) -> Outcome<CulvertBarrel> {
    let entry = outlet_rows::BARRELS
        .get(index)
        .ok_or_else(|| format!("no outlet fixture barrel {index}"))?
        .barrel;
    arc(outlet_shape(entry.shape), entry.name, entry.inputs)
}

/// The first geometry evidence barrel of each arc shape, in fixture order.
pub fn arc_barrels() -> Outcome<Vec<CulvertBarrel>> {
    first_of_each(BARRELS, |entry| geometry_shape(entry.shape))
        .into_iter()
        .map(|index| barrel(&BARRELS[index]))
        .collect()
}

/// Index of the first entry of each shape in `entries`, in fixture order.
fn first_of_each<T>(entries: &[T], shape: impl Fn(&T) -> CulvertBarrelShape) -> Vec<usize> {
    let mut seen = Vec::new();
    let mut first = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let found = shape(entry);
        if !seen.contains(&found) {
            seen.push(found);
            first.push(index);
        }
    }
    first
}

/// `arcs` with one `closed` lane after every third, the closed lanes taken in turn.
fn mixed<T: Copy>(arcs: Vec<T>, closed: &[T]) -> Vec<T> {
    let mut made = Vec::with_capacity(arcs.len() + arcs.len() / 3);
    let mut extra = closed.iter().cycle();
    for (k, lane) in arcs.into_iter().enumerate() {
        made.push(lane);
        if k % 3 == 2
            && let Some(next) = extra.next()
        {
            made.push(*next);
        }
    }
    made
}

/// The circle and the box beside the arc lanes.
fn closed() -> Outcome<[CulvertBarrel; 2]> {
    Ok([
        CulvertBarrel::circular(1.3)?,
        CulvertBarrel::rectangular_box(2.0, 1.1)?,
    ])
}

/// Circle and box depth fractions of their rise. The range query skips those at the crown.
const CLOSED_FRACTIONS: [f64; 6] = [0.0, 1e-3, 0.05, 0.5, 0.999, 1.25];

/// Circle and box critical discharges, dry to above the crown, in m³/s.
const CLOSED_CRITICAL: [f64; 4] = [0.0, 0.3, 2.0, 50.0];

/// Circle and box normal-depth discharge, Manning n and slope, up to above capacity.
const CLOSED_NORMAL: [[f64; 3]; 4] = [
    [0.5, 0.013, 0.01],
    [3.0, 0.013, 0.001],
    [100.0, 0.013, 0.01],
    [0.5, 0.013, 0.0],
];

/// Lanes of `query`: arc rows of one barrel per shape, circles and boxes interleaved.
pub fn barrel_lanes(query: Query) -> Outcome<Vec<BarrelLane>> {
    let closed = closed()?;
    let mut extra = Vec::new();
    let arcs: Vec<BarrelLane> = match query {
        Query::Geometry | Query::Range | Query::Full => {
            let range = query == Query::Range;
            for fraction in CLOSED_FRACTIONS {
                for culvert in closed {
                    let depth = fraction * culvert.rise_m();
                    if !range || depth < culvert.rise_m() {
                        extra.push(BarrelLane {
                            barrel: culvert,
                            values: if range {
                                [0.5 * depth, depth, 0.0]
                            } else {
                                [depth, 0.0, 0.0]
                            },
                        });
                    }
                }
            }
            let owners = first_of_each(BARRELS, |entry| geometry_shape(entry.shape));
            let mut made = Vec::new();
            for row in ROWS.iter().filter(|row| owners.contains(&row.barrel)) {
                let culvert = barrel(&BARRELS[row.barrel])?;
                let depth = f64::from_bits(row.depth);
                if range && depth >= culvert.rise_m() {
                    continue;
                }
                made.push(BarrelLane {
                    barrel: culvert,
                    values: if range {
                        [0.5 * depth, depth, 0.0]
                    } else {
                        [depth, 0.0, 0.0]
                    },
                });
            }
            made
        }
        Query::Critical => {
            for discharge in CLOSED_CRITICAL {
                for culvert in closed {
                    extra.push(BarrelLane {
                        barrel: culvert,
                        values: [discharge, 0.0, 0.0],
                    });
                }
            }
            let owners = first_of_each(CERTIFIED, |entry| geometry_shape(entry.barrel.shape));
            CRITICAL
                .iter()
                .filter(|row| owners.contains(&row.barrel))
                .map(|row| {
                    Ok(BarrelLane {
                        barrel: barrel(&CERTIFIED[row.barrel].barrel)?,
                        values: [f64::from_bits(row.discharge), 0.0, 0.0],
                    })
                })
                .collect::<Outcome<_>>()?
        }
        Query::Normal => {
            for values in CLOSED_NORMAL {
                for culvert in closed {
                    extra.push(BarrelLane {
                        barrel: culvert,
                        values,
                    });
                }
            }
            normal_arcs()?
        }
    };
    Ok(mixed(arcs, &extra))
}

/// Normal-depth rows at the first fixture slope of one barrel per shape. They run from dry
/// past the conveyance peak, with zero and adverse slopes.
fn normal_arcs() -> Outcome<Vec<BarrelLane>> {
    let owners = first_of_each(CERTIFIED, |entry| geometry_shape(entry.barrel.shape));
    let mut made = Vec::new();
    for owner in owners {
        let entry = &CERTIFIED[owner];
        let rows: Vec<_> = NORMAL.iter().filter(|row| row.barrel == owner).collect();
        let first = rows.first().ok_or("barrel without normal rows")?.slope;
        // Each slope's rows end with the zero and adverse slope rows.
        let count = rows.iter().take_while(|row| row.slope == first).count() + 2;
        for row in rows.into_iter().take(count) {
            made.push(BarrelLane {
                barrel: barrel(&entry.barrel)?,
                values: [row.discharge, MANNING_N, row.slope].map(f64::from_bits),
            });
        }
    }
    Ok(made)
}

/// Inlet codes of the lanes: each shape, mitered ends and the second pipe-arch.
const INLET_CODES: [u8; 7] = [30, 34, 36, 37, 42, 45, 46];

/// Inlet row labels of the lanes: unsubmerged, both transition ends, submerged.
const INLET_LABELS: [&str; 4] = ["0.5", "3.5+", "3.75", "5.0"];

/// Inlet row of the frozen barrel, code and slope of `row`, at the frozen gravity.
pub fn inlet_row(row: &InletRow) -> Outcome<CulvertInletRow> {
    let slope = SLOPES
        .get(row.slope)
        .ok_or_else(|| format!("{}: no slope {}", row.label, row.slope))?;
    Ok(CulvertInletRow {
        barrel: inlet_barrel(row.barrel)?,
        inlet: CulvertInlet::hds5(row.code)?,
        slope: f64::from_bits(*slope),
        gravity_m_per_s2: f64::from_bits(inlet_rows::GRAVITY),
        face_blockage: hydr::structures::BlockageFraction::NONE,
    })
}

/// Inlet lanes: frozen rows of `INLET_CODES` at `INLET_LABELS`, the slope turning per code,
/// circles and boxes interleaved.
pub fn inlet_lanes() -> Outcome<Vec<InletLane>> {
    let mut arcs = Vec::new();
    for (k, code) in INLET_CODES.into_iter().enumerate() {
        let slope = k % SLOPES.len();
        for label in INLET_LABELS {
            let row = inlet_rows::ROWS
                .iter()
                .find(|row| row.code == code && row.slope == slope && row.label == label)
                .ok_or_else(|| format!("no inlet row {code} {slope} {label}"))?;
            arcs.push(InletLane {
                row: inlet_row(row)?,
                discharge: f64::from_bits(row.discharge),
            });
        }
    }
    let [circle, square] = closed()?;
    let gravity = f64::from_bits(inlet_rows::GRAVITY);
    let row = |barrel, code| -> Outcome<CulvertInletRow> {
        Ok(CulvertInletRow {
            barrel,
            inlet: CulvertInlet::hds5(code)?,
            slope: 0.01,
            gravity_m_per_s2: gravity,
            face_blockage: hydr::structures::BlockageFraction::NONE,
        })
    };
    let extra = [
        (row(circle, 1)?, 0.5),
        (row(circle, 3)?, 6.0),
        (row(square, 9)?, 2.0),
        (row(square, 26)?, 12.0),
    ]
    .map(|(row, discharge)| InletLane { row, discharge });
    Ok(mixed(arcs, &extra))
}

/// The lane of one frozen outlet row at the frozen gravity.
pub fn outlet_lane(row: &OutletRow) -> Outcome<OutletLane> {
    let inputs = &row.inputs;
    Ok(OutletLane {
        row: CulvertOutletRow {
            barrel: outlet_barrel(inputs.barrel)?,
            length_m: f64::from_bits(inputs.length),
            manning_n: f64::from_bits(inputs.manning),
            entrance_loss: f64::from_bits(inputs.ke),
            slope: f64::from_bits(inputs.slope),
            gravity_m_per_s2: f64::from_bits(outlet_rows::GRAVITY),
            face_blockage: hydr::structures::BlockageFraction::NONE,
        },
        discharge: f64::from_bits(inputs.discharge),
        tailwater: f64::from_bits(inputs.tailwater),
    })
}

/// Outlet profile rows of the lanes, one per regime: full, partly full with a critical outlet,
/// supercritical, jump, and critical inlet with a full exit.
const OUTLET_NAMES: [&str; 5] = [
    "full_mild",
    "m2_critical_outlet",
    "steep_supercritical",
    "steep_jump",
    "drowned_spanning",
];

/// Outlet lanes: `OUTLET_NAMES` rows of every outlet fixture barrel, circles and boxes
/// interleaved.
pub fn outlet_lanes() -> Outcome<Vec<OutletLane>> {
    let mut arcs = Vec::new();
    for index in 0..outlet_rows::BARRELS.len() {
        for name in OUTLET_NAMES {
            let row = PROFILE_ROWS
                .iter()
                .find(|row| row.inputs.barrel == index && row.name == name)
                .ok_or_else(|| format!("no outlet row {name} for barrel {index}"))?;
            arcs.push(outlet_lane(row)?);
        }
    }
    let [circle, square] = closed()?;
    let row = |barrel, slope| CulvertOutletRow {
        barrel,
        length_m: 40.0,
        manning_n: 0.013,
        entrance_loss: 0.5,
        slope,
        gravity_m_per_s2: 9.81,
        face_blockage: hydr::structures::BlockageFraction::NONE,
    };
    let extra = [
        (row(circle, 0.001), 0.8, 1.4),
        (row(circle, 0.03), 0.8, 0.7),
        (row(square, 0.001), 3.0, 0.0),
        (row(square, 0.02), 3.0, 2.0),
    ]
    .map(|(row, discharge, tailwater)| OutletLane {
        row,
        discharge,
        tailwater,
    });
    Ok(mixed(arcs, &extra))
}
