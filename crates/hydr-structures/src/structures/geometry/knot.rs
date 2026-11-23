//! Knots and parameter selectors for piecewise-linear and step structure geometry schedules.

/// Continuous scalar geometry parameter of a structure that a schedule can vary linearly or by step.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarGeometryParameter {
    /// Suppressed rectangular weir crest elevation, in metres.
    WeirCrestElevation,
    /// Suppressed rectangular weir crest length, in metres.
    WeirCrestLength,
    /// Suppressed rectangular weir blocked crest-length share in `[0, 1)`.
    WeirBlockage,
    /// Submerged orifice opening area, in m².
    OrificeArea,
    /// Submerged orifice sill elevation, in metres.
    OrificeSillElevation,
    /// Submerged orifice crown elevation, in metres.
    OrificeCrownElevation,
    /// Submerged orifice blocked area share in `[0, 1)`.
    OrificeBlockage,
    /// Hydrostatic rectangular gate aperture width, in metres.
    GateWidth,
    /// Hydrostatic rectangular gate sill elevation, in metres.
    GateSillElevation,
    /// Hydrostatic rectangular gate maximum opening, in metres.
    GateMaximumOpening,
    /// Hydrostatic rectangular gate blocked width share in `[0, 1)`.
    GateWidthBlockage,
    /// Hydrostatic rectangular gate blocked opening-range share in `[0, 1)`.
    GateOpeningRangeBlockage,
    /// Culvert entrance face A blocked share in `[0, 1)`.
    CulvertFaceABlockage,
    /// Culvert entrance face B blocked share in `[0, 1)`.
    CulvertFaceBBlockage,
}

impl ScalarGeometryParameter {
    /// Human-readable name for error messages.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::WeirCrestElevation => "WeirCrestElevation",
            Self::WeirCrestLength => "WeirCrestLength",
            Self::WeirBlockage => "WeirBlockage",
            Self::OrificeArea => "OrificeArea",
            Self::OrificeSillElevation => "OrificeSillElevation",
            Self::OrificeCrownElevation => "OrificeCrownElevation",
            Self::OrificeBlockage => "OrificeBlockage",
            Self::GateWidth => "GateWidth",
            Self::GateSillElevation => "GateSillElevation",
            Self::GateMaximumOpening => "GateMaximumOpening",
            Self::GateWidthBlockage => "GateWidthBlockage",
            Self::GateOpeningRangeBlockage => "GateOpeningRangeBlockage",
            Self::CulvertFaceABlockage => "CulvertFaceABlockage",
            Self::CulvertFaceBBlockage => "CulvertFaceBBlockage",
        }
    }
}

/// One knot of a scalar geometry schedule.
///
/// `left` applies immediately before `time_s`, and `right` applies at and immediately after
/// `time_s`. Equal `left` and `right` give a continuous piecewise-linear knot; distinct values
/// give a step change at `time_s`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructureGeometryKnot {
    /// Event time in seconds.
    pub time_s: f64,
    /// Value immediately before `time_s`.
    pub left: f64,
    /// Value at and immediately after `time_s`.
    pub right: f64,
}

impl StructureGeometryKnot {
    /// A continuous knot where `left` and `right` both equal `value`.
    pub const fn continuous(time_s: f64, value: f64) -> Self {
        Self {
            time_s,
            left: value,
            right: value,
        }
    }

    /// A step knot jumping from `left` before `time_s` to `right` at and after `time_s`.
    pub const fn step(time_s: f64, left: f64, right: f64) -> Self {
        Self {
            time_s,
            left,
            right,
        }
    }
}

/// One step knot of a discrete geometry schedule (`u32` closed barrels or `bool` closed state).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscreteGeometryKnot<T: Copy> {
    /// Event time in seconds.
    pub time_s: f64,
    /// Value before `time_s`.
    pub left: T,
    /// Value at and after `time_s`.
    pub right: T,
}

impl<T: Copy> DiscreteGeometryKnot<T> {
    /// A step knot jumping from `left` before `time_s` to `right` at and after `time_s`.
    pub const fn step(time_s: f64, left: T, right: T) -> Self {
        Self {
            time_s,
            left,
            right,
        }
    }
}

/// Typed track of one geometry schedule on a single structure.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum StructureGeometryTrack {
    /// Piecewise-linear scalar parameter (metres, m², or blocked share in `[0, 1)`).
    Scalar {
        /// Geometric parameter driven by this track.
        parameter: ScalarGeometryParameter,
        /// Strictly increasing knots in time.
        knots: Vec<StructureGeometryKnot>,
    },
    /// Piecewise-linear uniform culvert sediment depth in metres, with a fixed bed Manning `n`.
    ///
    /// A depth of `0.0` clears sediment.
    CulvertSediment {
        /// Manning roughness `n_bed` of the sediment bed, in `s/m^(1/3)`.
        bed_manning_n: f64,
        /// Strictly increasing knots of sediment depth in metres.
        knots: Vec<StructureGeometryKnot>,
    },
    /// Step schedule of closed culvert barrel counts `k` (`0 <= k < N`).
    CulvertClosedBarrels {
        /// Strictly increasing step knots of closed barrel counts.
        knots: Vec<DiscreteGeometryKnot<u32>>,
    },
    /// Step schedule of `StructureCondition::Closed` (`true` = closed, `false` = open).
    ///
    /// Closure is always a step change because no linear path reaches blockage fraction 1.
    Closed {
        /// Strictly increasing step knots of closed state.
        knots: Vec<DiscreteGeometryKnot<bool>>,
    },
}

/// Discriminant identifying which parameter of a structure a track writes, for duplicate checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum TrackKey {
    Scalar(u8),
    CulvertSediment,
    CulvertClosedBarrels,
    Closed,
}

impl StructureGeometryTrack {
    pub(crate) fn key(&self) -> TrackKey {
        match self {
            Self::Scalar { parameter, .. } => TrackKey::Scalar(*parameter as u8),
            Self::CulvertSediment { .. } => TrackKey::CulvertSediment,
            Self::CulvertClosedBarrels { .. } => TrackKey::CulvertClosedBarrels,
            Self::Closed { .. } => TrackKey::Closed,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Scalar { parameter, .. } => parameter.name(),
            Self::CulvertSediment { .. } => "CulvertSediment",
            Self::CulvertClosedBarrels { .. } => "CulvertClosedBarrels",
            Self::Closed { .. } => "Closed",
        }
    }

    /// Knot times in ascending order.
    pub fn knot_times(&self) -> Vec<f64> {
        match self {
            Self::Scalar { knots, .. } | Self::CulvertSediment { knots, .. } => {
                knots.iter().map(|k| k.time_s).collect()
            }
            Self::CulvertClosedBarrels { knots } => knots.iter().map(|k| k.time_s).collect(),
            Self::Closed { knots } => knots.iter().map(|k| k.time_s).collect(),
        }
    }
}
