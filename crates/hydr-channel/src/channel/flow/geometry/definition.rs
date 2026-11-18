use super::super::{ChannelError, RectangularChannelCell};
use super::surveyed_config::ChannelSurveyedConfig;
use crate::channel::section::CrossSectionDefinition;

/// One surveyed-profile cell in increasing longitudinal order.
#[derive(Clone, Debug, PartialEq)]
pub struct SurveyedChannelCell {
    /// Unique stable cell identity.
    pub id: String,
    /// Longitudinal length in metres.
    pub length_m: f64,
    /// Manning roughness in s/m^(1/3), including zero for frictionless flow.
    pub manning_n: f64,
}

/// One triangular cell in increasing longitudinal order.
#[derive(Clone, Debug, PartialEq)]
pub struct TriangularChannelCell {
    /// Unique stable cell identity.
    pub id: String,
    /// Longitudinal length in metres.
    pub length_m: f64,
    /// Highest supported depth in metres.
    pub maximum_depth_m: f64,
    /// Manning roughness in s/m^(1/3), including zero for frictionless flow.
    pub manning_n: f64,
}

/// Declared section geometry for one connected reach.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelGeometry {
    /// One horizontal surveyed section shared by every cell.
    PrismaticSurveyed {
        /// Original surveyed profile and its declared vertical datum.
        profile: CrossSectionDefinition,
        /// Ordered connected cells.
        cells: Vec<SurveyedChannelCell>,
        /// Geometry allowances and native work ceilings.
        numerical: ChannelSurveyedConfig,
    },
    /// Constant rectangular sections with explicit local beds and widths.
    Rectangular {
        /// Ordered connected cells.
        cells: Vec<RectangularChannelCell>,
    },
    /// One horizontal triangular section shared by every cell.
    PrismaticTriangular {
        /// Left bank horizontal metres per vertical metre.
        left_bank_slope_h_per_v: f64,
        /// Right bank horizontal metres per vertical metre.
        right_bank_slope_h_per_v: f64,
        /// Common bed elevation in the declared datum, in metres.
        bed_elevation_m: f64,
        /// Ordered connected cells.
        cells: Vec<TriangularChannelCell>,
    },
}

#[derive(Clone, Copy)]
pub(in crate::channel::flow) struct CellMetadata<'a> {
    pub id: &'a str,
    pub length_m: f64,
    pub bed_elevation_m: f64,
    pub maximum_depth_m: f64,
    pub manning_n: f64,
}

impl ChannelGeometry {
    pub(in crate::channel::flow) fn len(&self) -> usize {
        match self {
            Self::PrismaticSurveyed { cells, .. } => cells.len(),
            Self::Rectangular { cells } => cells.len(),
            Self::PrismaticTriangular { cells, .. } => cells.len(),
        }
    }

    pub(in crate::channel::flow) fn cell(&self, index: usize) -> CellMetadata<'_> {
        match self {
            Self::PrismaticSurveyed { profile, cells, .. } => {
                let c = &cells[index];
                let bed = match profile
                    .points
                    .iter()
                    .map(|p| p.elevation_m)
                    .reduce(f64::min)
                {
                    Some(value) => value,
                    None => profile.maximum_stage_m,
                };
                CellMetadata {
                    id: &c.id,
                    length_m: c.length_m,
                    bed_elevation_m: bed,
                    maximum_depth_m: profile.maximum_stage_m - bed,
                    manning_n: c.manning_n,
                }
            }
            Self::Rectangular { cells } => {
                let c = &cells[index];
                CellMetadata {
                    id: &c.id,
                    length_m: c.length_m,
                    bed_elevation_m: c.bed_elevation_m,
                    maximum_depth_m: c.maximum_depth_m,
                    manning_n: c.manning_n,
                }
            }
            Self::PrismaticTriangular {
                cells,
                bed_elevation_m,
                ..
            } => {
                let c = &cells[index];
                CellMetadata {
                    id: &c.id,
                    length_m: c.length_m,
                    bed_elevation_m: *bed_elevation_m,
                    maximum_depth_m: c.maximum_depth_m,
                    manning_n: c.manning_n,
                }
            }
        }
    }

    pub(in crate::channel::flow) fn cells(
        &self,
    ) -> impl ExactSizeIterator<Item = CellMetadata<'_>> {
        let surveyed = match self {
            Self::PrismaticSurveyed { cells, .. } if !cells.is_empty() => Some(self.cell(0)),
            _ => None,
        };
        (0..self.len()).map(move |index| {
            if let (Self::PrismaticSurveyed { cells, .. }, Some(common)) = (self, surveyed) {
                let c = &cells[index];
                CellMetadata {
                    id: &c.id,
                    length_m: c.length_m,
                    bed_elevation_m: common.bed_elevation_m,
                    maximum_depth_m: common.maximum_depth_m,
                    manning_n: c.manning_n,
                }
            } else {
                self.cell(index)
            }
        })
    }

    pub(in crate::channel::flow) fn same_bits(&self, other: &Self) -> bool {
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits();
        let same_section = match (self, other) {
            (
                Self::PrismaticSurveyed {
                    profile: a,
                    numerical: an,
                    ..
                },
                Self::PrismaticSurveyed {
                    profile: b,
                    numerical: bn,
                    ..
                },
            ) => {
                a.id == b.id
                    && a.vertical_datum == b.vertical_datum
                    && a.profile_provenance == b.profile_provenance
                    && same(a.maximum_stage_m, b.maximum_stage_m)
                    && a.points.len() == b.points.len()
                    && a.points.iter().zip(&b.points).all(|(a, b)| {
                        same(a.station_m, b.station_m) && same(a.elevation_m, b.elevation_m)
                    })
                    && an.same_bits(bn)
            }
            (Self::Rectangular { cells: a }, Self::Rectangular { cells: b }) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a.width_m, b.width_m))
            }
            (
                Self::PrismaticTriangular {
                    left_bank_slope_h_per_v: al,
                    right_bank_slope_h_per_v: ar,
                    bed_elevation_m: ab,
                    ..
                },
                Self::PrismaticTriangular {
                    left_bank_slope_h_per_v: bl,
                    right_bank_slope_h_per_v: br,
                    bed_elevation_m: bb,
                    ..
                },
            ) => same(*al, *bl) && same(*ar, *br) && same(*ab, *bb),
            _ => false,
        };
        same_section
            && self.len() == other.len()
            && self.cells().zip(other.cells()).all(|(a, b)| {
                a.id == b.id
                    && same(a.length_m, b.length_m)
                    && same(a.bed_elevation_m, b.bed_elevation_m)
                    && same(a.maximum_depth_m, b.maximum_depth_m)
                    && same(a.manning_n, b.manning_n)
            })
    }

    pub(in crate::channel::flow) fn check(&self) -> Result<(), ChannelError> {
        match self {
            Self::PrismaticSurveyed { numerical, .. } => numerical.check()?,
            Self::Rectangular { cells } => {
                for (index, c) in cells.iter().enumerate() {
                    if !c.width_m.is_finite() || c.width_m <= 0.0 {
                        return Err(ChannelError::input(format!(
                            "width_m={}: supply a finite positive width",
                            c.width_m
                        ))
                        .at(index));
                    }
                    let plan_area = c.width_m * c.length_m;
                    if !plan_area.is_finite() || plan_area <= 0.0 {
                        return Err(ChannelError::input(format!(
                            "plan_area={plan_area}: supply representable cell dimensions"
                        ))
                        .at(index));
                    }
                }
            }
            Self::PrismaticTriangular {
                left_bank_slope_h_per_v: left,
                right_bank_slope_h_per_v: right,
                ..
            } => {
                if !left.is_finite()
                    || !right.is_finite()
                    || *left < 0.0
                    || *right < 0.0
                    || (*left == 0.0 && *right == 0.0)
                {
                    return Err(ChannelError::input(format!(
                        "bank slopes=({left},{right}): supply finite nonnegative slopes with at least one positive bank"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(in crate::channel::flow) fn analytic_maximum_volume(&self, index: usize) -> Option<f64> {
        if matches!(self, Self::PrismaticSurveyed { .. }) {
            return None;
        }
        let c = self.cell(index);
        match self {
            Self::PrismaticSurveyed { .. } => None,
            Self::Rectangular { cells } => {
                Some((cells[index].width_m * c.length_m) * c.maximum_depth_m)
            }
            Self::PrismaticTriangular {
                left_bank_slope_h_per_v: l,
                right_bank_slope_h_per_v: r,
                ..
            } => {
                let k = l.midpoint(*r);
                Some(((k * c.maximum_depth_m) * c.maximum_depth_m) * c.length_m)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::geometry;
    use super::*;

    #[test]
    fn triangular_identity_retains_original_section_bits() {
        let section = geometry(1.0, 1.0, 0.0);
        assert!(section.same_bits(&section.clone()));
        assert!(!section.same_bits(&geometry(0.0, 2.0, 0.0)));
        assert!(!section.same_bits(&geometry(1.0, 1.0, -0.0)));
        assert!(!geometry(0.0, 2.0, 0.0).same_bits(&geometry(-0.0, 2.0, 0.0)));
        let mut changed = section.clone();
        if let ChannelGeometry::PrismaticTriangular { cells, .. } = &mut changed {
            cells[0].maximum_depth_m = f64::from_bits(3.0_f64.to_bits() + 1);
        }
        assert!(!section.same_bits(&changed));
    }

    #[test]
    fn triangular_banks_require_a_finite_positive_section() {
        assert!(geometry(0.0, 2.0, 0.0).check().is_ok());
        for (left, right) in [
            (0.0, 0.0),
            (-1.0, 2.0),
            (f64::INFINITY, 1.0),
            (1.0, f64::NAN),
        ] {
            assert!(geometry(left, right, 0.0).check().is_err());
        }
    }
}
