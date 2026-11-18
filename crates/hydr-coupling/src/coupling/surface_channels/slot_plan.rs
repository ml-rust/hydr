//! Plan geometry of linked channel cells.
//!
//! Width never shrinks as stage rises, so the width at the lowest stage a bound covers gives the
//! largest stage change per cubic metre.

use super::SurfaceChannelError;
use crate::channel::{ChannelGeometry, ChannelReachDefinition};

/// Cross-section of one linked channel cell.
#[derive(Clone, Debug)]
pub(super) enum SlotSection {
    Rectangular {
        width_m: f64,
    },
    Triangular {
        slope_sum: f64,
        bed_m: f64,
    },
    /// Ordered `(station, elevation)` profile points, in metres.
    Surveyed {
        points: Vec<(f64, f64)>,
    },
}

/// Plan geometry of one linked channel cell.
#[derive(Clone, Debug)]
pub(super) struct SlotPlan {
    pub length_m: f64,
    pub section: SlotSection,
}

impl SlotPlan {
    /// Plan geometry of cell `cell` of `definition`.
    pub fn new(
        definition: &ChannelReachDefinition,
        cell: usize,
    ) -> Result<Self, SurfaceChannelError> {
        let missing = || {
            SurfaceChannelError::input(format!("reach cell={cell}: link an existing reach cell"))
        };
        Ok(match &definition.geometry {
            ChannelGeometry::Rectangular { cells } => {
                let cell = cells.get(cell).ok_or_else(missing)?;
                Self {
                    length_m: cell.length_m,
                    section: SlotSection::Rectangular {
                        width_m: cell.width_m,
                    },
                }
            }
            ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v,
                right_bank_slope_h_per_v,
                bed_elevation_m,
                cells,
            } => Self {
                length_m: cells.get(cell).ok_or_else(missing)?.length_m,
                section: SlotSection::Triangular {
                    slope_sum: left_bank_slope_h_per_v + right_bank_slope_h_per_v,
                    bed_m: *bed_elevation_m,
                },
            },
            ChannelGeometry::PrismaticSurveyed { profile, cells, .. } => Self {
                length_m: cells.get(cell).ok_or_else(missing)?.length_m,
                section: SlotSection::Surveyed {
                    points: profile
                        .points
                        .iter()
                        .map(|p| (p.station_m, p.elevation_m))
                        .collect(),
                },
            },
        })
    }

    /// Water-surface width at `stage`, in metres. A bench at `stage` adds no width.
    pub fn width(&self, stage: f64) -> f64 {
        match &self.section {
            SlotSection::Rectangular { width_m } => *width_m,
            SlotSection::Triangular { slope_sum, bed_m } => slope_sum * (stage - bed_m).max(0.0),
            SlotSection::Surveyed { points } => points
                .windows(2)
                .map(|pair| {
                    let &[(x0, z0), (x1, z1)] = pair else {
                        return 0.0;
                    };
                    let (low, high) = (z0.min(z1), z0.max(z1));
                    let run = (x1 - x0).abs();
                    if high < stage {
                        run
                    } else if low < stage {
                        run * (stage - low) / (high - low)
                    } else {
                        0.0
                    }
                })
                .sum(),
        }
    }

    /// Plan area at `stage`, in m².
    pub fn area(&self, stage: f64) -> f64 {
        self.length_m * self.width(stage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surveyed_width_grows_with_stage() {
        let plan = SlotPlan {
            length_m: 2.0,
            section: SlotSection::Surveyed {
                points: vec![(0.0, 4.0), (1.0, 0.0), (3.0, 0.0), (5.0, 4.0)],
            },
        };
        assert_eq!(plan.width(0.0), 0.0);
        assert_eq!(plan.width(2.0), 3.5);
        assert_eq!(plan.width(4.0), 5.0);
        assert_eq!(plan.area(2.0), 7.0);
    }

    #[test]
    fn prismatic_widths_match_their_sections() {
        let rectangle = SlotPlan {
            length_m: 2.0,
            section: SlotSection::Rectangular { width_m: 1.5 },
        };
        assert_eq!(rectangle.area(-3.0), 3.0);
        let triangle = SlotPlan {
            length_m: 2.0,
            section: SlotSection::Triangular {
                slope_sum: 3.0,
                bed_m: 1.0,
            },
        };
        assert_eq!(triangle.width(0.5), 0.0);
        assert_eq!(triangle.width(2.0), 3.0);
    }
}
