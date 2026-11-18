//! Plan areas of a stage-volume curve.
//!
//! Stage is piecewise linear in volume. Segment `k` has area `A_k = (V_{k+1} - V_k) /
//! (H_{k+1} - H_k)`. Over a stage range, the smallest segment area gives the largest stage change
//! per cubic metre, so it bounds stage change for any curve shape.

use crate::storage::LevelPoolCurve;

/// Relative shrink of each segment area, in units of `f64::EPSILON`, so that a rounded quotient
/// never exceeds the exact area.
const AREA_MARGIN_ULPS: f64 = 4.0;

/// Segment areas and knot stages of one pool curve.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PoolPlan {
    /// Knot stages in increasing order, in metres.
    stages: Vec<f64>,
    /// Knot volumes in increasing order, in m³.
    volumes: Vec<f64>,
    /// Area of each segment, rounded down, in m².
    areas: Vec<f64>,
}

impl PoolPlan {
    /// Plan of `curve`.
    pub fn new(curve: &LevelPoolCurve) -> Self {
        let knots = curve.knots();
        let areas = knots
            .windows(2)
            .map(|pair| {
                let area =
                    (pair[1].volume_m3 - pair[0].volume_m3) / (pair[1].stage_m - pair[0].stage_m);
                area * (1.0 - AREA_MARGIN_ULPS * f64::EPSILON)
            })
            .collect();
        Self {
            stages: knots.iter().map(|k| k.stage_m).collect(),
            volumes: knots.iter().map(|k| k.volume_m3).collect(),
            areas,
        }
    }

    /// Smallest segment area over the stage range `[low, high]`, in m².
    ///
    /// A range below the first knot, a dry pool included, takes the first segment. A range above
    /// the last knot takes the last segment.
    pub fn smallest(&self, low: f64, high: f64) -> f64 {
        let (low, high) = (low.min(high), low.max(high));
        let last = self.areas.len().saturating_sub(1);
        let segment = |stage: f64| {
            self.stages
                .partition_point(|&knot| knot <= stage)
                .saturating_sub(1)
                .min(last)
        };
        let (first, end) = (segment(low), segment(high));
        self.areas
            .get(first..=end)
            .and_then(|areas| areas.iter().copied().reduce(f64::min))
            .unwrap_or(0.0)
    }

    /// Stage at `volume`, clamped to the curve's closed volume range, in metres.
    pub fn stage(&self, volume: f64) -> f64 {
        let last = self.volumes.len().saturating_sub(1);
        let volume = volume.clamp(0.0, self.volumes.get(last).copied().unwrap_or(0.0));
        let right = self
            .volumes
            .partition_point(|&knot| knot < volume)
            .clamp(1, last.max(1));
        let (v0, v1) = (self.volumes[right - 1], self.volumes[right]);
        let (h0, h1) = (self.stages[right - 1], self.stages[right]);
        h0 + (volume - v0) * (h1 - h0) / (v1 - v0)
    }

    /// Secant area of the segment that holds `volume`, its right side included, in m².
    ///
    /// A volume at an interior knot takes the segment above it. Volumes outside the curve take
    /// the end segments.
    pub fn secant_area(&self, volume: f64) -> f64 {
        let last = self.volumes.len().saturating_sub(2);
        let segment = self
            .volumes
            .partition_point(|&knot| knot <= volume)
            .saturating_sub(1)
            .min(last);
        match (
            self.volumes.get(segment..segment + 2),
            self.stages.get(segment..segment + 2),
        ) {
            (Some(&[v0, v1]), Some(&[h0, h1])) => (v1 - v0) / (h1 - h0),
            _ => 0.0,
        }
    }

    /// Stage of the last knot, in metres.
    pub fn top(&self) -> f64 {
        self.stages.last().copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LevelPoolKnot;

    fn plan() -> Result<PoolPlan, Box<dyn std::error::Error>> {
        let knot = |volume_m3, stage_m| LevelPoolKnot { volume_m3, stage_m };
        let curve = LevelPoolCurve::new(
            vec![
                knot(0.0, 1.0),
                knot(8.0, 2.0),
                knot(12.0, 3.0),
                knot(32.0, 4.0),
            ],
            "d".into(),
            "Synthetic curve".into(),
        )?;
        Ok(PoolPlan::new(&curve))
    }

    #[test]
    fn the_smallest_area_covers_every_segment_of_the_range()
    -> Result<(), Box<dyn std::error::Error>> {
        let plan = plan()?;
        let shrink = 1.0 - AREA_MARGIN_ULPS * f64::EPSILON;
        assert_eq!(plan.smallest(1.2, 1.8), 8.0 * shrink);
        assert_eq!(plan.smallest(1.5, 3.5), 4.0 * shrink);
        assert_eq!(plan.smallest(3.5, 3.2), 20.0 * shrink);
        // Below the origin and above the top take the end segments.
        assert_eq!(plan.smallest(0.0, 0.5), 8.0 * shrink);
        assert_eq!(plan.smallest(5.0, 6.0), 20.0 * shrink);
        Ok(())
    }

    #[test]
    fn stage_interpolates_and_clamps() -> Result<(), Box<dyn std::error::Error>> {
        let plan = plan()?;
        assert_eq!(plan.stage(0.0), 1.0);
        assert_eq!(plan.stage(4.0), 1.5);
        assert_eq!(plan.stage(10.0), 2.5);
        assert_eq!(plan.stage(100.0), 4.0);
        assert_eq!(plan.top(), 4.0);
        // Knots (0, 1), (8, 2), (12, 3), (32, 4): segment areas 8, 4, 20.
        assert_eq!(plan.secant_area(4.0), 8.0);
        assert_eq!(plan.secant_area(8.0), 4.0);
        assert_eq!(plan.secant_area(32.0), 20.0);
        assert_eq!(plan.secant_area(100.0), 20.0);
        Ok(())
    }
}
