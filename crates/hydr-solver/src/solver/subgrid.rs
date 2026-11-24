//! Subgrid terrain storage-elevation and wet-fraction tables for coarse 2D grids.
//!
//! Builds exact piecewise-linear/piecewise-quadratic stage-volume curves from fine-resolution
//! DEM sub-cells inside each coarse grid cell so coarse-grid runs preserve fine-scale channel
//! and depression storage capacity and accurate water-surface elevation inversion.

use crate::contract::EngineError;
use hydr_core::grid::CartesianGrid;

/// Per-cell monotone subgrid storage-elevation curve built from sorted fine DEM sub-cell beds.
#[derive(Clone, Debug, PartialEq)]
pub struct SubgridCellCurve {
    /// Sorted fine sub-cell bed elevations (`m`, non-decreasing).
    elevations_m: Vec<f64>,
    /// Cumulative storage volume (`m³`) at each sorted elevation knot.
    volumes_m3: Vec<f64>,
    /// Coarse cell plan area (`m²`).
    cell_area_m2: f64,
}

impl SubgridCellCurve {
    /// Constructs a subgrid curve from `subcell_bed_m` inside a coarse cell of `cell_area_m2`.
    pub fn from_subcell_dem(subcell_bed_m: &[f64], cell_area_m2: f64) -> Result<Self, EngineError> {
        if subcell_bed_m.is_empty() {
            return Err(EngineError::rejected(
                "subgrid cell curve",
                "subcell_bed_m must not be empty",
            ));
        }
        if !cell_area_m2.is_finite() || cell_area_m2 <= 0.0 {
            return Err(EngineError::rejected(
                "subgrid cell curve",
                format!("cell_area_m2 must be finite and > 0, got {cell_area_m2}"),
            ));
        }
        if subcell_bed_m.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::rejected(
                "subgrid cell curve",
                "subcell_bed_m contains non-finite elevation",
            ));
        }
        let mut elevations_m = subcell_bed_m.to_vec();
        elevations_m.sort_by(f64::total_cmp);
        let sub_area = cell_area_m2 / (elevations_m.len() as f64);
        let mut volumes_m3 = Vec::with_capacity(elevations_m.len());
        let mut cum = 0.0_f64;
        volumes_m3.push(0.0);
        for k in 1..elevations_m.len() {
            let dz = elevations_m[k] - elevations_m[k - 1];
            cum += (k as f64) * sub_area * dz;
            volumes_m3.push(cum);
        }
        Ok(Self {
            elevations_m,
            volumes_m3,
            cell_area_m2,
        })
    }

    /// Minimum (thalweg) bed elevation inside the coarse cell (`m`).
    pub fn min_bed_m(&self) -> f64 {
        self.elevations_m[0]
    }

    /// Maximum (bank/ridge) bed elevation inside the coarse cell (`m`).
    pub fn max_bed_m(&self) -> f64 {
        self.elevations_m[self.elevations_m.len() - 1]
    }

    /// Coarse cell plan area (`m²`).
    pub fn cell_area_m2(&self) -> f64 {
        self.cell_area_m2
    }

    /// Exact stored volume (`m³`) at water-surface elevation `stage_m`.
    pub fn volume_at_stage(&self, stage_m: f64) -> f64 {
        let n = self.elevations_m.len();
        if stage_m <= self.elevations_m[0] {
            return 0.0;
        }
        let idx = self.elevations_m.partition_point(|&z| z <= stage_m);
        let sub_area = self.cell_area_m2 / (n as f64);
        if idx >= n {
            self.volumes_m3[n - 1] + self.cell_area_m2 * (stage_m - self.elevations_m[n - 1])
        } else {
            self.volumes_m3[idx - 1]
                + (idx as f64) * sub_area * (stage_m - self.elevations_m[idx - 1])
        }
    }

    /// Exact water-surface elevation (`m`) for stored volume `volume_m3` (`>= 0`).
    pub fn stage_at_volume(&self, volume_m3: f64) -> f64 {
        let n = self.elevations_m.len();
        if volume_m3 <= 0.0 {
            return self.elevations_m[0];
        }
        let v_top = self.volumes_m3[n - 1];
        if volume_m3 >= v_top {
            return self.elevations_m[n - 1] + (volume_m3 - v_top) / self.cell_area_m2;
        }
        let idx = self.volumes_m3.partition_point(|&v| v <= volume_m3);
        let k = idx.saturating_sub(1);
        let sub_area = self.cell_area_m2 / (n as f64);
        let wetted_area = ((k + 1) as f64) * sub_area;
        self.elevations_m[k] + (volume_m3 - self.volumes_m3[k]) / wetted_area
    }

    /// Fraction of sub-cells submerged at `stage_m`, in `[0.0, 1.0]`.
    pub fn wet_fraction_at_stage(&self, stage_m: f64) -> f64 {
        if stage_m <= self.elevations_m[0] {
            return 0.0;
        }
        let wet = self.elevations_m.partition_point(|&z| z < stage_m);
        (wet as f64) / (self.elevations_m.len() as f64)
    }

    /// Effective coarse depth `volume_m3 / cell_area_m2` (`m`) at `stage_m`.
    pub fn effective_depth_at_stage(&self, stage_m: f64) -> f64 {
        self.volume_at_stage(stage_m) / self.cell_area_m2
    }
}

/// Grid-wide subgrid terrain storage and wet-fraction lookup tables.
#[derive(Clone, Debug, PartialEq)]
pub struct SubgridTerrain {
    grid: CartesianGrid,
    curves: Vec<SubgridCellCurve>,
}

impl SubgridTerrain {
    /// Builds subgrid tables over `coarse_grid` from a fine DEM of dimensions `(fine_nx, fine_ny)`
    /// where `fine_nx` is an exact positive multiple of `coarse_nx` and `fine_ny` of `coarse_ny`.
    pub fn from_fine_dem(
        coarse_grid: CartesianGrid,
        fine_nx: usize,
        fine_ny: usize,
        fine_bed_m: &[f64],
    ) -> Result<Self, EngineError> {
        let (cx, cy) = coarse_grid.dimensions();
        if fine_nx == 0
            || fine_ny == 0
            || !fine_nx.is_multiple_of(cx)
            || !fine_ny.is_multiple_of(cy)
        {
            return Err(EngineError::rejected(
                "subgrid terrain",
                format!(
                    "fine dimensions ({fine_nx}x{fine_ny}) must be positive multiples of coarse grid ({cx}x{cy})"
                ),
            ));
        }
        if fine_bed_m.len() != fine_nx * fine_ny {
            return Err(EngineError::rejected(
                "subgrid terrain",
                format!(
                    "fine_bed_m length ({}) != fine_nx * fine_ny ({})",
                    fine_bed_m.len(),
                    fine_nx * fine_ny
                ),
            ));
        }
        let rx = fine_nx / cx;
        let ry = fine_ny / cy;
        let area = coarse_grid.cell_area();
        let mut curves = Vec::with_capacity(cx * cy);
        let mut scratch = Vec::with_capacity(rx * ry);
        for row in 0..cy {
            for col in 0..cx {
                scratch.clear();
                for sy in 0..ry {
                    let fy = row * ry + sy;
                    for sx in 0..rx {
                        let fx = col * rx + sx;
                        scratch.push(fine_bed_m[fy * fine_nx + fx]);
                    }
                }
                curves.push(SubgridCellCurve::from_subcell_dem(&scratch, area)?);
            }
        }
        Ok(Self {
            grid: coarse_grid,
            curves,
        })
    }

    /// Coarse grid geometry.
    pub fn grid(&self) -> CartesianGrid {
        self.grid
    }

    /// Per-cell subgrid curves in row-major order.
    pub fn curves(&self) -> &[SubgridCellCurve] {
        &self.curves
    }

    /// Converts coarse effective depths (`m`) to exact subgrid water-surface elevations (`m`).
    pub fn stage_field(&self, effective_depths_m: &[f64]) -> Vec<f64> {
        let area = self.grid.cell_area();
        self.curves
            .iter()
            .zip(effective_depths_m)
            .map(|(c, &h)| c.stage_at_volume((h * area).max(0.0)))
            .collect()
    }

    /// Converts coarse effective depths (`m`) to exact subgrid wet fractions in `[0.0, 1.0]`.
    pub fn wet_fraction_field(&self, effective_depths_m: &[f64]) -> Vec<f64> {
        let area = self.grid.cell_area();
        self.curves
            .iter()
            .zip(effective_depths_m)
            .map(|(c, &h)| {
                let stage = c.stage_at_volume((h * area).max(0.0));
                c.wet_fraction_at_stage(stage)
            })
            .collect()
    }

    /// Converts per-cell water-surface elevations (`m`) to exact coarse effective depths (`m`).
    pub fn effective_depth_field(&self, stages_m: &[f64]) -> Vec<f64> {
        self.curves
            .iter()
            .zip(stages_m)
            .map(|(c, &eta)| c.effective_depth_at_stage(eta))
            .collect()
    }
}
