//! Subgrid terrain storage-elevation and wet-fraction tables for coarse 2D grids.
//!
//! Builds exact piecewise-linear/piecewise-quadratic stage-volume curves from fine-resolution
//! DEM sub-cells inside each coarse grid cell so coarse-grid runs preserve fine-scale channel
//! and depression storage capacity and accurate water-surface elevation inversion.

use crate::contract::{Axis, EngineError};
use super::tensor::FaceSupport;
use hydr_core::grid::CartesianGrid;

/// Per-cell monotone subgrid storage-elevation curve built from sorted fine DEM sub-cell beds,
/// with an optional sub-pixel incised river channel `(channel_bed_m, channel_area_m2)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SubgridCellCurve {
    /// Sorted fine sub-cell bed elevations (`m`, non-decreasing).
    elevations_m: Vec<f64>,
    /// Cumulative storage volume (`m³`) at each sorted elevation knot (above `elevations_m[0]`).
    volumes_m3: Vec<f64>,
    /// Coarse cell plan area (`m²`).
    cell_area_m2: f64,
    /// Optional sub-pixel channel `(channel_bed_m, channel_area_m2)` below `elevations_m[0]`.
    channel: Option<(f64, f64)>,
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
            channel: None,
        })
    }

    /// Incises a sub-pixel river channel of `depth_m` below `elevations_m[0]` with plan area `channel_area_m2`.
    pub fn with_channel(
        mut self,
        depth_m: f64,
        channel_area_m2: f64,
    ) -> Result<Self, EngineError> {
        if !depth_m.is_finite() || depth_m <= 0.0 {
            return Err(EngineError::rejected(
                "subgrid channel",
                format!("channel depth_m must be finite and > 0, got {depth_m}"),
            ));
        }
        if !channel_area_m2.is_finite()
            || channel_area_m2 <= 0.0
            || channel_area_m2 >= self.cell_area_m2
        {
            return Err(EngineError::rejected(
                "subgrid channel",
                format!(
                    "channel_area_m2 ({channel_area_m2}) must be in (0, cell_area_m2={})",
                    self.cell_area_m2
                ),
            ));
        }
        let ch_bed = self.elevations_m[0] - depth_m;
        self.channel = Some((ch_bed, channel_area_m2));
        Ok(self)
    }

    /// Minimum (thalweg) bed elevation inside the coarse cell (`m`).
    pub fn min_bed_m(&self) -> f64 {
        self.channel
            .map_or(self.elevations_m[0], |(ch_bed, _)| ch_bed)
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
        let z0 = self.elevations_m[0];
        let v_ch = match self.channel {
            Some((ch_bed, ch_area)) => {
                if stage_m <= ch_bed {
                    return 0.0;
                }
                if stage_m <= z0 {
                    return ch_area * (stage_m - ch_bed);
                }
                ch_area * (z0 - ch_bed)
            }
            None => {
                if stage_m <= z0 {
                    return 0.0;
                }
                0.0
            }
        };
        let n = self.elevations_m.len();
        let idx = self.elevations_m.partition_point(|&z| z <= stage_m);
        let sub_area = self.cell_area_m2 / (n as f64);
        let v_dem = if idx >= n {
            self.volumes_m3[n - 1] + self.cell_area_m2 * (stage_m - self.elevations_m[n - 1])
        } else {
            self.volumes_m3[idx - 1]
                + (idx as f64) * sub_area * (stage_m - self.elevations_m[idx - 1])
        };
        v_ch + v_dem
    }

    /// Exact water-surface elevation (`m`) for stored volume `volume_m3` (`>= 0`).
    pub fn stage_at_volume(&self, volume_m3: f64) -> f64 {
        let z0 = self.elevations_m[0];
        let rem_vol = match self.channel {
            Some((ch_bed, ch_area)) => {
                if volume_m3 <= 0.0 {
                    return ch_bed;
                }
                let v_bankfull = ch_area * (z0 - ch_bed);
                if volume_m3 <= v_bankfull {
                    return ch_bed + volume_m3 / ch_area;
                }
                volume_m3 - v_bankfull
            }
            None => {
                if volume_m3 <= 0.0 {
                    return z0;
                }
                volume_m3
            }
        };
        let n = self.elevations_m.len();
        let v_top = self.volumes_m3[n - 1];
        if rem_vol >= v_top {
            return self.elevations_m[n - 1] + (rem_vol - v_top) / self.cell_area_m2;
        }
        let idx = self.volumes_m3.partition_point(|&v| v <= rem_vol);
        let k = idx.saturating_sub(1);
        let sub_area = self.cell_area_m2 / (n as f64);
        let wetted_area = ((k + 1) as f64) * sub_area;
        self.elevations_m[k] + (rem_vol - self.volumes_m3[k]) / wetted_area
    }

    /// Fraction of cell area submerged at `stage_m`, in `[0.0, 1.0]`.
    pub fn wet_fraction_at_stage(&self, stage_m: f64) -> f64 {
        let ch_frac = match self.channel {
            Some((ch_bed, ch_area)) if stage_m > ch_bed => ch_area / self.cell_area_m2,
            _ => 0.0,
        };
        let dem_frac = if stage_m <= self.elevations_m[0] {
            0.0
        } else {
            let wet = self.elevations_m.partition_point(|&z| z < stage_m);
            (wet as f64) / (self.elevations_m.len() as f64)
        };
        ch_frac.max(dem_frac)
    }

    /// Effective coarse depth `volume_m3 / cell_area_m2` (`m`) at `stage_m`.
    pub fn effective_depth_at_stage(&self, stage_m: f64) -> f64 {
        self.volume_at_stage(stage_m) / self.cell_area_m2
    }
}

/// Per-face subgrid sill elevation profile along an interior coarse cell boundary,
/// with an optional sub-pixel river channel sill `(sill_m, width_fraction)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SubgridFaceCurve {
    /// Sorted fine sub-face sill elevations (`m`, non-decreasing).
    sills_m: Vec<f64>,
    /// Optional sub-pixel river channel `(sill_m, width_fraction)` in `(0.0, 1.0)`.
    channel_sill: Option<(f64, f64)>,
}

impl SubgridFaceCurve {
    /// Constructs a subgrid face curve from fine sub-face sill elevations (`m`).
    pub fn from_subface_sills(sills_m: &[f64]) -> Result<Self, EngineError> {
        if sills_m.is_empty() {
            return Err(EngineError::rejected(
                "subgrid face curve",
                "sills_m must not be empty",
            ));
        }
        if sills_m.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::rejected(
                "subgrid face curve",
                "sills_m contains non-finite elevation",
            ));
        }
        let mut sorted = sills_m.to_vec();
        sorted.sort_by(f64::total_cmp);
        Ok(Self {
            sills_m: sorted,
            channel_sill: None,
        })
    }

    /// Attaches a sub-pixel channel sill `(sill_m, width_fraction)` across this face.
    pub fn with_channel_sill(mut self, sill_m: f64, width_fraction: f64) -> Self {
        if sill_m.is_finite() && width_fraction.is_finite() && width_fraction > 0.0 {
            self.channel_sill = Some((sill_m, width_fraction.clamp(0.0, 0.99)));
        }
        self
    }

    /// Lowest sill (thalweg/culvert) elevation crossing the face (`m`).
    pub fn min_sill_m(&self) -> f64 {
        let base = self.sills_m[0];
        self.channel_sill
            .map_or(base, |(sill, _)| base.min(sill))
    }

    /// Highest barrier (bank/levee) elevation along the face (`m`).
    pub fn max_sill_m(&self) -> f64 {
        self.sills_m[self.sills_m.len() - 1]
    }

    /// Fraction of face width submerged at water-surface elevation `stage_m`, in `[0.0, 1.0]`.
    pub fn wet_width_fraction(&self, stage_m: f64) -> f64 {
        let ch_w = match self.channel_sill {
            Some((sill, wf)) if stage_m > sill => wf,
            _ => 0.0,
        };
        let dem_w = if stage_m <= self.sills_m[0] {
            0.0
        } else {
            let wet = self.sills_m.partition_point(|&z| z < stage_m);
            (wet as f64) / (self.sills_m.len() as f64)
        };
        ch_w.max(dem_w)
    }

    /// Mean subgrid flow depth across the face at `stage_m` (`m`).
    pub fn effective_depth_at_stage(&self, stage_m: f64) -> f64 {
        let dem_d = if stage_m <= self.sills_m[0] {
            0.0
        } else {
            let sum: f64 = self
                .sills_m
                .iter()
                .take_while(|&&z| z < stage_m)
                .map(|&z| stage_m - z)
                .sum();
            sum / (self.sills_m.len() as f64)
        };
        match self.channel_sill {
            Some((sill, wf)) => {
                let ch_d = (stage_m - sill).max(0.0);
                wf * ch_d + (1.0 - wf) * dem_d
            }
            None => dem_d,
        }
    }
}

/// Grid-wide subgrid terrain storage, wet-fraction, and face-conveyance lookup tables.
#[derive(Clone, Debug, PartialEq)]
pub struct SubgridTerrain {
    grid: CartesianGrid,
    curves: Vec<SubgridCellCurve>,
    /// Interior X-faces between `(col, row)` and `(col + 1, row)`, length `(cx - 1) * cy`.
    x_faces: Vec<SubgridFaceCurve>,
    /// Interior Y-faces between `(col, row)` and `(col, row + 1)`, length `cx * (cy - 1)`.
    y_faces: Vec<SubgridFaceCurve>,
}

impl SubgridTerrain {
    /// Builds subgrid cell and face tables over `coarse_grid` from a fine DEM of dimensions `(fine_nx, fine_ny)`
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
        let mut x_faces = Vec::with_capacity(cx.saturating_sub(1) * cy);
        let mut face_scratch = Vec::with_capacity(rx.max(ry));
        for row in 0..cy {
            for col in 0..cx.saturating_sub(1) {
                face_scratch.clear();
                let fx_left = (col + 1) * rx - 1;
                let fx_right = (col + 1) * rx;
                for sy in 0..ry {
                    let fy = row * ry + sy;
                    let zl = fine_bed_m[fy * fine_nx + fx_left];
                    let zr = fine_bed_m[fy * fine_nx + fx_right];
                    face_scratch.push(zl.max(zr));
                }
                x_faces.push(SubgridFaceCurve::from_subface_sills(&face_scratch)?);
            }
        }
        let mut y_faces = Vec::with_capacity(cx * cy.saturating_sub(1));
        for row in 0..cy.saturating_sub(1) {
            for col in 0..cx {
                face_scratch.clear();
                let fy_bot = (row + 1) * ry - 1;
                let fy_top = (row + 1) * ry;
                for sx in 0..rx {
                    let fx = col * rx + sx;
                    let zb = fine_bed_m[fy_bot * fine_nx + fx];
                    let zt = fine_bed_m[fy_top * fine_nx + fx];
                    face_scratch.push(zb.max(zt));
                }
                y_faces.push(SubgridFaceCurve::from_subface_sills(&face_scratch)?);
            }
        }
        Ok(Self {
            grid: coarse_grid,
            curves,
            x_faces,
            y_faces,
        })
    }

    /// Incises opt-in sub-pixel river channels `(width_m, depth_m)` into matching cells and
    /// their shared interior faces so narrow (`10–100 m`) rivers route continuously below coarse DEM banks.
    pub fn with_channels(
        mut self,
        cell_channels: &[Option<(f64, f64)>],
    ) -> Result<Self, EngineError> {
        let (cx, cy) = self.grid.dimensions();
        let n = cx * cy;
        if cell_channels.len() != n {
            return Err(EngineError::rejected(
                "subgrid channels",
                format!(
                    "cell_channels length ({}) != coarse cells ({n})",
                    cell_channels.len()
                ),
            ));
        }
        let (dx, dy) = self.grid.spacing();
        let cell_len = (dx * dy).sqrt();
        let cell_area = self.grid.cell_area();
        for (i, ch) in cell_channels.iter().enumerate() {
            if let Some((width_m, depth_m)) = *ch {
                if !width_m.is_finite() || width_m <= 0.0 || !depth_m.is_finite() || depth_m <= 0.0 {
                    return Err(EngineError::rejected(
                        "subgrid channels",
                        format!("cell {i}: width_m={width_m} and depth_m={depth_m} must be finite > 0"),
                    ));
                }
                let ch_area = (width_m * cell_len).min(0.5 * cell_area);
                let curve = std::mem::replace(
                    &mut self.curves[i],
                    SubgridCellCurve {
                        elevations_m: Vec::new(),
                        volumes_m3: Vec::new(),
                        cell_area_m2: cell_area,
                        channel: None,
                    },
                );
                self.curves[i] = curve.with_channel(depth_m, ch_area)?;
            }
        }
        // Wire shared interior X-faces where both neighbors carry a sub-pixel channel.
        for row in 0..cy {
            for col in 0..cx.saturating_sub(1) {
                let l = row * cx + col;
                let r = l + 1;
                if let (Some((wl, _)), Some((wr, _))) = (cell_channels[l], cell_channels[r]) {
                    let sill = self.curves[l].min_bed_m().max(self.curves[r].min_bed_m());
                    let wf = (wl.min(wr) / dy).clamp(0.0, 0.5);
                    let fidx = row * (cx - 1) + col;
                    let face = std::mem::replace(
                        &mut self.x_faces[fidx],
                        SubgridFaceCurve {
                            sills_m: Vec::new(),
                            channel_sill: None,
                        },
                    );
                    self.x_faces[fidx] = face.with_channel_sill(sill, wf);
                }
            }
        }
        // Wire shared interior Y-faces where both neighbors carry a sub-pixel channel.
        for row in 0..cy.saturating_sub(1) {
            for col in 0..cx {
                let b = row * cx + col;
                let t = b + cx;
                if let (Some((wb, _)), Some((wt, _))) = (cell_channels[b], cell_channels[t]) {
                    let sill = self.curves[b].min_bed_m().max(self.curves[t].min_bed_m());
                    let wf = (wb.min(wt) / dx).clamp(0.0, 0.5);
                    let fidx = row * cx + col;
                    let face = std::mem::replace(
                        &mut self.y_faces[fidx],
                        SubgridFaceCurve {
                            sills_m: Vec::new(),
                            channel_sill: None,
                        },
                    );
                    self.y_faces[fidx] = face.with_channel_sill(sill, wf);
                }
            }
        }
        Ok(self)
    }

    /// Coarse grid geometry.
    pub fn grid(&self) -> CartesianGrid {
        self.grid
    }

    /// Per-cell subgrid curves in row-major order.
    pub fn curves(&self) -> &[SubgridCellCurve] {
        &self.curves
    }

    /// Subgrid face curve for interior X-face between `(col, row)` and `(col + 1, row)`.
    pub fn x_face(&self, col: usize, row: usize) -> Option<&SubgridFaceCurve> {
        let (cx, cy) = self.grid.dimensions();
        if col + 1 < cx && row < cy {
            self.x_faces.get(row * (cx - 1) + col)
        } else {
            None
        }
    }

    /// Subgrid face curve for interior Y-face between `(col, row)` and `(col, row + 1)`.
    pub fn y_face(&self, col: usize, row: usize) -> Option<&SubgridFaceCurve> {
        let (cx, cy) = self.grid.dimensions();
        if col < cx && row + 1 < cy {
            self.y_faces.get(row * cx + col)
        } else {
            None
        }
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

    /// Computes well-balanced effective bed `z_eff = stage(V) - h` so `z_eff + h == stage(V)` identically.
    pub fn effective_bed_field(&self, effective_depths_m: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let stages = self.stage_field(effective_depths_m);
        let eff_bed = stages
            .iter()
            .zip(effective_depths_m)
            .map(|(&eta, &h)| eta - h)
            .collect();
        (stages, eff_bed)
    }

    /// Computes per-face subgrid area-conveyance fraction in `[0.0, 1.0]` for every solver topology face.
    ///
    /// Exterior boundary faces keep `1.0`. Interior faces evaluate
    /// `clamp(effective_depth_at_stage(max(eta_L, eta_R)) / max(h_L, h_R), 0.0, 1.0)`, which:
    /// - blocks flow identically (`0.0`) when `eta_f <= min_sill`,
    /// - ramps smoothly from `0.0` as levees/sills overtop (no step discontinuities), and
    /// - avoids double-counting narrow-channel width already encoded in `h_coarse = f_wet * h_wet`.
    pub(crate) fn face_conveyance_fractions(
        &self,
        faces: &[FaceSupport],
        stages_m: &[f64],
        depths_m: &[f64],
    ) -> Vec<f64> {
        faces
            .iter()
            .map(|support| match (support.left, support.right) {
                (Some(left), Some(right)) => {
                    let h_face = depths_m[left].max(depths_m[right]);
                    if h_face <= 0.0 {
                        return 0.0;
                    }
                    let eta_f = stages_m[left].max(stages_m[right]);
                    let curve = match support.face.axis {
                        Axis::X => support
                            .face
                            .column
                            .checked_sub(1)
                            .and_then(|c| self.x_face(c, support.face.row)),
                        Axis::Y => support
                            .face
                            .row
                            .checked_sub(1)
                            .and_then(|r| self.y_face(support.face.column, r)),
                    };
                    curve.map_or(1.0, |c| {
                        (c.effective_depth_at_stage(eta_f) / h_face).clamp(0.0, 1.0)
                    })
                }
                _ => 1.0,
            })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, ForcingInterval, ForcingSchedule,
        SolverConfig,
    };
    use crate::model::{CellState, Domain, FlowState, SpatialReference};
    use crate::soil::{LandCell, LandModel, SoilState};
    use crate::solver::{InitialState, Solver, SolverSetup};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn make_solver(
        nx: usize,
        ny: usize,
        dx: f64,
        fine_nx: usize,
        fine_ny: usize,
        fine_bed: &[f64],
        init_h: &[f64],
        open_right: bool,
        end_s: f64,
    ) -> Solver<CpuRuntime> {
        let grid = CartesianGrid::new(nx, ny, dx, dx).expect("grid");
        let sg = SubgridTerrain::from_fine_dem(grid, fine_nx, fine_ny, fine_bed).expect("subgrid");
        let coarse_bed: Vec<f64> = sg.curves().iter().map(|c| c.min_bed_m()).collect();
        let manning = vec![0.03; nx * ny];
        let domain = Domain::new(
            grid,
            coarse_bed,
            manning,
            vec![true; nx * ny],
            SpatialReference::new("LOCAL".to_owned(), "MSL".to_owned(), [0.0, 0.0]).expect("crs"),
        )
        .expect("domain");
        let land = LandModel::new(&domain, vec![LandCell::Impervious; nx * ny]).expect("land");
        let soil = SoilState::initialized(&land).expect("soil");
        let open_faces = if open_right {
            (0..ny)
                .map(|row| BoundaryFace {
                    axis: Axis::X,
                    column: nx,
                    row,
                })
                .collect()
        } else {
            vec![]
        };
        let boundaries = BoundaryConditions::with_open_faces(&domain, open_faces.clone()).expect("bc");
        let boundary_forcing = open_faces
            .into_iter()
            .map(BoundaryForcing::Transmissive)
            .collect();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: end_s,
                rain: vec![0.0; nx * ny],
                potential_evaporation: vec![0.0; nx * ny],
                withdrawal: vec![0.0; nx * ny],
                boundaries: boundary_forcing,
            }],
        )
        .expect("forcing");
        let cells: Vec<CellState> = init_h
            .iter()
            .map(|&h| CellState {
                h,
                qx: 0.0,
                qy: 0.0,
            })
            .collect();
        let flow = FlowState::new(&domain, cells).expect("flow");
        let setup = SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config: SolverConfig::default(),
        };
        let initial = InitialState {
            time: 0.0,
            flow,
            soil,
        };
        let client = CpuClient::new(CpuDevice::new());
        Solver::<CpuRuntime>::new(client, setup, initial)
            .expect("solver")
            .with_subgrid_terrain(sg)
            .expect("attach subgrid")
    }

    #[test]
    fn subgrid_lake_at_rest_is_well_balanced() {
        // 2x1 coarse cells, 4x4 fine subcells per coarse cell (8x4 fine DEM).
        // Cell 0 has a deep channel (0m..4m); Cell 1 has a shallow shelf (1m..4m).
        let mut fine = vec![4.0; 8 * 4];
        for r in 0..4 {
            for c in 0..4 {
                fine[r * 8 + c] = if r < 2 { 0.0 } else { 4.0 };
            }
            for c in 4..8 {
                fine[r * 8 + c] = if r < 2 { 1.0 } else { 4.0 };
            }
        }
        let grid = CartesianGrid::new(2, 1, 100.0, 100.0).expect("grid");
        let sg = SubgridTerrain::from_fine_dem(grid, 8, 4, &fine).expect("sg");
        // Both cells initialized at identical WSE = 2.0m -> different effective depths (1.0m vs 0.5m)!
        let init_h = sg.effective_depth_field(&[2.0, 2.0]);
        assert!((init_h[0] - 1.0).abs() < 1e-12);
        assert!((init_h[1] - 0.5).abs() < 1e-12);

        let mut solver = make_solver(2, 1, 100.0, 8, 4, &fine, &init_h, false, 60.0);
        while solver.time() < 60.0 {
            solver.advance_to(60.0, 10_000).expect("step");
        }
        let st = solver.state().expect("state");
        let wse = solver.water_surface_elevation().expect("wse");
        for (i, cell) in st.cells().iter().enumerate() {
            assert!(cell.qx.abs() < 1e-12, "cell {i} qx={}", cell.qx);
            assert!(cell.qy.abs() < 1e-12, "cell {i} qy={}", cell.qy);
            assert!((wse[i] - 2.0).abs() < 1e-12, "cell {i} wse={}", wse[i]);
        }
    }

    #[test]
    fn subgrid_narrow_channel_drains_below_coarse_average_bed() {
        // 4x1 coarse cells (dx=100m), 4x4 fine subcells per coarse cell (16x4 fine DEM).
        // A 1-subcell-wide channel (25% width) at z=0m runs through z=5m banks (avg bed = 3.75m).
        let mut fine = vec![5.0; 16 * 4];
        for c in 0..16 {
            fine[1 * 16 + c] = 0.0; // narrow channel along row 1
        }
        let grid = CartesianGrid::new(4, 1, 100.0, 100.0).expect("grid");
        let sg = SubgridTerrain::from_fine_dem(grid, 16, 4, &fine).expect("sg");
        // Put water ONLY in cell 0 at WSE=1.5m (far below coarse average bed 3.75m!).
        let init_h = sg.effective_depth_field(&[1.5, 0.0, 0.0, 0.0]);
        let mut solver = make_solver(4, 1, 100.0, 16, 4, &fine, &init_h, false, 300.0);
        while solver.time() < 300.0 {
            solver.advance_to(300.0, 10_000).expect("step");
        }
        let st = solver.state().expect("state");
        let wf = solver.wet_fraction().expect("wf");
        assert!(
            st.cells()[1].h > 0.01 && st.cells()[2].h > 0.005,
            "water must propagate through narrow subgrid channel: {:?}",
            st.cells().iter().map(|c| c.h).collect::<Vec<_>>()
        );
        assert!(
            wf[1] <= 0.30,
            "in-bank channel flow must keep wet_fraction <= 0.30, got {}",
            wf[1]
        );
    }

    #[test]
    fn subgrid_levee_blocks_below_crest_and_overtops_above() {
        // 2x1 coarse cells, 4x4 fine subcells per cell (8x4 fine DEM).
        // Flat valley z=0m, with a fine-scale levee z=3.0m along the shared face (col 3 & col 4).
        let mut fine = vec![0.0; 8 * 4];
        for r in 0..4 {
            fine[r * 8 + 3] = 3.0;
        }
        let grid = CartesianGrid::new(2, 1, 100.0, 100.0).expect("grid");
        let sg = SubgridTerrain::from_fine_dem(grid, 8, 4, &fine).expect("sg");

        // Case 1: WSE = 2.0m < 3.0m levee crest -> ZERO water crosses to cell 1.
        let h_below = sg.effective_depth_field(&[2.0, 0.0]);
        let mut s_below = make_solver(2, 1, 100.0, 8, 4, &fine, &h_below, false, 60.0);
        while s_below.time() < 60.0 {
            s_below.advance_to(60.0, 10_000).expect("step");
        }
        assert_eq!(
            s_below.state().expect("st").cells()[1].h,
            0.0,
            "levee must block flow below crest"
        );

        // Case 2: WSE = 3.5m > 3.0m levee crest -> overtops into cell 1 with exact mass conservation.
        let h_above = sg.effective_depth_field(&[3.5, 0.0]);
        let init_vol: f64 = h_above.iter().sum::<f64>() * 100.0 * 100.0;
        let mut s_above = make_solver(2, 1, 100.0, 8, 4, &fine, &h_above, false, 60.0);
        while s_above.time() < 60.0 {
            s_above.advance_to(60.0, 10_000).expect("step");
        }
        let cells = s_above.state().expect("st").cells().to_vec();
        let end_vol: f64 = cells.iter().map(|c| c.h).sum::<f64>() * 100.0 * 100.0;
        assert!(cells[1].h > 0.05, "water must overtop levee above crest");
        assert!(
            (end_vol - init_vol).abs() < 1e-6,
            "mass conservation violated: init={init_vol}, end={end_vol}"
        );
    }

    #[test]
    fn subgrid_checkpoint_resume_matches_uninterrupted() {
        // 3x1 coarse cells, 4x4 fine subcells (12x4 fine DEM) with sloped subgrid channel.
        let mut fine = vec![5.0; 12 * 4];
        for c in 0..12 {
            fine[1 * 12 + c] = 2.0 - 0.1 * (c as f64);
        }
        let grid = CartesianGrid::new(3, 1, 100.0, 100.0).expect("grid");
        let sg = SubgridTerrain::from_fine_dem(grid, 12, 4, &fine).expect("sg");
        let init_h = sg.effective_depth_field(&[3.0, 0.0, 0.0]);

        // Uninterrupted 120s run
        let mut full = make_solver(3, 1, 100.0, 12, 4, &fine, &init_h, false, 120.0);
        while full.time() < 120.0 {
            full.advance_to(120.0, 10_000).expect("full step");
        }
        let full_cells = full.state().expect("full state").cells().to_vec();

        // Split 60s -> checkpoint -> resume 60s..120s
        let mut part1 = make_solver(3, 1, 100.0, 12, 4, &fine, &init_h, false, 120.0);
        while part1.time() < 60.0 {
            part1.advance_to(60.0, 10_000).expect("part1 step");
        }
        let snap = part1.snapshot().data().expect("checkpoint");
        let forcing_resume = part1.definition.forcing.clone();
        let client = CpuClient::new(CpuDevice::new());
        let mut part2 = Solver::<CpuRuntime>::resume_with_subgrid(client, &snap, forcing_resume, sg)
            .expect("resume with subgrid");
        while part2.time() < 120.0 {
            part2.advance_to(120.0, 10_000).expect("part2 step");
        }
        let split_cells = part2.state().expect("split state").cells().to_vec();

        for (i, (a, b)) in full_cells.iter().zip(split_cells.iter()).enumerate() {
            assert!(
                (a.h - b.h).abs() < 1e-12 && (a.qx - b.qx).abs() < 1e-12,
                "cell {i} mismatch: full={a:?} split={b:?}"
            );
        }
    }

    #[test]
    fn subgrid_face_conveyance_ramps_smoothly_and_preserves_narrow_channel_flux() {
        // 2x1 coarse cells, 4x4 fine subcells (8x4 fine DEM):
        // Row 1 is a 1-subcell-wide channel (z=0m) inside z=5m banks.
        let mut fine = vec![5.0; 8 * 4];
        for c in 0..8 {
            fine[1 * 8 + c] = 0.0;
        }
        let grid = CartesianGrid::new(2, 1, 100.0, 100.0).expect("grid");
        let sg = SubgridTerrain::from_fine_dem(grid, 8, 4, &fine).expect("sg");
        let face = vec![FaceSupport {
            face: BoundaryFace {
                axis: Axis::X,
                column: 1,
                row: 0,
            },
            left: Some(0),
            right: Some(1),
        }];

        // Narrow channel at WSE = 1.0m -> h_coarse = 0.25m, face effective_depth = 0.25m -> phi = 1.0
        // (avoids 4x double-counting narrow width already encoded in h_coarse).
        let h_ch = sg.effective_depth_field(&[1.0, 0.0]);
        let phi_ch = sg.face_conveyance_fractions(&face, &[1.0, 0.0], &h_ch);
        assert!(
            (phi_ch[0] - 1.0).abs() < 1e-12,
            "narrow channel phi must be 1.0, got {}",
            phi_ch[0]
        );

        // Levee at z=5.0m blocking full face: below crest -> 0.0; 0.05m overtop -> smooth tiny ramp.
        let mut levee = vec![0.0; 8 * 4];
        for r in 0..4 {
            levee[r * 8 + 3] = 5.0;
        }
        let sg_levee = SubgridTerrain::from_fine_dem(grid, 8, 4, &levee).expect("levee");
        let phi_below = sg_levee.face_conveyance_fractions(&face, &[4.5, 0.0], &[4.5, 0.0]);
        assert_eq!(phi_below[0], 0.0);
        let phi_tiny = sg_levee.face_conveyance_fractions(&face, &[5.05, 0.0], &[5.05, 0.0]);
        assert!(
            phi_tiny[0] > 0.0 && phi_tiny[0] < 0.02,
            "slight levee overtopping must ramp smoothly from 0, got {}",
            phi_tiny[0]
        );
    }

    #[test]
    fn subgrid_subpixel_channel_lake_at_rest_and_inbank_routing() {
        // 3x1 coarse cells (dx = 1000m), flat fine DEM at z = 10.0m,
        // with a 30m-wide, 3.0m-deep sub-pixel river channel incised to z_ch = 7.0m.
        let fine = vec![10.0; 12 * 4];
        let grid = CartesianGrid::new(3, 1, 1000.0, 1000.0).expect("grid");
        let channels = vec![Some((30.0, 3.0)), Some((30.0, 3.0)), Some((30.0, 3.0))];
        let sg = SubgridTerrain::from_fine_dem(grid, 12, 4, &fine)
            .expect("sg")
            .with_channels(&channels)
            .expect("channels");

        // 1) Exact stage-volume inversion both in-bank (8.5m) and overbank (10.5m).
        for &stage in &[7.0, 8.5, 10.0, 10.5] {
            let v = sg.curves()[0].volume_at_stage(stage);
            let inv = sg.curves()[0].stage_at_volume(v);
            assert!(
                (inv - stage).abs() < 1e-11,
                "stage={stage} v={v} inv={inv}"
            );
        }

        // 2) Lake-at-rest in-bank (WSE = 8.5m, completely below the 10m DEM!).
        let init_rest = sg.effective_depth_field(&[8.5, 8.5, 8.5]);
        let coarse_bed: Vec<f64> = sg.curves().iter().map(|c| c.min_bed_m()).collect();
        let domain = Domain::new(
            grid,
            coarse_bed.clone(),
            vec![0.03; 3],
            vec![true; 3],
            SpatialReference::new("LOCAL".to_owned(), "MSL".to_owned(), [0.0, 0.0]).expect("crs"),
        )
        .expect("domain");
        let land = LandModel::new(&domain, vec![LandCell::Impervious; 3]).expect("land");
        let soil = SoilState::initialized(&land).expect("soil");
        let boundaries = BoundaryConditions::with_open_faces(&domain, vec![]).expect("bc");
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 60.0,
                rain: vec![0.0; 3],
                potential_evaporation: vec![0.0; 3],
                withdrawal: vec![0.0; 3],
                boundaries: vec![],
            }],
        )
        .expect("forcing");
        let flow = FlowState::new(
            &domain,
            init_rest
                .iter()
                .map(|&h| CellState { h, qx: 0.0, qy: 0.0 })
                .collect(),
        )
        .expect("flow");
        let mut solver = Solver::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new()),
            SolverSetup {
                domain: domain.clone(),
                land: land.clone(),
                boundaries: boundaries.clone(),
                forcing: forcing.clone(),
                config: SolverConfig::default(),
            },
            InitialState {
                time: 0.0,
                flow,
                soil: soil.clone(),
            },
        )
        .expect("solver")
        .with_subgrid_terrain(sg.clone())
        .expect("attach");
        while solver.time() < 60.0 {
            solver.advance_to(60.0, 10_000).expect("step");
        }
        let wse = solver.water_surface_elevation().expect("wse");
        for (i, cell) in solver.state().expect("st").cells().iter().enumerate() {
            assert!(cell.qx.abs() < 1e-12, "cell {i} qx={}", cell.qx);
            assert!((wse[i] - 8.5).abs() < 1e-11, "cell {i} wse={}", wse[i]);
        }

        // 3) In-bank flood wave (WSE = 9.5m in cell 0, dry in cells 1 & 2) routes downstream below 10m DEM!
        let init_wave = sg.effective_depth_field(&[9.5, 7.0, 7.0]);
        let flow_w = FlowState::new(
            &domain,
            init_wave
                .iter()
                .map(|&h| CellState { h, qx: 0.0, qy: 0.0 })
                .collect(),
        )
        .expect("flow_w");
        let mut solver_w = Solver::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new()),
            SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            InitialState {
                time: 0.0,
                flow: flow_w,
                soil,
            },
        )
        .expect("solver_w")
        .with_subgrid_terrain(sg)
        .expect("attach_w");
        while solver_w.time() < 60.0 {
            solver_w.advance_to(60.0, 10_000).expect("step_w");
        }
        let wse_w = solver_w.water_surface_elevation().expect("wse_w");
        assert!(
            wse_w[1] > 7.05 && wse_w[1] < 10.0,
            "in-bank 30m channel must route water below 10m DEM, got cell 1 wse={}",
            wse_w[1]
        );
    }
}
