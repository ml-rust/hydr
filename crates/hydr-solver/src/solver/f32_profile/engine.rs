//! Opt-in `F32` 2D shallow-water solver engine for CPU, CUDA, and WGPU.

use super::boundaries::{
    SurfaceF32Boundaries, SurfaceF32CoastalForcing, apply_coastal_forcing_f32, apply_manning_f32,
};
use super::flux::{sweep_x, sweep_y};
use super::soil::{SoilStateF32, SurfaceF32Soil};
use super::state::{
    SURFACE_SOLVER_F32_CONTRACT_VERSION, SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA,
    SurfaceSolverF32Config, SurfaceSolverF32Ledger, SurfaceSolverF32SnapshotData,
};
use crate::contract::EngineError;
use crate::model::{CellState, Domain, FlowState};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Opt-in 2D shallow-water surface solver executing natively in `DType::F32`.
///
/// Supports `CpuRuntime`, `CudaRuntime`, and `WgpuRuntime` (Vulkan, Metal, DX12), with open
/// exterior boundaries, Modified Green-Ampt soil infiltration, and coastal wind/Coriolis/pressure
/// forcing. Use [`super::super::Solver`] when certified `F64` precision is required.
pub struct SurfaceSolverF32<R: Runtime<DType = DType>> {
    client: R::Client,
    domain: Domain,
    config: SurfaceSolverF32Config,
    boundaries: SurfaceF32Boundaries,
    coastal: SurfaceF32CoastalForcing,
    soil: Option<SoilStateF32<R>>,
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
    bed: Tensor<R>,
    manning_sq: Tensor<R>,
    active: Tensor<R>,
    h: Tensor<R>,
    qx: Tensor<R>,
    qy: Tensor<R>,
    time_s: f64,
    accepted_steps: u64,
    ledger: SurfaceSolverF32Ledger,
}

impl<R: Runtime<DType = DType>> SurfaceSolverF32<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    /// Constructs an `F32` 2D shallow-water solver on `client` with impermeable walls and
    /// impervious land. Attach open boundaries, Green-Ampt soil, or coastal forcing via
    /// [`Self::with_boundaries`], [`Self::with_soil`], and [`Self::with_coastal_forcing`].
    pub fn new(
        client: R::Client,
        domain: Domain,
        initial: FlowState,
        config: SurfaceSolverF32Config,
    ) -> Result<Self, EngineError> {
        if !R::computes_float_dtype(DType::F32) {
            return Err(EngineError::rejected(
                "surface solver f32 runtime",
                format!("runtime `{}` does not compute F32", R::name()),
            ));
        }
        config.validate()?;
        let grid = domain.grid();
        let (nx, ny) = grid.dimensions();
        let (dx_f64, dy_f64) = grid.spacing();
        let count = grid.cell_count();
        if initial.cells().len() != count {
            return Err(EngineError::rejected(
                "surface solver f32 initial state",
                format!(
                    "initial cells ({}) != domain cells ({count})",
                    initial.cells().len()
                ),
            ));
        }

        let device = client.device();
        let shape = [ny, nx];
        let bed_vec: Vec<f32> = domain.bed().iter().map(|&v| v as f32).collect();
        let manning_sq_vec: Vec<f32> = domain.manning().iter().map(|&n| (n * n) as f32).collect();
        let active_vec: Vec<f32> = domain
            .active()
            .iter()
            .map(|&a| if a { 1.0_f32 } else { 0.0_f32 })
            .collect();
        let h_vec: Vec<f32> = initial.cells().iter().map(|c| c.h as f32).collect();
        let qx_vec: Vec<f32> = initial.cells().iter().map(|c| c.qx as f32).collect();
        let qy_vec: Vec<f32> = initial.cells().iter().map(|c| c.qy as f32).collect();

        let cell_area = grid.cell_area();
        let initial_storage_m3: f64 = h_vec.iter().map(|&v| f64::from(v) * cell_area).sum();

        Ok(Self {
            bed: Tensor::<R>::from_slice(&bed_vec, &shape, device)?,
            manning_sq: Tensor::<R>::from_slice(&manning_sq_vec, &shape, device)?,
            active: Tensor::<R>::from_slice(&active_vec, &shape, device)?,
            h: Tensor::<R>::from_slice(&h_vec, &shape, device)?,
            qx: Tensor::<R>::from_slice(&qx_vec, &shape, device)?,
            qy: Tensor::<R>::from_slice(&qy_vec, &shape, device)?,
            client,
            domain,
            config,
            boundaries: SurfaceF32Boundaries::impermeable_walls(),
            coastal: SurfaceF32CoastalForcing::default(),
            soil: None,
            nx,
            ny,
            dx: dx_f64 as f32,
            dy: dy_f64 as f32,
            time_s: 0.0,
            accepted_steps: 0,
            ledger: SurfaceSolverF32Ledger::zero(initial_storage_m3),
        })
    }

    /// Configures exterior open boundaries (stage, discharge, transmissive, or wall).
    pub fn with_boundaries(
        mut self,
        boundaries: SurfaceF32Boundaries,
    ) -> Result<Self, EngineError> {
        self.set_boundaries(boundaries)?;
        Ok(self)
    }

    /// Updates exterior boundaries between intervals (e.g., tidal stage or hydrograph discharge).
    pub fn set_boundaries(&mut self, boundaries: SurfaceF32Boundaries) -> Result<(), EngineError> {
        boundaries.validate()?;
        self.boundaries = boundaries;
        Ok(())
    }

    /// Attaches Modified Green-Ampt soil infiltration in `DType::F32`.
    pub fn with_soil(mut self, soil: &SurfaceF32Soil) -> Result<Self, EngineError> {
        self.soil = Some(SoilStateF32::new(&self.client, soil, self.ny, self.nx)?);
        Ok(self)
    }

    /// Configures opt-in coastal wind stress, Coriolis, and atmospheric pressure gradient forcing.
    pub fn with_coastal_forcing(
        mut self,
        coastal: SurfaceF32CoastalForcing,
    ) -> Result<Self, EngineError> {
        self.set_coastal_forcing(coastal)?;
        Ok(self)
    }

    /// Updates coastal wind/Coriolis/pressure forcing between intervals.
    pub fn set_coastal_forcing(
        &mut self,
        coastal: SurfaceF32CoastalForcing,
    ) -> Result<(), EngineError> {
        coastal.validate()?;
        self.coastal = coastal;
        Ok(())
    }

    /// Domain geometry.
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Numerical configuration.
    pub fn config(&self) -> SurfaceSolverF32Config {
        self.config
    }

    /// Active exterior boundary configuration.
    pub fn boundaries(&self) -> SurfaceF32Boundaries {
        self.boundaries
    }

    /// Active coastal momentum forcing.
    pub fn coastal_forcing(&self) -> SurfaceF32CoastalForcing {
        self.coastal
    }

    /// Current accepted simulation time in seconds.
    pub fn time_s(&self) -> f64 {
        self.time_s
    }

    /// Number of accepted steps.
    pub fn accepted_steps(&self) -> u64 {
        self.accepted_steps
    }

    /// Current cumulative SI water-balance ledger.
    pub fn ledger(&self) -> SurfaceSolverF32Ledger {
        self.ledger
    }

    /// Returns cumulative Modified Green-Ampt infiltrated depth (`m`) per cell if soil is attached.
    pub fn cumulative_infiltration_m(&self) -> Result<Option<Vec<f32>>, EngineError> {
        match &self.soil {
            Some(soil) => Ok(Some(soil.cumulative_to_vec()?)),
            None => Ok(None),
        }
    }

    /// Copies the accepted depth and discharge fields back to host `CellState`s.
    pub fn state(&self) -> Result<Vec<CellState>, EngineError> {
        let h = self.h.try_to_vec::<f32>()?;
        let qx = self.qx.try_to_vec::<f32>()?;
        let qy = self.qy.try_to_vec::<f32>()?;
        Ok(h.into_iter()
            .zip(qx)
            .zip(qy)
            .map(|((h, qx), qy)| CellState {
                h: f64::from(h),
                qx: f64::from(qx),
                qy: f64::from(qy),
            })
            .collect())
    }

    /// Advances to `target_s` under constant per-cell `rain_m_per_s`, `evap_m_per_s`, and
    /// `withdrawal_m_per_s` (`m/s`), shortening steps automatically for CFL and positivity.
    pub fn advance_to(
        &mut self,
        target_s: f64,
        rain_m_per_s: &[f32],
        evap_m_per_s: &[f32],
        withdrawal_m_per_s: &[f32],
        max_steps: usize,
    ) -> Result<usize, EngineError> {
        if !target_s.is_finite() || target_s < self.time_s {
            return Err(EngineError::rejected(
                "surface solver f32 target",
                format!(
                    "target_s ({target_s}) must be finite and >= time_s ({})",
                    self.time_s
                ),
            ));
        }
        let count = self.nx * self.ny;
        for (label, slice) in [
            ("rain_m_per_s", rain_m_per_s),
            ("evap_m_per_s", evap_m_per_s),
            ("withdrawal_m_per_s", withdrawal_m_per_s),
        ] {
            if slice.len() != count || slice.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(EngineError::rejected(
                    "surface solver f32 forcing",
                    format!("{label} must have {count} finite nonnegative entries"),
                ));
            }
        }

        let device = self.client.device();
        let shape = [self.ny, self.nx];
        let rain_t = self.client.mul(
            &Tensor::<R>::from_slice(rain_m_per_s, &shape, device)?,
            &self.active,
        )?;
        let evap_t = self.client.mul(
            &Tensor::<R>::from_slice(evap_m_per_s, &shape, device)?,
            &self.active,
        )?;
        let with_t = self.client.mul(
            &Tensor::<R>::from_slice(withdrawal_m_per_s, &shape, device)?,
            &self.active,
        )?;

        let mut steps_taken = 0usize;
        while self.time_s < target_s - 1e-12 {
            if steps_taken >= max_steps {
                return Err(EngineError::rejected(
                    "surface solver f32 step budget",
                    format!("reached max_steps ({max_steps}) at time_s={}", self.time_s),
                ));
            }
            let req_end = (self.time_s + self.config.max_step_s).min(target_s);
            self.take_one_step(req_end, &rain_t, &evap_t, &with_t)?;
            steps_taken += 1;
        }
        Ok(steps_taken)
    }

    fn take_one_step(
        &mut self,
        requested_end_s: f64,
        rain_t: &Tensor<R>,
        evap_t: &Tensor<R>,
        with_t: &Tensor<R>,
    ) -> Result<(), EngineError> {
        let mut dt = requested_end_s - self.time_s;
        let shape = [self.ny, self.nx];
        let zero = Tensor::<R>::zeros(&shape, DType::F32, self.client.device())?;

        for retry in 0..=self.config.max_retries {
            if retry > 0 && dt < self.config.min_step_s {
                return Err(EngineError::rejected(
                    "surface solver f32 step",
                    format!(
                        "step shortened to {dt} s below min_step_s ({})",
                        self.config.min_step_s
                    ),
                ));
            }
            let dt_f32 = dt as f32;

            // 1. Sources (rain -> optional Green-Ampt infiltration -> capped evap -> capped withdrawal).
            let h_rain = self
                .client
                .add(&self.h, &self.client.mul_scalar(rain_t, f64::from(dt_f32))?)?;
            let (h_soil, infil_act, next_soil_cum) = match &self.soil {
                Some(soil) => {
                    let (hs, inf, cum) = soil.step(&self.client, &h_rain, &self.active, dt_f32)?;
                    (hs, Some(inf), Some(cum))
                }
                None => (h_rain, None, None),
            };
            let evap_dem = self.client.mul_scalar(evap_t, f64::from(dt_f32))?;
            let evap_act = self.client.minimum(&h_soil, &evap_dem)?;
            let h_evap = self.client.sub(&h_soil, &evap_act)?;
            let with_dem = self.client.mul_scalar(with_t, f64::from(dt_f32))?;
            let with_act = self.client.minimum(&h_evap, &with_dem)?;
            let h_src = self.client.sub(&h_evap, &with_act)?;

            // 2. X and Y HLL sweeps + hydrostatic bed corrections + open boundaries.
            let sx = sweep_x(
                &self.client,
                &h_src,
                &self.qx,
                &self.qy,
                &self.bed,
                &self.active,
                self.boundaries.west,
                self.boundaries.east,
                self.dx,
                self.dy,
                self.config.gravity,
                self.config.dry_depth_m,
                self.config.spatial_order,
            )?;
            let sy = sweep_y(
                &self.client,
                &h_src,
                &self.qx,
                &self.qy,
                &self.bed,
                &self.active,
                self.boundaries.south,
                self.boundaries.north,
                self.dx,
                self.dy,
                self.config.gravity,
                self.config.dry_depth_m,
                self.config.spatial_order,
            )?;

            // 3. CFL check.
            let wave_rate = f64::from(sx.max_speed) / f64::from(self.dx)
                + f64::from(sy.max_speed) / f64::from(self.dy);
            if wave_rate > 0.0 {
                let cfl_dt = f64::from(self.config.cfl) / wave_rate;
                if dt > cfl_dt {
                    dt = 0.95 * cfl_dt;
                    continue;
                }
            }

            // 4. Explicit update + positivity check.
            let dh_tot = self.client.add(&sx.dh, &sy.dh)?;
            let h_next = self
                .client
                .add(&h_src, &self.client.mul_scalar(&dh_tot, f64::from(dt_f32))?)?;
            let min_h = self.client.min(&h_next, &[0, 1], false)?.item::<f32>()?;
            if !min_h.is_finite() || min_h < -1e-7 {
                dt *= 0.5;
                continue;
            }
            let h_clean = self.client.maximum(&h_next, &zero)?;

            let dqx_tot = self.client.add(&sx.dqx, &sy.dqx)?;
            let dqy_tot = self.client.add(&sx.dqy, &sy.dqy)?;
            let qx_adv = self.client.add(
                &self.qx,
                &self.client.mul_scalar(&dqx_tot, f64::from(dt_f32))?,
            )?;
            let qy_adv = self.client.add(
                &self.qy,
                &self.client.mul_scalar(&dqy_tot, f64::from(dt_f32))?,
            )?;

            // 5. Coastal wind/pressure/Coriolis forcing + semi-implicit Manning friction.
            let (qx_coast, qy_coast) = apply_coastal_forcing_f32(
                &self.client,
                &h_clean,
                &qx_adv,
                &qy_adv,
                &self.active,
                self.coastal,
                dt_f32,
                self.config.dry_depth_m,
            )?;
            let (qx_next, qy_next) = apply_manning_f32(
                &self.client,
                &h_clean,
                &qx_coast,
                &qy_coast,
                &self.manning_sq,
                dt_f32,
                self.config.gravity,
                self.config.dry_depth_m,
            )?;

            // Commit accepted step and update SI ledger.
            let area = self.domain.grid().cell_area();
            let step_rain =
                f64::from(self.client.sum(rain_t, &[0, 1], false)?.item::<f32>()?) * dt * area;
            let step_evap =
                f64::from(self.client.sum(&evap_act, &[0, 1], false)?.item::<f32>()?) * area;
            let step_with =
                f64::from(self.client.sum(&with_act, &[0, 1], false)?.item::<f32>()?) * area;
            let step_infil = match &infil_act {
                Some(inf) => f64::from(self.client.sum(inf, &[0, 1], false)?.item::<f32>()?) * area,
                None => 0.0,
            };
            let step_b_in = (sx.boundary_in_m3_per_s + sy.boundary_in_m3_per_s) * dt;
            let step_b_out = (sx.boundary_out_m3_per_s + sy.boundary_out_m3_per_s) * dt;
            let final_storage =
                f64::from(self.client.sum(&h_clean, &[0, 1], false)?.item::<f32>()?) * area;

            self.h = h_clean;
            self.qx = qx_next;
            self.qy = qy_next;
            if let (Some(soil), Some(cum)) = (&mut self.soil, next_soil_cum) {
                soil.set_cumulative(cum);
            }
            self.time_s += dt;
            self.accepted_steps = self.accepted_steps.saturating_add(1);
            self.ledger.record_step(
                step_rain,
                step_evap,
                step_with,
                step_infil,
                step_b_in,
                step_b_out,
                final_storage,
            );
            return Ok(());
        }

        Err(EngineError::rejected(
            "surface solver f32 step",
            format!("exceeded max_retries ({})", self.config.max_retries),
        ))
    }

    /// Exports a portable complete `F32` checkpoint.
    pub fn snapshot_data(&self) -> Result<SurfaceSolverF32SnapshotData, EngineError> {
        Ok(SurfaceSolverF32SnapshotData {
            schema: SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA.to_string(),
            contract: SURFACE_SOLVER_F32_CONTRACT_VERSION.to_string(),
            time_s: self.time_s,
            accepted_steps: self.accepted_steps,
            h_m: self.h.try_to_vec::<f32>()?,
            qx_m2_per_s: self.qx.try_to_vec::<f32>()?,
            qy_m2_per_s: self.qy.try_to_vec::<f32>()?,
            cumulative_infiltration_m: self.cumulative_infiltration_m()?,
            ledger: self.ledger,
        })
    }

    /// Atomically restores a previously exported `F32` checkpoint.
    pub fn restore_data(&mut self, data: &SurfaceSolverF32SnapshotData) -> Result<(), EngineError> {
        data.validate(self.nx * self.ny)?;
        match (&mut self.soil, &data.cumulative_infiltration_m) {
            (Some(soil), Some(vals)) => {
                soil.restore_cumulative(&self.client, vals, self.ny, self.nx)?;
            }
            (None, None) => {}
            _ => {
                return Err(EngineError::rejected(
                    "surface solver f32 snapshot",
                    "snapshot soil infiltration presence disagrees with solver soil attachment",
                ));
            }
        }
        let shape = [self.ny, self.nx];
        let device = self.client.device();
        self.h = Tensor::<R>::from_slice(&data.h_m, &shape, device)?;
        self.qx = Tensor::<R>::from_slice(&data.qx_m2_per_s, &shape, device)?;
        self.qy = Tensor::<R>::from_slice(&data.qy_m2_per_s, &shape, device)?;
        self.time_s = data.time_s;
        self.accepted_steps = data.accepted_steps;
        self.ledger = data.ledger;
        Ok(())
    }
}
