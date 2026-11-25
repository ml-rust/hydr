//! Native-async WebGPU execution for [`SurfaceSolverF32<WgpuRuntime>`].
//!
//! Avoids all synchronous host readbacks (`Tensor::item`, `Tensor::try_to_vec`) so the solver
//! runs without deadlocking in browser `wasm32` WebGPU contexts.

use super::boundaries::{apply_coastal_forcing_f32, apply_manning_f32};
use super::engine::SurfaceSolverF32;
use super::flux::{sweep_x_tensors, sweep_y_tensors};
use super::state::{
    SURFACE_SOLVER_F32_CONTRACT_VERSION, SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA,
    SurfaceSolverF32Ledger, SurfaceSolverF32SnapshotData,
};
use crate::contract::EngineError;
use crate::model::CellState;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, ShapeOps};
use numr::runtime::RuntimeClient;
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

impl SurfaceSolverF32<WgpuRuntime> {
    /// Copies the accepted water depth field (`m`) back to the host asynchronously.
    pub async fn depth_m_async(&self) -> Result<Vec<f32>, EngineError> {
        Ok(self.client.read_tensor_to_vec_async::<f32>(&self.h).await?)
    }

    /// Copies the accepted depth and discharge fields back to host [`CellState`]s asynchronously.
    pub async fn state_async(&self) -> Result<Vec<CellState>, EngineError> {
        let h = self.client.read_tensor_to_vec_async::<f32>(&self.h).await?;
        let qx = self
            .client
            .read_tensor_to_vec_async::<f32>(&self.qx)
            .await?;
        let qy = self
            .client
            .read_tensor_to_vec_async::<f32>(&self.qy)
            .await?;
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

    /// Returns cumulative Modified Green-Ampt infiltrated depth (`m`) per cell asynchronously if soil is attached.
    pub async fn cumulative_infiltration_m_async(&self) -> Result<Option<Vec<f32>>, EngineError> {
        match &self.soil {
            Some(soil) => Ok(Some(
                self.client
                    .read_tensor_to_vec_async::<f32>(soil.cumulative_tensor())
                    .await?,
            )),
            None => Ok(None),
        }
    }

    /// Exports a portable complete `F32` checkpoint asynchronously.
    pub async fn snapshot_data_async(&self) -> Result<SurfaceSolverF32SnapshotData, EngineError> {
        Ok(SurfaceSolverF32SnapshotData {
            schema: SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA.to_string(),
            contract: SURFACE_SOLVER_F32_CONTRACT_VERSION.to_string(),
            time_s: self.time_s,
            accepted_steps: self.accepted_steps,
            h_m: self.client.read_tensor_to_vec_async::<f32>(&self.h).await?,
            qx_m2_per_s: self
                .client
                .read_tensor_to_vec_async::<f32>(&self.qx)
                .await?,
            qy_m2_per_s: self
                .client
                .read_tensor_to_vec_async::<f32>(&self.qy)
                .await?,
            cumulative_infiltration_m: self.cumulative_infiltration_m_async().await?,
            ledger: self.ledger,
        })
    }

    /// Advances to `target_s` asynchronously under constant per-cell forcing without blocking
    /// the calling thread on WebGPU readbacks.
    pub async fn advance_to_async(
        &mut self,
        target_s: f64,
        rain_m_per_s: &[f32],
        evap_m_per_s: &[f32],
        withdrawal_m_per_s: &[f32],
        max_steps: usize,
    ) -> Result<usize, EngineError> {
        self.advance_controlled_async(
            target_s,
            rain_m_per_s,
            evap_m_per_s,
            withdrawal_m_per_s,
            max_steps,
            || false,
            |_, _, _| {},
        )
        .await
    }

    /// Advances to `target_s` asynchronously with cancellation and per-accepted-step observation.
    ///
    /// `on_accepted` receives `(time_s, accepted_steps, ledger)` after each accepted step without
    /// forcing a full-grid GPU readback. Call [`Self::depth_m_async`] or [`Self::state_async`]
    /// when a frame readback is needed.
    #[allow(clippy::too_many_arguments)]
    pub async fn advance_controlled_async(
        &mut self,
        target_s: f64,
        rain_m_per_s: &[f32],
        evap_m_per_s: &[f32],
        withdrawal_m_per_s: &[f32],
        max_steps: usize,
        mut cancelled: impl FnMut() -> bool,
        mut on_accepted: impl FnMut(f64, u64, SurfaceSolverF32Ledger),
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
            &Tensor::<WgpuRuntime>::from_slice(rain_m_per_s, &shape, device)?,
            &self.active,
        )?;
        let evap_t = self.client.mul(
            &Tensor::<WgpuRuntime>::from_slice(evap_m_per_s, &shape, device)?,
            &self.active,
        )?;
        let with_t = self.client.mul(
            &Tensor::<WgpuRuntime>::from_slice(withdrawal_m_per_s, &shape, device)?,
            &self.active,
        )?;

        let mut steps_taken = 0usize;
        while self.time_s < target_s - 1e-12 {
            if cancelled() {
                break;
            }
            if steps_taken >= max_steps {
                return Err(EngineError::rejected(
                    "surface solver f32 step budget",
                    format!("reached max_steps ({max_steps}) at time_s={}", self.time_s),
                ));
            }
            let req_end = (self.time_s + self.config.max_step_s).min(target_s);
            self.take_one_step_async(req_end, &rain_t, &evap_t, &with_t)
                .await?;
            steps_taken += 1;
            on_accepted(self.time_s, self.accepted_steps, self.ledger);
        }
        Ok(steps_taken)
    }

    async fn take_one_step_async(
        &mut self,
        requested_end_s: f64,
        rain_t: &Tensor<WgpuRuntime>,
        evap_t: &Tensor<WgpuRuntime>,
        with_t: &Tensor<WgpuRuntime>,
    ) -> Result<(), EngineError> {
        let mut dt = requested_end_s - self.time_s;
        let shape = [self.ny, self.nx];
        let zero = Tensor::<WgpuRuntime>::zeros(&shape, DType::F32, self.client.device())?;

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

            // 2. X and Y HLL sweeps (tensor-only, zero sync readbacks).
            let sx = sweep_x_tensors(
                &self.client,
                &h_src,
                &self.qx,
                &self.qy,
                &self.bed,
                &self.active,
                self.boundaries.west,
                self.boundaries.east,
                self.open_masks.west.as_ref(),
                self.open_masks.east.as_ref(),
                self.dx,
                self.config.gravity,
                self.config.dry_depth_m,
                self.config.spatial_order,
            )?;
            let sy = sweep_y_tensors(
                &self.client,
                &h_src,
                &self.qx,
                &self.qy,
                &self.bed,
                &self.active,
                self.boundaries.south,
                self.boundaries.north,
                self.open_masks.south.as_ref(),
                self.open_masks.north.as_ref(),
                self.dy,
                self.config.gravity,
                self.config.dry_depth_m,
                self.config.spatial_order,
            )?;

            // 3. CFL check via a single batched async readback of [max_speed_x, max_speed_y].
            let speeds_t = self.client.cat(&[&sx.max_speed, &sy.max_speed], 0)?;
            let speeds = self
                .client
                .read_tensor_to_vec_async::<f32>(&speeds_t)
                .await?;
            let max_speed_x = speeds.first().copied().unwrap_or(0.0);
            let max_speed_y = speeds.get(1).copied().unwrap_or(0.0);
            let wave_rate = f64::from(max_speed_x) / f64::from(self.dx)
                + f64::from(max_speed_y) / f64::from(self.dy);
            if wave_rate > 0.0 {
                let cfl_dt = f64::from(self.config.cfl) / wave_rate;
                if dt > cfl_dt {
                    dt = 0.95 * cfl_dt;
                    continue;
                }
            }

            // 4. Explicit update + async positivity check.
            let dh_tot = self.client.add(&sx.dh, &sy.dh)?;
            let h_next = self
                .client
                .add(&h_src, &self.client.mul_scalar(&dh_tot, f64::from(dt_f32))?)?;
            let min_h_t = self.client.min(&h_next, &[0, 1], false)?;
            let min_h = self.client.read_scalar_async::<f32>(&min_h_t).await?;
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

            // 6. Commit accepted step and update SI ledger via one batched async readback.
            let area = self.domain.grid().cell_area();
            let rain_sum = self.client.sum(rain_t, &[0, 1], false)?.reshape(&[1])?;
            let evap_sum = self.client.sum(&evap_act, &[0, 1], false)?.reshape(&[1])?;
            let with_sum = self.client.sum(&with_act, &[0, 1], false)?.reshape(&[1])?;
            let infil_sum = match &infil_act {
                Some(inf) => self.client.sum(inf, &[0, 1], false)?.reshape(&[1])?,
                None => Tensor::<WgpuRuntime>::zeros(&[1], DType::F32, self.client.device())?,
            };
            let h_sum = self.client.sum(&h_clean, &[0, 1], false)?.reshape(&[1])?;
            let ledger_t = self.client.cat(
                &[
                    &rain_sum,
                    &evap_sum,
                    &with_sum,
                    &infil_sum,
                    &sx.boundary_in_flux,
                    &sx.boundary_out_flux,
                    &sy.boundary_in_flux,
                    &sy.boundary_out_flux,
                    &h_sum,
                ],
                0,
            )?;
            let lvals = self
                .client
                .read_tensor_to_vec_async::<f32>(&ledger_t)
                .await?;
            let get = |i: usize| f64::from(lvals.get(i).copied().unwrap_or(0.0));
            let step_rain = get(0) * dt * area;
            let step_evap = get(1) * area;
            let step_with = get(2) * area;
            let step_infil = get(3) * area;
            let step_b_in = (get(4) * f64::from(self.dy) + get(6) * f64::from(self.dx)) * dt;
            let step_b_out = (get(5) * f64::from(self.dy) + get(7) * f64::from(self.dx)) * dt;
            let final_storage = get(8) * area;

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
}
