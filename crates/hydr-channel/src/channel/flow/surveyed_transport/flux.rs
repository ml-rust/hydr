use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::assembly::{is_direct, numerical_range},
    conserved_update::{Parts, parts},
    context::Context,
    flux::{Flux, SurveyedRates},
    geometry::SurveyedGeometry,
    riemann_diagnostics::RiemannDiagnosticState,
    speed_memo::SpeedEntry,
    surveyed::work::PrismaticRiemannWork,
    surveyed_points::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{
            BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms,
            CertifiedUpperBracketAlgorithms,
        },
    },
};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Solve every face outside the direct end faces from the two-part cell points.
    ///
    /// The caller must leave at least one such face.
    pub(super) fn surveyed_point_riemann(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<PrismaticPointRiemannResult<R>, ChannelError> {
        let faces = self.surveyed_faces(geometry, fields, left, right)?;
        let (start, len) = numerical_range(self.count(), left, right);
        geometry
            .section
            .riemann_flux_points_tensor(
                &geometry.face_owners.narrow(0, start, len)?,
                &faces.left_area,
                &faces.left_discharge,
                &faces.right_area,
                &faces.right_discharge,
                &geometry.riemann,
            )
            .map_err(riemann_error)
    }

    pub(in crate::channel::flow) fn surveyed_flux(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Flux<R>, ChannelError> {
        let frame = self.surveyed_direct_frame(geometry, fields, left, right, true)?;
        let frame = frame.as_ref();
        let (start, len) = numerical_range(self.count(), left, right);
        // A kept candidate evaluation of these exact operands holds this solve and its speeds.
        let key = self.speed_key(fields, left, right)?;
        let kept = self.speed_memo.find(&key, len > 0);
        let result = match kept.as_ref().and_then(|entry| entry.riemann.clone()) {
            Some(result) => Some(result),
            None if len > 0 => Some(Arc::new(
                self.surveyed_point_riemann(geometry, fields, left, right)?,
            )),
            None => None,
        };
        let numerical = result
            .as_ref()
            .map(|result| {
                Ok::<_, ChannelError>((
                    self.surveyed_wall_water(result, left, right, start)?,
                    parts(result.momentum_flux_m4_per_s2()),
                    self.surveyed_width(result.water_flux_bound_m3_per_s())?,
                    self.surveyed_width(result.momentum_flux_bound_m4_per_s2())?,
                ))
            })
            .transpose()?;
        let (water, momentum, water_width, momentum_width) = match &numerical {
            Some((water, momentum, water_width, momentum_width)) => (
                Some(water),
                Some(momentum),
                Some(water_width),
                Some(momentum_width),
            ),
            None => (None, None, None, None),
        };
        let water =
            self.surveyed_direct_parts(frame, water, frame.map(|f| &f.longitudinal_water))?;
        let momentum =
            self.surveyed_direct_parts(frame, momentum, frame.map(|f| &f.momentum_flux_m4_per_s2))?;
        let water_flux_width = self.assemble_boundary_faces(
            frame,
            water_width,
            frame.map(|f| &f.water_flux_width_m3_per_s),
        )?;
        let momentum_flux_width = self.assemble_boundary_faces(
            frame,
            momentum_width,
            frame.map(|f| &f.momentum_flux_width_m4_per_s2),
        )?;
        let (speed, dry_fronts) = match kept {
            Some(entry) => (entry.speeds.outer.clone(), entry.speeds.fronts.clone()),
            None => {
                let speeds = self.surveyed_cfl_speeds(
                    geometry,
                    fields,
                    left,
                    right,
                    frame,
                    result.as_deref(),
                )?;
                let entry = SpeedEntry {
                    speeds: speeds.clone(),
                    riemann: result.clone(),
                };
                self.speed_memo.keep(key, entry);
                (speeds.outer, speeds.fronts)
            }
        };
        let zero = self.constant(&water.high, 0.0)?;
        // Direct rates carry their detected underflow bounds on the F64 path.
        let numerical_zero = numerical
            .as_ref()
            .map(|(water, ..)| self.constant(&water.high, 0.0))
            .transpose()?;
        let water_bound = self.assemble_boundary_faces(
            frame,
            numerical_zero.as_ref(),
            frame.map(|f| &f.water_underflow_bound),
        )?;
        let momentum_bound = self.assemble_boundary_faces(
            frame,
            numerical_zero.as_ref(),
            frame.map(|f| &f.momentum_underflow_bound),
        )?;
        let prescribed_water =
            self.prescribed_water_mask(frame, numerical_zero.as_ref(), left, right)?;
        let work = result
            .as_ref()
            .map_or_else(PrismaticRiemannWork::default, |r| r.work());
        let root_iterations = work
            .maximum_root_iterations
            .max(frame.map_or(0, |f| f.root_iterations));
        Ok(Flux {
            boundary_traces: frame.and_then(|f| f.traces.clone()),
            prescribed_water,
            water: water.high.clone(),
            momentum: momentum.high.clone(),
            surveyed: Some(SurveyedRates {
                water,
                momentum,
                direct: [is_direct(left), is_direct(right)],
            }),
            speed,
            internal_speed: None,
            dry_fronts,
            water_bound,
            momentum_bound,
            riemann: Some(RiemannDiagnosticState {
                water_flux_width,
                momentum_flux_width,
                root_iterations,
                surveyed_work: Some(work),
            }),
            left_pressure: zero.clone(),
            right_pressure: zero.clone(),
            left_reconstructed_pressure: zero.clone(),
            right_reconstructed_pressure: zero.clone(),
            left_pressure_bound: zero.clone(),
            right_pressure_bound: zero.clone(),
            left_reconstructed_pressure_bound: zero.clone(),
            right_reconstructed_pressure_bound: zero,
        })
    }

    /// Selected water point with every wall component set to `+0.0`.
    ///
    /// A wall face must enclose zero water flux and select a star or dry trace. The result
    /// holds the faces from `start` on. A wall end is never direct, so it lies in that range.
    fn surveyed_wall_water(
        &self,
        result: &PrismaticPointRiemannResult<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
        start: usize,
    ) -> Result<Parts<R>, ChannelError> {
        let c = &self.client;
        let bounds = result.water_flux_bound_m3_per_s();
        let mut water = parts(result.water_flux_m3_per_s());
        for (boundary, face) in [(left, 0), (right, self.count())] {
            if !matches!(boundary, ChannelTensorBoundary::Wall) {
                continue;
            }
            let lane = face - start;
            let zero = self.constant(&water.high.narrow(0, lane, 1)?, 0.0)?;
            let contains_zero = c.mul(
                &c.le(&bounds.lower().narrow(0, lane, 1)?, &zero)?,
                &c.ge(&bounds.upper().narrow(0, lane, 1)?, &zero)?,
            )?;
            if !self.all(&contains_zero)? {
                return Err(ChannelError::numerical(format!(
                    "surveyed wall face={face} excludes zero water flux: inspect reflected Riemann arithmetic"
                )));
            }
            let code = result.trace_kind().narrow(0, lane, 1)?.item::<i64>()?;
            let kind = PrismaticPointTraceKind::from_code(code);
            if !matches!(
                kind,
                Some(PrismaticPointTraceKind::Star | PrismaticPointTraceKind::Dry)
            ) {
                return Err(ChannelError::numerical(format!(
                    "surveyed wall face={face} selects trace {kind:?} (code {code}), not a star or dry trace: inspect the reflected wall state"
                )));
            }
            water = Parts {
                high: self.replace_face(&water.high, lane, &zero)?,
                low: self.replace_face(&water.low, lane, &zero)?,
                bound: self.replace_face(&water.bound, lane, &zero)?,
            };
        }
        Ok(water)
    }

    /// Copy of `value` with `face` set to the one-element `replacement`.
    fn replace_face(
        &self,
        value: &Tensor<R>,
        face: usize,
        replacement: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let faces = value.numel();
        let mut pieces = Vec::with_capacity(3);
        if face > 0 {
            pieces.push(value.narrow(0, 0, face)?);
        }
        pieces.push(replacement.clone());
        if face + 1 < faces {
            pieces.push(value.narrow(0, face + 1, faces - face - 1)?);
        }
        let pieces: Vec<&Tensor<R>> = pieces.iter().collect();
        Ok(self.client.cat(&pieces, 0)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelError, ChannelReach, ChannelTensorBoundary,
        geometry::TensorGeometry,
        surveyed_update::tests::{Client, TestResult, cpu, moving_reach},
    };
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn bits<R: Runtime<DType = DType>>(value: &Tensor<R>) -> TestResult<Vec<u64>> {
        Ok(value
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect())
    }

    fn read<R: Runtime<DType = DType>>(value: &Tensor<R>) -> TestResult<Vec<f64>> {
        Ok(value.try_to_vec::<f64>()?)
    }

    /// Three unequal moving cells between walls.
    fn reach<R>(client: R::Client) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        moving_reach::<R>(client, &[20.0, 10.0, 15.0], &[4.0, -2.0, 3.0], &[0.0; 3])
    }

    fn walls_carry_zero_water_and_the_entry_momentum<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client)?;
        let context = &reach.context;
        let TensorGeometry::Surveyed(geometry) = &context.geometry else {
            return Err("surveyed geometry required".into());
        };
        let wall = ChannelTensorBoundary::Wall;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let flux = context.flux(&fields, &wall, &wall)?;
        let result = context.surveyed_point_riemann(geometry, &fields, &wall, &wall)?;
        let rates = flux.surveyed.as_ref().ok_or("surveyed rates required")?;
        let n = context.count();
        for face in [0, n] {
            for value in [&rates.water.high, &rates.water.low, &rates.water.bound] {
                assert_eq!(value.narrow(0, face, 1)?.item::<f64>()?.to_bits(), 0);
            }
        }
        let water = result.water_flux_m3_per_s();
        for (actual, expected) in [
            (&rates.water.high, water.high()),
            (&rates.water.low, water.low()),
            (&rates.water.bound, water.abs_error()),
        ] {
            assert_eq!(
                bits(&actual.narrow(0, 1, n - 1)?)?,
                bits(&expected.narrow(0, 1, n - 1)?)?
            );
        }
        let momentum = result.momentum_flux_m4_per_s2();
        assert_eq!(bits(&rates.momentum.high)?, bits(momentum.high())?);
        assert_eq!(bits(&rates.momentum.low)?, bits(momentum.low())?);
        assert_eq!(bits(&rates.momentum.bound)?, bits(momentum.abs_error())?);
        assert_eq!(bits(&flux.water)?, bits(&rates.water.high)?);
        assert_eq!(bits(&flux.momentum)?, bits(&rates.momentum.high)?);
        Ok(())
    }

    fn widths_and_speed_come_from_the_entry<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client)?;
        let context = &reach.context;
        let TensorGeometry::Surveyed(geometry) = &context.geometry else {
            return Err("surveyed geometry required".into());
        };
        let wall = ChannelTensorBoundary::Wall;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let flux = context.flux(&fields, &wall, &wall)?;
        let result = context.surveyed_point_riemann(geometry, &fields, &wall, &wall)?;
        let riemann = flux
            .riemann
            .as_ref()
            .ok_or("Riemann diagnostics required")?;
        for (width, bound) in [
            (
                &riemann.water_flux_width,
                result.water_flux_bound_m3_per_s(),
            ),
            (
                &riemann.momentum_flux_width,
                result.momentum_flux_bound_m4_per_s2(),
            ),
        ] {
            let lower = read(bound.lower())?;
            let upper = read(bound.upper())?;
            for (face, width) in read(width)?.into_iter().enumerate() {
                // The rounded difference is within one ulp of the exact width.
                let nearest = upper[face] - lower[face];
                assert!(
                    width == nearest || width == nearest.next_up(),
                    "face={face}, width={width:e}, nearest={nearest:e}"
                );
                assert!(width >= 0.0);
            }
        }
        let lower = read(result.wave_speed_m_per_s().lower())?;
        let upper = read(result.wave_speed_m_per_s().upper())?;
        // Each face splits its signed interval into the speeds entering its two cells.
        let rightward: Vec<f64> = upper.iter().map(|u| u.max(0.0)).collect();
        let leftward: Vec<f64> = lower.iter().map(|l| (-l).max(0.0)).collect();
        assert_eq!(read(&flux.speed.rightward)?, rightward);
        assert_eq!(read(&flux.speed.leftward)?, leftward);
        let speed = context.candidate_speed(&fields, &wall, &wall)?;
        assert_eq!(read(&speed.outer.rightward)?, rightward);
        assert_eq!(read(&speed.outer.leftward)?, leftward);
        assert_eq!(riemann.surveyed_work, Some(result.work()));
        Ok(())
    }

    fn rejected_start_state_names_the_reason<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        // Six of seven cap square metres moving at 5 m/s into the left wall.
        let reach = moving_reach::<R>(client, &[30.0, 30.0], &[-150.0, -150.0], &[0.0; 2])?;
        let context = &reach.context;
        let wall = ChannelTensorBoundary::Wall;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let error = context
            .flux(&fields, &wall, &wall)
            .err()
            .ok_or("start flux must reject")?;
        let ChannelError::Numerical { context: text, .. } = &error else {
            return Err(format!("unexpected error: {error}").into());
        };
        assert!(
            text.contains("surveyed reach Riemann CapExceeded"),
            "{text}"
        );
        assert!(text.contains("face=Some(0)"), "{text}");
        Ok(())
    }

    #[test]
    fn cpu_surveyed_walls_carry_zero_water_and_the_entry_momentum() -> TestResult {
        walls_carry_zero_water_and_the_entry_momentum::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_surveyed_widths_and_speed_come_from_the_entry() -> TestResult {
        widths_and_speed_come_from_the_entry::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_surveyed_rejected_start_state_names_the_reason() -> TestResult {
        rejected_start_state_names_the_reason::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_surveyed_walls_carry_zero_water_and_the_entry_momentum() -> TestResult {
            walls_carry_zero_water_and_the_entry_momentum::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }

        #[test]
        fn cuda_surveyed_widths_and_speed_come_from_the_entry() -> TestResult {
            widths_and_speed_come_from_the_entry::<CudaRuntime>(CudaClient::new(CudaDevice::new(
                0,
            ))?)
        }

        #[test]
        fn cuda_surveyed_rejected_start_state_names_the_reason() -> TestResult {
            rejected_start_state_names_the_reason::<CudaRuntime>(CudaClient::new(CudaDevice::new(
                0,
            ))?)
        }
    }
}
