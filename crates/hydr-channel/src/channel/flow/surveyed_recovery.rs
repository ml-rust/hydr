//! Surveyed area and discharge recovery from the two-part conserved state.

use super::{
    ChannelError,
    conserved_point::ConservedPoint,
    conserved_update::{Arithmetic, ExpansionClient, center, exact_over, expanded},
    context::Context,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, CertifiedExpansion};

/// Selected two-part fields and the radius each selection discards.
pub(super) struct PointRecovery<R: Runtime<DType = DType>> {
    pub area: ConservedPoint<R>,
    pub discharge: ConservedPoint<R>,
    /// Selected high part of `M / V`, or `M` on dry cells.
    pub velocity: Tensor<R>,
    pub area_bound: Tensor<R>,
    pub discharge_bound: Tensor<R>,
}

/// Cap membership of the selected area, before the high-part recovery runs.
struct CapMembership<R: Runtime<DType = DType>> {
    /// Volume parts of `cap_area_upper · length`.
    cap_volume: ConservedPoint<R>,
    /// 1 where the selected area point exceeds the cap area.
    exceeds: Tensor<R>,
    /// 1 where the volume is at or below the cap volume.
    within: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: ExpansionClient<R>,
{
    /// Recover `A = V / dx`, `Q = M / dx` and `M / V` from both parts of the state.
    ///
    /// Rejects an area point above the cap area. Volumes at or below the cap volume
    /// reject with [`ChannelError::CapUnresolved`].
    pub(super) fn surveyed_points(
        &self,
        volume: &ConservedPoint<R>,
        momentum: &ConservedPoint<R>,
        cap_area_upper: &Tensor<R>,
    ) -> Result<PointRecovery<R>, ChannelError> {
        let c = &self.client;
        let (recovery, membership) = expanded(c, "surveyed area and discharge recovery", |a| {
            let zero = Tensor::zeros(volume.high.shape(), DType::F64, c.device())?;
            let one = Tensor::full_scalar(volume.high.shape(), DType::F64, 1.0, c.device())?;
            let v = exact_over(a, &volume.high, &volume.low, &zero)?;
            let m = exact_over(a, &momentum.high, &momentum.low, &zero)?;
            let length = a.expansion_point(&self.length)?;
            // The divisor is positive. The dividend keeps its sign and its low part.
            let area = a.expansion_div_positive(&v, &length)?;
            let discharge = a.expansion_div_positive(&m, &length)?;
            let wet = c.gt(&volume.high, &zero)?;
            let safe = exact_over(
                a,
                &c.where_cond(&wet, &volume.high, &one)?,
                &c.where_cond(&wet, &volume.low, &zero)?,
                &zero,
            )?;
            let velocity = a.expansion_div_positive(&m, &safe)?;
            let recovery = PointRecovery {
                area: center(&area),
                discharge: center(&discharge),
                velocity: velocity.high().clone(),
                area_bound: area.abs_error().clone(),
                discharge_bound: discharge.abs_error().clone(),
            };
            let membership = cap_membership(a, c, &v, &recovery.area, cap_area_upper, &length)?;
            Ok((recovery, membership))
        })?;
        self.check_cap(volume, &recovery, &membership, cap_area_upper)?;
        Ok(recovery)
    }

    /// Reject the first cell whose area point lies above the cap area.
    ///
    /// The mask read is the only host read when every area is within the cap.
    /// A high part above the cap is left to the section recovery and its message.
    fn check_cap(
        &self,
        volume: &ConservedPoint<R>,
        recovery: &PointRecovery<R>,
        membership: &CapMembership<R>,
        cap_area_upper: &Tensor<R>,
    ) -> Result<(), ChannelError> {
        let c = &self.client;
        let zero = self.constant(&membership.exceeds, 0.0)?;
        if self.all(&c.eq(&membership.exceeds, &zero)?)? {
            return Ok(());
        }
        let unresolved = c.mul(&membership.exceeds, &membership.within)?;
        let low_only = c.mul(
            &membership.exceeds,
            &c.le(&recovery.area.high, cap_area_upper)?,
        )?;
        let detail = |index: usize| -> Result<String, ChannelError> {
            let read = |value: &Tensor<R>| value.narrow(0, index, 1)?.item::<f64>();
            Ok(format!(
                "volume_m3=({:?}, {:?}), cap_volume_m3=({:?}, {:?}), length_m={:?}, area_m2=({:?}, {:?}), area_selection_bound_m2={:?}, cap_area_m2={:?}",
                read(&volume.high)?,
                read(&volume.low)?,
                read(&membership.cap_volume.high)?,
                read(&membership.cap_volume.low)?,
                read(&self.length)?,
                read(&recovery.area.high)?,
                read(&recovery.area.low)?,
                read(&recovery.area_bound)?,
                read(cap_area_upper)?,
            ))
        };
        let resolved = c.eq(&unresolved, &zero)?;
        if !self.all(&resolved)? {
            let cell = self.first_rejected(&resolved)?;
            return Err(ChannelError::CapUnresolved {
                cell,
                context: format!(
                    "{}: the volume is at or below the cap volume but the selected area exceeds the cap area. Keep the accepted state, then request a shorter interval or extend the surveyed profile above the cap",
                    detail(cell)?
                ),
            });
        }
        let supported = c.eq(&low_only, &zero)?;
        if !self.all(&supported)? {
            let cell = self.first_rejected(&supported)?;
            return Err(ChannelError::input(format!(
                "{}: the volume is not certified within the cap volume and the area point exceeds surveyed support. Check the original operand units or extend the surveyed profile",
                detail(cell)?
            ))
            .at(cell));
        }
        Ok(())
    }
}

/// Compare the volume with `cap · length` and the selected area with the cap, both exactly.
fn cap_membership<R>(
    a: &mut Arithmetic<'_, R>,
    c: &R::Client,
    volume: &CertifiedExpansion<R>,
    area: &ConservedPoint<R>,
    cap_area_upper: &Tensor<R>,
    length: &CertifiedExpansion<R>,
) -> ArithmeticResult<CapMembership<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let zero = Tensor::zeros(area.high.shape(), DType::F64, c.device())?;
    let cap = a.expansion_point(cap_area_upper)?;
    let cap_volume = a.expansion_mul(&cap, length)?;
    let margin = a.expansion_sub(&cap_volume, volume)?;
    let within = c.ge(a.expansion_interval(&margin)?.lower(), &zero)?;
    let selected = exact_over(a, &area.high, &area.low, &zero)?;
    let excess = a.expansion_sub(&selected, &cap)?;
    let exceeds = c.gt(a.expansion_interval(&excess)?.lower(), &zero)?;
    Ok(CapMembership {
        cap_volume: center(&cap_volume),
        exceeds,
        within,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReach, ChannelReachDefinition,
        ChannelSurveyedConfig, SurveyedChannelCell,
        conserved_update::exact,
        geometry::TensorGeometry,
        surveyed_update::tests::{Client, TestResult, cpu},
    };
    use super::*;
    use crate::channel::{CrossSectionDefinition, CrossSectionPoint};
    use numr::runtime::cpu::CpuRuntime;

    /// Width-4 rectangular survey with cap area 14 m² and cells of one length.
    fn reach<R>(client: R::Client, length_m: f64, count: usize) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        Ok(ChannelReach::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "Synthetic SI datum".into(),
                provenance: "Surveyed two-part recovery".into(),
                geometry: ChannelGeometry::PrismaticSurveyed {
                    profile: CrossSectionDefinition {
                        id: "section".into(),
                        vertical_datum: "Synthetic SI datum".into(),
                        points: [(-2.0, 3.5), (-2.0, 0.0), (2.0, 0.0), (2.0, 3.5)]
                            .into_iter()
                            .map(|(station_m, elevation_m)| CrossSectionPoint {
                                station_m,
                                elevation_m,
                            })
                            .collect(),
                        maximum_stage_m: 3.5,
                        profile_provenance: "Four-metre rectangular survey".into(),
                    },
                    cells: (0..count)
                        .map(|index| SurveyedChannelCell {
                            id: format!("cell-{index}"),
                            length_m,
                            manning_n: 0.0,
                        })
                        .collect(),
                    numerical: ChannelSurveyedConfig::default(),
                },
                lateral_bindings: Vec::new(),
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![length_m; count],
                longitudinal_momentum_m4_per_s: vec![0.0; count],
            },
        )?)
    }

    fn point<R>(reach: &ChannelReach<R>, high: &[f64], low: &[f64]) -> TestResult<ConservedPoint<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        Ok(ConservedPoint {
            high: reach.context.tensor(high)?,
            low: reach.context.tensor(low)?,
        })
    }

    fn values<R: Runtime<DType = DType>>(value: &Tensor<R>) -> TestResult<Vec<f64>> {
        Ok(value.try_to_vec::<f64>()?)
    }

    fn cap_area<R: Runtime<DType = DType>>(reach: &ChannelReach<R>) -> TestResult<Tensor<R>> {
        match &reach.context.geometry {
            TensorGeometry::Surveyed(geometry) => Ok(geometry.cap_area_upper.clone()),
            _ => Err("surveyed geometry required".into()),
        }
    }

    /// `V = (35, -x)` over 2.5 m keeps a positive deficit below the 14 m² cap.
    fn deficit_below_cap<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client, 2.5, 1)?;
        assert_eq!(values(&cap_area(&reach)?)?, vec![14.0]);
        let x = 2f64.powi(-60);
        let fields = reach.context.fields(
            &point(&reach, &[35.0], &[-x])?,
            &point(&reach, &[0.0], &[0.0])?,
        )?;
        let high = values(&fields.area_point.high)?[0];
        let low = values(&fields.area_point.low)?[0];
        let bound = values(&fields.area_selection_bound)?[0];
        // Exact A = 14 - x / 2.5. The low part is the rounded quotient of the exact deficit.
        assert_eq!(high.to_bits(), 14f64.to_bits());
        assert_eq!(low.to_bits(), (-x / 2.5).to_bits());
        assert_eq!(values(&fields.area)?[0].to_bits(), 14f64.to_bits());
        assert!(low < 0.0, "deficit={:e}", -low);
        // The residual 2.5·low + x is the exact oracle error times the length.
        let residual = 2.5f64.mul_add(low, x).abs() / 2.5;
        assert!(residual > 0.0);
        assert!(
            bound >= residual && bound <= 2f64.powi(-100),
            "bound={bound:e}"
        );
        assert_eq!(values(&fields.discharge_point.high)?, vec![0.0]);
        assert_eq!(values(&fields.discharge_point.low)?, vec![0.0]);
        Ok(())
    }

    /// Exactly 35 m³ over 2.5 m recovers the cap area with zero radius.
    fn exact_cap<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client, 2.5, 1)?;
        let fields = reach.context.fields(
            &point(&reach, &[35.0], &[0.0])?,
            &point(&reach, &[0.0], &[0.0])?,
        )?;
        assert_eq!(
            values(&fields.area_point.high)?[0].to_bits(),
            14f64.to_bits()
        );
        assert_eq!(values(&fields.area_point.low)?[0].to_bits(), 0f64.to_bits());
        assert_eq!(values(&fields.area_selection_bound)?, vec![0.0]);
        assert_eq!(values(&fields.discharge_selection_bound)?, vec![0.0]);
        Ok(())
    }

    /// A selected area above the cap while the volume is within the cap volume rejects.
    ///
    /// The correctly rounded division never selects above a represented cap from a volume
    /// within the cap, so the test supplies the selected area directly.
    fn unresolved_cap<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client, 2.5, 2)?;
        let context = &reach.context;
        let cap = cap_area(&reach)?;
        let x = 2f64.powi(-60);
        let above = 2f64.powi(-100);
        let volume = point(&reach, &[35.0, 35.0], &[0.0, -x])?;
        let recovery = PointRecovery {
            area: point(&reach, &[14.0, 14.0], &[0.0, above])?,
            discharge: point(&reach, &[0.0, 0.0], &[0.0, 0.0])?,
            velocity: context.tensor(&[0.0, 0.0])?,
            area_bound: context.tensor(&[0.0, 2.0 * above])?,
            discharge_bound: context.tensor(&[0.0, 0.0])?,
        };
        let membership = expanded(&context.client, "test cap membership", |a| {
            let v = exact(a, &volume.high, &volume.low)?;
            let length = a.expansion_point(&context.length)?;
            cap_membership(a, &context.client, &v, &recovery.area, &cap, &length)
        })?;
        assert_eq!(values(&membership.within)?, vec![1.0, 1.0]);
        assert_eq!(values(&membership.exceeds)?, vec![0.0, 1.0]);
        let error = context
            .check_cap(&volume, &recovery, &membership, &cap)
            .err()
            .ok_or("an area above the cap requires rejection")?;
        assert!(
            matches!(error, ChannelError::CapUnresolved { cell: 1, .. }),
            "{error}"
        );
        assert_eq!(error.cell(), Some(1));
        let message = error.to_string();
        for part in [
            "channel cap unresolved, cell=1",
            "volume_m3=(35.0, -8.673617379884035e-19)",
            "cap_volume_m3=(35.0, 0.0)",
            "area_m2=(14.0, 7.888609052210118e-31)",
            "cap_area_m2=14.0",
            "Keep the accepted state",
        ] {
            assert!(message.contains(part), "{message}");
        }
        Ok(())
    }

    /// A positive low part above the cap volume is rejected even though the high part fits.
    fn low_part_above_cap<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client, 2.5, 1)?;
        let error = reach
            .context
            .fields(
                &point(&reach, &[35.0], &[2f64.powi(-60)])?,
                &point(&reach, &[0.0], &[0.0])?,
            )
            .err()
            .ok_or("an area point above the cap requires rejection")?;
        assert!(
            matches!(error, ChannelError::InvalidInput { cell: Some(0), .. }),
            "{error}"
        );
        assert!(
            error.to_string().contains("exceeds surveyed support"),
            "{error}"
        );
        Ok(())
    }

    /// Negative momentum keeps its sign and its low part through recovery.
    fn negative_momentum<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client, 2.5, 1)?;
        let y = 2f64.powi(-70);
        let fields = reach.context.fields(
            &point(&reach, &[35.0], &[0.0])?,
            &point(&reach, &[-7.5], &[-y])?,
        )?;
        // Exact Q = -3 - y / 2.5.
        assert_eq!(
            values(&fields.discharge_point.high)?[0].to_bits(),
            (-3f64).to_bits()
        );
        assert_eq!(
            values(&fields.discharge_point.low)?[0].to_bits(),
            (-y / 2.5).to_bits()
        );
        assert_eq!(values(&fields.discharge)?[0].to_bits(), (-3f64).to_bits());
        let residual = 2.5f64.mul_add(-y / 2.5, y).abs() / 2.5;
        let bound = values(&fields.discharge_selection_bound)?[0];
        assert!(
            bound >= residual && bound <= 2f64.powi(-100),
            "bound={bound:e}"
        );
        assert_eq!(
            values(&fields.velocity)?[0].to_bits(),
            (-7.5f64 / 35.0).to_bits()
        );
        Ok(())
    }

    #[test]
    fn cpu_deficit_below_cap_keeps_both_area_parts() -> TestResult {
        deficit_below_cap::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_exact_cap_volume_recovers_cap_area_with_zero_radius() -> TestResult {
        exact_cap::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_selected_area_above_cap_within_cap_volume_rejects_cap_unresolved() -> TestResult {
        unresolved_cap::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_low_part_above_cap_volume_rejects_as_support_excess() -> TestResult {
        low_part_above_cap::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_negative_momentum_keeps_sign_and_low_part() -> TestResult {
        negative_momentum::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        fn client() -> TestResult<CudaClient> {
            Ok(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_deficit_below_cap_keeps_both_area_parts() -> TestResult {
            deficit_below_cap::<CudaRuntime>(client()?)
        }

        #[test]
        fn cuda_exact_cap_volume_recovers_cap_area_with_zero_radius() -> TestResult {
            exact_cap::<CudaRuntime>(client()?)
        }

        #[test]
        fn cuda_selected_area_above_cap_within_cap_volume_rejects_cap_unresolved() -> TestResult {
            unresolved_cap::<CudaRuntime>(client()?)
        }

        #[test]
        fn cuda_low_part_above_cap_volume_rejects_as_support_excess() -> TestResult {
            low_part_above_cap::<CudaRuntime>(client()?)
        }

        #[test]
        fn cuda_negative_momentum_keeps_sign_and_low_part() -> TestResult {
            negative_momentum::<CudaRuntime>(client()?)
        }
    }
}
