//! Surveyed lateral exchange and transport as two-part expansion updates.

use super::super::ChannelError;
use super::super::conserved_point::ConservedPoint;
use super::super::conserved_update::{ExpansionClient, Parts, bounded, center, exact, expanded};
use super::application::{
    INFLOW_ABSORBED, OUTFLOW_EXCEEDS, WITHDRAWAL_ABSORBED, WITHDRAWAL_EXCEEDS,
};
use super::arithmetic::{add_bounds, scalar_charge, upper_sum, zeros};
use super::expanded_momentum::momentum_update;
use super::{LateralContext, PreparedLateral, SourceApplication};
use crate::numerics::water_transfers::reject;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Selected water states and the radius each selection discards.
struct WaterUpdate<R: Runtime<DType = DType>> {
    retained: ConservedPoint<R>,
    withdrawal_radius: Tensor<R>,
    transported: ConservedPoint<R>,
    transport_radius: Tensor<R>,
    volume: ConservedPoint<R>,
    incoming_radius: Tensor<R>,
}

impl<R: Runtime<DType = DType>> LateralContext<R>
where
    R::Client: ExpansionClient<R>,
{
    /// Apply withdrawal, surveyed transfers, and inflow as two-part expansion updates.
    ///
    /// Each update selects its center once. Withdrawal and inflow radii join the assembly
    /// bounds. Transfer radii join the selection bounds. No radius carries into the next update.
    pub fn apply_lateral_expanded(
        &self,
        client: &R::Client,
        original_volume: &ConservedPoint<R>,
        original_momentum: &ConservedPoint<R>,
        water: &Parts<R>,
        momentum: &Parts<R>,
        prepared: &PreparedLateral<R>,
    ) -> Result<SourceApplication<R>, ChannelError> {
        let c = client;
        let zero = zeros(c, 1)?;
        let update = self.water_update(c, original_volume, water, prepared)?;
        let withdrawal_active = c.gt(&prepared.withdrawal_volume, &zero)?;
        // A normalized point has the sign of its high part.
        reject(c, &c.lt(&update.retained.high, &zero)?, WITHDRAWAL_EXCEEDS)?;
        reject(
            c,
            &c.mul(
                &withdrawal_active,
                &same_point(c, &update.retained, original_volume)?,
            )?,
            WITHDRAWAL_ABSORBED,
        )?;
        reject(c, &c.lt(&update.transported.high, &zero)?, OUTFLOW_EXCEEDS)?;
        let both_positive = c.mul(
            &c.gt(&update.transported.high, &zero)?,
            &c.gt(&prepared.incoming_volume, &zero)?,
        )?;
        let incoming_point = ConservedPoint {
            high: prepared.incoming_volume.clone(),
            low: zeros(c, self.cells())?,
        };
        let absorbed = c.maximum(
            &same_point(c, &update.volume, &update.transported)?,
            &same_point(c, &update.volume, &incoming_point)?,
        )?;
        reject(c, &c.mul(&both_positive, &absorbed)?, INFLOW_ABSORBED)?;
        let water_assembly_bound = add_bounds(
            c,
            &[
                &prepared.incoming_volume_bound,
                &prepared.withdrawal_volume_bound,
                &update.withdrawal_radius,
                &update.incoming_radius,
            ],
        )?;
        let water_assembly_charge_m3 = scalar_charge(
            upper_sum(c, &water_assembly_bound, self.cell_reduction)?,
            prepared.summary.physical_total_roundoff_bound_m3,
        )?;
        let moved = expanded(c, "surveyed lateral momentum update", |a| {
            momentum_update(
                a,
                c,
                [original_volume, &update.retained],
                original_momentum,
                &withdrawal_active,
                momentum,
                prepared,
            )
        })?;
        let [retained_radius, incoming_radius, relationship] = &moved.assembly;
        let momentum_assembly_bound = add_bounds(
            c,
            &[
                &prepared.incoming_momentum_bound,
                &prepared.withdrawal_momentum_bound,
                retained_radius,
                incoming_radius,
                relationship,
            ],
        )?;
        Ok(SourceApplication {
            volume: update.volume.high,
            momentum: moved.momentum.high,
            volume_low: update.volume.low,
            momentum_low: moved.momentum.low,
            water_selection_bound: update.transport_radius,
            momentum_selection_bound: moved.transport_radius,
            water_assembly_bound,
            momentum_assembly_bound,
            water_assembly_charge_m3,
            withdrawn_momentum: Some(moved.withdrawn),
        })
    }

    fn water_update(
        &self,
        client: &R::Client,
        volume: &ConservedPoint<R>,
        water: &Parts<R>,
        prepared: &PreparedLateral<R>,
    ) -> Result<WaterUpdate<R>, ChannelError> {
        expanded(client, "surveyed lateral water update", |a| {
            let original = exact(a, &volume.high, &volume.low)?;
            let withdrawal = a.expansion_point(&prepared.withdrawal_volume)?;
            let retained = a.expansion_sub(&original, &withdrawal)?;
            let retained_point = exact(a, retained.high(), retained.low())?;
            let transfer = bounded(a, water)?;
            let transported = a.expansion_add(&retained_point, &transfer)?;
            let transported_point = exact(a, transported.high(), transported.low())?;
            let incoming = a.expansion_point(&prepared.incoming_volume)?;
            let accepted = a.expansion_add(&transported_point, &incoming)?;
            Ok(WaterUpdate {
                retained: center(&retained),
                withdrawal_radius: retained.abs_error().clone(),
                transported: center(&transported),
                transport_radius: transported.abs_error().clone(),
                volume: center(&accepted),
                incoming_radius: accepted.abs_error().clone(),
            })
        })
    }
}

/// Mask of cells whose two points match in both parts.
fn same_point<R>(
    client: &R::Client,
    a: &ConservedPoint<R>,
    b: &ConservedPoint<R>,
) -> Result<Tensor<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    Ok(client.mul(&client.eq(&a.high, &b.high)?, &client.eq(&a.low, &b.low)?)?)
}

#[cfg(test)]
mod tests {
    use super::super::super::conserved_update::lower_bound;
    use super::super::super::surveyed_update::tests::{Client, TestResult, cpu, exact_sum};
    use super::super::super::{
        ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry,
        ChannelInitialState, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
        ChannelReach, ChannelReachDefinition, ChannelSurveyedConfig, SurveyedChannelCell,
    };
    use super::*;
    use crate::channel::{CrossSectionDefinition, CrossSectionPoint};
    use crate::forcing::WaterAmount;

    /// Width-2 rectangular survey. Cells of length 5 hold 35 m³ at the cap stage.
    fn reach<R>(
        client: R::Client,
        volumes: &[f64],
        momenta: &[f64],
        bindings: &[(&str, usize, ChannelLateralDirection)],
    ) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        Ok(ChannelReach::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "Synthetic SI datum".into(),
                provenance: "Surveyed two-part lateral exchange".into(),
                geometry: ChannelGeometry::PrismaticSurveyed {
                    profile: CrossSectionDefinition {
                        id: "section".into(),
                        vertical_datum: "Synthetic SI datum".into(),
                        points: [(-1.0, 3.5), (-1.0, 0.0), (1.0, 0.0), (1.0, 3.5)]
                            .into_iter()
                            .map(|(station_m, elevation_m)| CrossSectionPoint {
                                station_m,
                                elevation_m,
                            })
                            .collect(),
                        maximum_stage_m: 3.5,
                        profile_provenance: "Two-metre rectangular survey".into(),
                    },
                    cells: (0..volumes.len())
                        .map(|index| SurveyedChannelCell {
                            id: format!("cell-{index}"),
                            length_m: 5.0,
                            manning_n: 0.0,
                        })
                        .collect(),
                    numerical: ChannelSurveyedConfig::default(),
                },
                lateral_bindings: bindings
                    .iter()
                    .map(|&(id, cell, direction)| ChannelLateralBinding {
                        boundary_id: id.into(),
                        cell_id: format!("cell-{cell}"),
                        direction,
                        ownership_assertion: format!("Independent surveyed source {id}"),
                    })
                    .collect(),
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: volumes.to_vec(),
                longitudinal_momentum_m4_per_s: momenta.to_vec(),
            },
        )?)
    }

    fn walls(start_s: f64, end_s: f64, inputs: Vec<ChannelLateralInput>) -> ChannelForcingInterval {
        ChannelForcingInterval {
            start_s,
            end_s,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: inputs,
        }
    }

    fn withdrawal(volume_m3: f64) -> ChannelLateralInput {
        ChannelLateralInput::Withdrawal {
            boundary_id: "out".into(),
            amount: WaterAmount::IntervalVolume { volume_m3 },
        }
    }

    fn withdrawal_absorption<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(
            client,
            &[35.0],
            &[0.0],
            &[("out", 0, ChannelLateralDirection::Withdrawal)],
        )?;
        let context = &reach.context;
        let c = &context.client;
        let zero = context.tensor(&[0.0])?;
        let none = Parts {
            high: zero.clone(),
            low: zero.clone(),
            bound: zero.clone(),
        };
        let still = ConservedPoint {
            high: zero.clone(),
            low: zero.clone(),
        };
        let apply = |high: f64, low: f64, amount: f64| {
            let volume = ConservedPoint::from_parts(context, "volume", &[high], &[low])?;
            let forcing = context
                .lateral
                .bind_host(c, &walls(0.0, 1.0, vec![withdrawal(amount)]))?;
            let prepared = context.lateral.prepare(
                c,
                &forcing.lateral,
                &reach.accepted.lateral,
                &lower_bound(c, &volume)?,
                &zero,
                1.0,
            )?;
            context
                .lateral
                .apply_lateral_expanded(c, &volume, &still, &none, &none, &prepared)
        };
        // A withdrawal below the ulp of the low part leaves both parts unchanged.
        let absorbed = apply(35.0, 2f64.powi(-60), 2f64.powi(-120));
        let message = absorbed.err().map(|error| error.to_string());
        assert!(
            message
                .as_deref()
                .is_some_and(|text| text.contains("withdrawal is absorbed in donor storage")),
            "{message:?}"
        );
        // A withdrawal below the ulp of the high part is kept in the low part.
        for amount in [2f64.powi(-60), 2f64.powi(-120)] {
            let kept = apply(35.0, 0.0, amount)?;
            assert_eq!(kept.volume.item::<f64>()?.to_bits(), 35f64.to_bits());
            assert_eq!(
                kept.volume_low.item::<f64>()?.to_bits(),
                (-amount).to_bits()
            );
            assert_eq!(kept.water_assembly_bound.item::<f64>()?, 0.0);
            assert_eq!(kept.water_selection_bound.item::<f64>()?, 0.0);
            assert_eq!(kept.water_assembly_charge_m3, 0.0);
            assert_eq!(kept.momentum.item::<f64>()?, 0.0);
            assert_eq!(kept.momentum_low.item::<f64>()?, 0.0);
        }
        Ok(())
    }

    fn sourced_steps_conserve<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let origin = reach::<R>(
            client,
            &[30.0, 20.0, 10.0],
            &[3.0, -1.0, 2.0],
            &[
                ("in", 0, ChannelLateralDirection::Inflow),
                ("out", 2, ChannelLateralDirection::Withdrawal),
            ],
        )?;
        let parts = |high: &[f64], low: &[f64]| high.iter().chain(low).copied().collect::<Vec<_>>();
        let initial = origin.snapshot_data()?;
        let initial_water = parts(&initial.volume_m3, &initial.volume_low_m3);
        let mut current = origin;
        let mut nonzero_low = false;
        for _ in 0..3 {
            let start = current.time();
            let inputs = vec![
                ChannelLateralInput::Inflow {
                    boundary_id: "in".into(),
                    amount: WaterAmount::IntervalVolume { volume_m3: 0.1 },
                    longitudinal_velocity_m_per_s: 0.3,
                },
                withdrawal(0.07),
            ];
            let before = current.snapshot_data()?;
            current = current
                .step(start + 0.01, &walls(start, start + 0.01, inputs))?
                .next;
            let after = current.snapshot_data()?;
            nonzero_low |= after
                .volume_low_m3
                .iter()
                .chain(&after.longitudinal_momentum_low_m4_per_s)
                .any(|low| *low != 0.0);
            let receipt = current.last_step()?.ok_or("Missing sourced receipt")?;
            let faces = receipt.face_momentum_m4_per_s.len() - 1;
            // Interior faces cancel. Wall faces, friction, and receipts remain.
            let mut terms = parts(
                &after.longitudinal_momentum_m4_per_s,
                &after.longitudinal_momentum_low_m4_per_s,
            );
            terms.extend(
                parts(
                    &before.longitudinal_momentum_m4_per_s,
                    &before.longitudinal_momentum_low_m4_per_s,
                )
                .iter()
                .map(|part| -part),
            );
            terms.extend([
                -receipt.face_momentum_m4_per_s[0],
                -receipt.face_momentum_low_m4_per_s[0],
                receipt.face_momentum_m4_per_s[faces],
                receipt.face_momentum_low_m4_per_s[faces],
            ]);
            terms.extend(
                receipt
                    .friction_impulse_m4_per_s
                    .iter()
                    .chain(&receipt.friction_impulse_low_m4_per_s)
                    .map(|f| -f),
            );
            terms.extend(
                receipt
                    .lateral_receipts
                    .iter()
                    .map(|r| -r.momentum_m4_per_s),
            );
            let residual: f64 = exact_sum(terms).iter().sum();
            let bound: f64 = receipt
                .momentum_underflow_bound_m4_per_s
                .iter()
                .chain(&receipt.lateral_momentum_abs_bound_m4_per_s)
                .sum();
            assert!(bound < 1e-12, "momentum bound={bound:e}");
            assert!(
                residual.abs() <= bound * (1.0 + 4.0 * f64::EPSILON),
                "momentum residual={residual:e}, bound={bound:e}"
            );
        }
        assert!(nonzero_low, "the fixture must exercise nonzero low parts");
        let data = current.snapshot_data()?;
        let ledger = &data.ledger;
        let change: f64 = exact_sum(
            parts(&data.volume_m3, &data.volume_low_m3)
                .into_iter()
                .chain(initial_water.iter().map(|part| -part)),
        )
        .iter()
        .sum();
        let external = (ledger.boundary_in_m3 - ledger.boundary_out_m3)
            + (ledger.lateral_in_m3 - ledger.lateral_out_m3);
        let bound = ledger.storage_selection_abs_bound_m3
            + ledger.numerical_transfer_underflow_bound_m3
            + ledger.lateral_assembly_abs_bound_m3
            + ledger.lateral_counter_abs_bound_m3;
        let slack = 4.0 * f64::EPSILON * change.abs().max(external.abs());
        assert!(
            (change - external).abs() <= bound + slack,
            "change={change:e}, external={external:e}, bound={bound:e}"
        );
        Ok(())
    }

    #[test]
    fn cpu_surveyed_withdrawal_absorption_uses_both_parts() -> TestResult {
        withdrawal_absorption::<numr::runtime::cpu::CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_sourced_surveyed_steps_conserve_water_and_momentum() -> TestResult {
        sourced_steps_conserve::<numr::runtime::cpu::CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_surveyed_withdrawal_absorption_uses_both_parts() -> TestResult {
            withdrawal_absorption::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_sourced_surveyed_steps_conserve_water_and_momentum() -> TestResult {
            sourced_steps_conserve::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }
}
