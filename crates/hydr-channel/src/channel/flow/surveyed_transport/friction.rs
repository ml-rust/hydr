mod division;
#[cfg(test)]
mod fixture;
mod manning;
mod selection;

use super::source_diagnostics::FrictionGeometryState;
use crate::channel::flow::{
    ChannelError, conserved_point::ConservedPoint, conserved_update::ExpansionClient,
    context::Context, geometry::TensorGeometry,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, MidpointOps, NextRepresentableOps,
    ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(in crate::channel::flow) struct FrictionTrial<R: Runtime<DType = DType>> {
    pub momentum: Tensor<R>,
    pub geometry: Option<FrictionGeometryState<R>>,
}

/// Denominator `1 + drag` of the active rows. Inactive rows hold one.
pub(in crate::channel::flow) struct FrictionDivisor<R: Runtime<DType = DType>> {
    pub denominator: Tensor<R>,
    pub active: Tensor<R>,
}

/// F64 friction trial and, when any row is active, the divisor of its two-part update.
pub(in crate::channel::flow) struct SurveyedDrag<R: Runtime<DType = DType>> {
    pub trial: FrictionTrial<R>,
    pub divisor: Option<FrictionDivisor<R>>,
}

/// Two-part momentum after friction, the friction impulse, and the radius they charge.
pub(in crate::channel::flow) struct SurveyedFriction<R: Runtime<DType = DType>> {
    pub point: ConservedPoint<R>,
    pub impulse: ConservedPoint<R>,
    /// Division radius plus impulse subtraction radius. Zero on inactive rows.
    pub radius: Tensor<R>,
    pub geometry: Option<FrictionGeometryState<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + CorrectlyRoundedOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + ExpansionClient<R>,
{
    pub fn friction_trial(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        dt: f64,
    ) -> Result<FrictionTrial<R>, ChannelError> {
        if matches!(&self.geometry, TensorGeometry::Surveyed(_)) {
            return self.surveyed_friction(volume, momentum, dt);
        }
        Ok(FrictionTrial {
            momentum: self.friction(volume, momentum, dt)?,
            geometry: None,
        })
    }

    pub(in crate::channel::flow) fn surveyed_friction(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        dt: f64,
    ) -> Result<FrictionTrial<R>, ChannelError> {
        Ok(self.surveyed_drag(volume, momentum, dt)?.trial)
    }

    pub(in crate::channel::flow) fn surveyed_drag(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        dt: f64,
    ) -> Result<SurveyedDrag<R>, ChannelError> {
        let TensorGeometry::Surveyed(geometry) = &self.geometry else {
            return Err(ChannelError::input(
                "surveyed Manning requires surveyed geometry: select the matching source method",
            ));
        };
        let c = &self.client;
        let zero = self.constant(momentum, 0.0)?;
        let identity = || SurveyedDrag {
            trial: FrictionTrial {
                momentum: momentum.clone(),
                geometry: Some(FrictionGeometryState {
                    momentum_abs_bound: zero.clone(),
                }),
            },
            divisor: None,
        };
        if dt == 0.0 {
            return Ok(identity());
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(ChannelError::input(format!(
                "surveyed Manning dt={dt}: supply a finite nonnegative duration"
            )));
        }
        let active = c.mul(
            &c.mul(&c.gt(&self.manning, &zero)?, &c.gt(volume, &zero)?)?,
            &c.gt(&c.abs(momentum)?, &zero)?,
        )?;
        let inactive = c.eq(&active, &zero)?;
        if self.all(&inactive)? {
            return Ok(identity());
        }
        let one = self.constant(volume, 1.0)?;
        let area = c.div(&c.where_cond(&active, volume, &zero)?, &self.length)?;
        self.finite(&area, "surveyed Manning original area")?;
        let recovery = geometry.recover_perimeter(&area)?;
        let enclosure = recovery.hydraulic_radius_m();
        let lower = c.where_cond(&active, enclosure.lower(), &one)?;
        let upper = c.where_cond(&active, enclosure.upper(), &one)?;
        let radius = c.midpoint(&lower, &upper)?;
        let magnitude = c.abs(momentum)?;
        let bound =
            self.surveyed_radius_selection_bound(&lower, &upper, &radius, &magnitude, &active)?;
        let (next, denominator) =
            self.surveyed_manning_update(volume, momentum, &radius, &active, dt)?;
        Ok(SurveyedDrag {
            trial: FrictionTrial {
                momentum: next,
                geometry: Some(FrictionGeometryState {
                    momentum_abs_bound: bound,
                }),
            },
            divisor: Some(FrictionDivisor {
                denominator,
                active,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::friction_test_data::ODE;
    use super::fixture::{TestResult, reach, reach_with_client};
    use super::*;
    use numr::ops::HypotOps;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use solvr::optimize::roots::BatchedBisectionAlgorithms;

    #[test]
    fn original_fixed_area_manning_matches_fourteen_exact_power_ode_references() -> TestResult {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        check_ode_references::<CpuRuntime>(client)
    }

    fn check_ode_references<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + CompareOps<R>
            + CorrectlyRoundedOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + BatchedBisectionAlgorithms<R>,
    {
        assert_eq!(ODE.len(), 14);
        for row in ODE {
            let operands = row.operands.map(f64::from_bits);
            assert_eq!((operands[2] / operands[0]).to_bits(), row.operands[1]);
            let reach = reach_with_client::<R>(client.clone(), row, operands[5])?;
            let before = reach.snapshot_data()?;
            let context = &reach.context;
            let trial = context.friction_trial(
                &reach.accepted.volume.high,
                &reach.accepted.momentum.high,
                operands[6],
            )?;
            let actual = trial.momentum.item::<f64>()?;
            let geometry = trial.geometry.ok_or("Missing Manning geometry charge")?;
            let charge = geometry.momentum_abs_bound.item::<f64>()?;
            assert!(
                charge.is_finite() && charge >= 0.0,
                "{}: charge={charge}",
                row.id
            );
            assert!(
                charge <= 1e-9 + 1e-12 * operands[3].abs(),
                "{}: charge={charge}",
                row.id
            );
            let reference = row.reference.map(f64::from_bits);
            if row.active {
                let error = (actual - reference[0])
                    .abs()
                    .max((actual - reference[1]).abs());
                let magnitude = reference[0].abs().min(reference[1].abs());
                let allowance = 1e-10 + 1e-11 * magnitude;
                assert!(
                    error <= allowance,
                    "{}: native={actual:e}, reference={reference:?}, error={error:e}, allowance={allowance:e}",
                    row.id
                );
                assert_eq!(
                    actual.is_sign_negative(),
                    operands[3].is_sign_negative(),
                    "{}",
                    row.id
                );
                assert!(
                    actual.abs() > 0.0 && actual.abs() <= operands[3].abs(),
                    "{}",
                    row.id
                );
            } else {
                assert_eq!(actual.to_bits(), row.operands[3], "{}", row.id);
                assert_eq!(charge.to_bits(), 0.0_f64.to_bits(), "{}", row.id);
            }
            assert_eq!(reach.snapshot_data()?, before);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn configured_cuda_manning_matches_fourteen_exact_power_ode_references() -> TestResult {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        check_ode_references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }

    #[test]
    fn active_manning_retains_duration_and_arithmetic_rejections() -> TestResult {
        let ordinary = ODE.first().ok_or("Missing ordinary Manning fixture")?;
        let normal = reach(ordinary, 0.03)?;
        for duration in [-1.0, f64::NAN, f64::INFINITY] {
            let result = normal.context.friction_trial(
                &normal.accepted.volume.high,
                &normal.accepted.momentum.high,
                duration,
            );
            let error = result
                .err()
                .ok_or("Invalid Manning duration requires rejection")?;
            assert!(error.to_string().contains("duration"));
        }
        for roughness in [f64::from_bits(1), f64::MAX] {
            let reach = reach(ordinary, roughness)?;
            let result = reach.context.friction_trial(
                &reach.accepted.volume.high,
                &reach.accepted.momentum.high,
                0.1,
            );
            let error = result
                .err()
                .ok_or("Unrepresented active Manning arithmetic requires rejection")?;
            assert!(error.to_string().contains("roughness"), "{error}");
        }
        let result = normal.context.friction_trial(
            &normal.accepted.volume.high,
            &normal.accepted.momentum.high,
            f64::from_bits(1),
        );
        let error = result
            .err()
            .ok_or("Unrepresented resistance requires rejection")?;
        assert!(error.to_string().contains("resistance"), "{error}");
        Ok(())
    }

    #[test]
    fn normal_roughness_distinguishes_equal_area_bank_perimeters() -> TestResult {
        let mut outputs = Vec::new();
        for id in ["symmetric_equal_area", "asymmetric_equal_area"] {
            let row = ODE
                .iter()
                .find(|row| row.id == id)
                .ok_or("Missing equal-area Manning fixture")?;
            let reach = reach(row, 0.03)?;
            outputs.push(
                reach
                    .context
                    .friction_trial(
                        &reach.accepted.volume.high,
                        &reach.accepted.momentum.high,
                        0.1,
                    )?
                    .momentum
                    .item::<f64>()?,
            );
        }
        assert!((outputs[0] - outputs[1]).abs() > 1e-6);
        Ok(())
    }
}
