use super::sources::{definition, initial, interval};
use super::support::{Client, TestResult, check_water};
use hydr::channel::{
    ChannelReach, ChannelTensorBoundary, ChannelTensorForcingInterval, ChannelTensorLateralInput,
    ChannelTensorWaterAmount,
};
use hydr::forcing::WaterAmount;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn operand<R: Runtime<DType = DType>>(client: &R::Client, value: f64) -> TestResult<Tensor<R>> {
    Ok(Tensor::from_slice(
        &[f64::NAN, value, f64::INFINITY, -999.0],
        &[2, 2],
        client.device(),
    )?
    .narrow(1, 1, 1)?
    .squeeze(Some(1))
    .narrow(0, 0, 1)?)
}

fn native<R: Runtime<DType = DType>>(
    client: &R::Client,
    start_s: f64,
    end_s: f64,
) -> TestResult<ChannelTensorForcingInterval<R>> {
    let exterior = || -> TestResult<ChannelTensorBoundary<R>> {
        Ok(ChannelTensorBoundary::Exterior {
            depth_m: operand::<R>(client, 1.0)?,
            velocity_m_per_s: operand::<R>(client, 3.0)?,
        })
    };
    Ok(ChannelTensorForcingInterval {
        start_s,
        end_s,
        left: exterior()?,
        right: exterior()?,
        lateral_inputs: vec![
            ChannelTensorLateralInput::Inflow {
                boundary_id: "in".into(),
                amount: ChannelTensorWaterAmount::IntervalVolume {
                    volume_m3: operand::<R>(client, 0.0002)?,
                },
                longitudinal_velocity_m_per_s: operand::<R>(client, -1.0)?,
            },
            ChannelTensorLateralInput::Withdrawal {
                boundary_id: "out".into(),
                amount: ChannelTensorWaterAmount::Rate {
                    rate_m3_per_s: operand::<R>(client, 0.1)?,
                },
            },
        ],
    })
}

fn overwrite<R>(client: &R::Client, held: &ChannelTensorForcingInterval<R>) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let replacement = operand::<R>(client, f64::NAN)?;
    for boundary in [&held.left, &held.right] {
        if let ChannelTensorBoundary::Exterior {
            depth_m,
            velocity_m_per_s,
        } = boundary
        {
            client.copy_into(depth_m, &replacement)?;
            client.copy_into(velocity_m_per_s, &replacement)?;
        }
    }
    for input in &held.lateral_inputs {
        let amount = match input {
            ChannelTensorLateralInput::Inflow {
                amount,
                longitudinal_velocity_m_per_s,
                ..
            } => {
                client.copy_into(longitudinal_velocity_m_per_s, &replacement)?;
                amount
            }
            ChannelTensorLateralInput::Withdrawal { amount, .. } => amount,
        };
        match amount {
            ChannelTensorWaterAmount::Rate { rate_m3_per_s } => {
                client.copy_into(rate_m3_per_s, &replacement)?
            }
            ChannelTensorWaterAmount::IntervalVolume { volume_m3 } => {
                client.copy_into(volume_m3, &replacement)?
            }
        }
    }
    Ok(())
}

/// Runs the native source, rejection, and restart checks on a reach of each cell count.
fn execute<R>(client: R::Client, counts: &[usize]) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    for &count in counts {
        let origin = ChannelReach::<R>::new(
            client.clone(),
            definition(count, 0.03, true),
            initial(count, true),
        )?;
        let before = origin.snapshot_data()?;
        let host = interval(
            0.0,
            0.001,
            WaterAmount::IntervalVolume { volume_m3: 0.0002 },
            WaterAmount::Rate { rate_m3_per_s: 0.1 },
        );
        let held = native::<R>(&client, 0.0, 0.001)?;
        let current = origin.step_tensor(0.001, &held)?.next;
        let saved = current.snapshot_data()?;
        assert_eq!(saved, origin.step(0.001, &host)?.next.snapshot_data()?);
        assert_eq!(current.time().to_bits(), 0.001_f64.to_bits());
        let receipt = current
            .last_step()?
            .ok_or("Missing native source receipt")?;
        assert!(receipt.friction_impulse_m4_per_s.iter().all(|v| *v < 0.0));
        assert!(receipt.friction_geometry.is_some());
        assert!(
            receipt
                .riemann
                .as_ref()
                .ok_or("Missing native flux receipt")?
                .surveyed_work
                .is_some()
        );
        check_water(&current)?;
        let checkpoint = current.snapshot();
        overwrite::<R>(&client, &held)?;
        assert!(origin.step_tensor(0.001, &held).is_err());
        assert_eq!(current.snapshot_data()?, saved);
        assert_eq!(checkpoint.data()?, saved);
        assert_eq!(origin.snapshot_data()?, before);
        let trusted = origin.restore(&checkpoint)?;
        let cold = origin.import_snapshot(&saved)?;
        assert_eq!(trusted.snapshot_data()?, saved);
        assert_eq!(cold.snapshot_data()?, saved);
        let next_held = native::<R>(&client, 0.001, 0.002)?;
        let expected = current
            .step_tensor(0.002, &next_held)?
            .next
            .snapshot_data()?;
        for restored in [trusted, cold] {
            assert_eq!(
                restored
                    .step_tensor(0.002, &next_held)?
                    .next
                    .snapshot_data()?,
                expected
            );
        }
        assert_eq!(current.snapshot_data()?, saved);
    }
    Ok(())
}

#[test]
fn native_strided_sources_copy_operands_and_replay_complete_packages() -> TestResult {
    execute::<crate::runtime::Runtime>(crate::runtime::client(), &[1, 33])
}

/// The device runs one reach with an interior face and separate end cells. The CPU test
/// covers the one-cell reach.
#[cfg(feature = "cuda")]
#[test]
fn cuda_surveyed_sources_and_restart_use_actual_selected_device() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    execute::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?, &[3])
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_rejects_surveyed_f64_without_fallback() -> TestResult {
    use hydr::channel::ChannelError;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    let error = ChannelReach::<WgpuRuntime>::new(
        WgpuClient::new(WgpuDevice::new(0))?,
        definition(1, 0.03, true),
        initial(1, true),
    )
    .err()
    .ok_or("WGPU accepted surveyed F64")?;
    assert!(matches!(error, ChannelError::InvalidInput { .. }));
    assert!(error.to_string().contains("F64"));
    Ok(())
}
