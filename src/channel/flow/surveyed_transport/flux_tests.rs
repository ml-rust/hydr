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
    let expected: Vec<u64> = lower
        .iter()
        .zip(&upper)
        .map(|(l, u)| l.abs().max(u.abs()).to_bits())
        .collect();
    assert_eq!(bits(&flux.speed)?, expected);
    let speed = context.candidate_speed(&fields, &wall, &wall)?;
    assert_eq!(bits(&speed.outer)?, expected);
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
        widths_and_speed_come_from_the_entry::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }

    #[test]
    fn cuda_surveyed_rejected_start_state_names_the_reason() -> TestResult {
        rejected_start_state_names_the_reason::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
