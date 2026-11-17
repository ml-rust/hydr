use super::super::super::point_flux_quantum::{self, MomentumInputs};
use super::{frozen, hydraulic, support};
use crate::channel::flow::surveyed::error::checked;
use frozen::Result;
use numr::{
    dtype::DType,
    ops::NextRepresentableOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let eta = hydraulic(checked::<R, _>(
        client,
        "caller-format quantum control",
        |a| Ok(client.next_up(&Tensor::zeros(&[], a.dtype(), client.device())?)?),
    ))?
    .item::<f64>()?;
    let tiny = 2.0_f64.powi(-600);
    let (pressure, momentum, original_pressure, original_momentum) = hydraulic(checked::<R, _>(
        client,
        "mixed exact rectangle quantum bounds",
        |a| {
            let area = a.point(&Tensor::from_slice(
                &[tiny, 1., tiny],
                &[3],
                client.device(),
            )?)?;
            let depth = Tensor::from_slice(&[tiny, 1., tiny], &[3], client.device())?;
            let discharge = a.point(&Tensor::from_slice(
                &[2.0_f64.powi(-900), 1., 0.],
                &[3],
                client.device(),
            )?)?;
            let original_pressure = a.bounds(
                &Tensor::from_slice(&[0., 0.5, 0.], &[3], client.device())?,
                &Tensor::from_slice(&[3. * eta, 0.5, 3. * eta], &[3], client.device())?,
            )?;
            let original_momentum = a.bounds(
                &Tensor::from_slice(&[0., 1.5, 0.], &[3], client.device())?,
                &Tensor::from_slice(&[6. * eta, 1.5, 3. * eta], &[3], client.device())?,
            )?;
            let pressure =
                point_flux_quantum::pressure(a, area.upper(), &depth, &original_pressure, 1.)?;
            let momentum = point_flux_quantum::momentum(
                a,
                MomentumInputs {
                    area: &area,
                    discharge: &discharge,
                    depth_upper: &depth,
                    pressure: &pressure,
                    gravity: 1.,
                },
                &original_momentum,
            )?;
            // Exact tiny values are P=2^-1201, adv=2^-1200, and J=3*2^-1201.
            Ok((pressure, momentum, original_pressure, original_momentum))
        },
    ))?;
    assert_eq!(
        frozen::endpoints(&pressure)?,
        [[0., eta], [0.5, 0.5], [0., eta]]
    );
    assert_eq!(
        frozen::endpoints(&momentum)?,
        [[0., eta], [1.5, 1.5], [0., eta]]
    );
    assert_eq!(
        frozen::endpoints(&original_pressure)?,
        [[0., 3. * eta], [0.5, 0.5], [0., 3. * eta]]
    );
    assert_eq!(
        frozen::endpoints(&original_momentum)?,
        [[0., 6. * eta], [1.5, 1.5], [0., 3. * eta]]
    );
    let (pressure, momentum, after) = hydraulic(checked::<R, _>(
        client,
        "two-quantum rectangle negative control",
        |a| {
            let one = Tensor::ones(&[1], a.dtype(), client.device())?;
            let area = a.point(&one)?;
            let discharge = a.point(&Tensor::zeros(&[1], a.dtype(), client.device())?)?;
            let physical = a.bounds(
                &Tensor::from_slice(&[eta], &[1], client.device())?,
                &Tensor::from_slice(&[3. * eta], &[1], client.device())?,
            )?;
            // A=h=1 and g=4eta give exact P=J=2eta, despite passing rounded work filters.
            let pressure =
                point_flux_quantum::pressure(a, area.upper(), &one, &physical, 4. * eta)?;
            let momentum = point_flux_quantum::momentum(
                a,
                MomentumInputs {
                    area: &area,
                    discharge: &discharge,
                    depth_upper: &one,
                    pressure: &pressure,
                    gravity: 4. * eta,
                },
                &physical,
            )?;
            let left = a.point(&one)?;
            let right = a.point(&one)?;
            let after = a.add(&left, &right)?;
            Ok((pressure, momentum, after))
        },
    ))?;
    assert_eq!(frozen::endpoints(&pressure)?, [[eta, 3. * eta]]);
    assert_eq!(frozen::endpoints(&momentum)?, [[eta, 3. * eta]]);
    assert_eq!(frozen::endpoints(&after)?, [[2., 2.]]);
    Ok(())
}

#[test]
fn configured_cpu_point_flux_quantum_proofs_preserve_normal_and_two_quantum_states() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    run::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_point_flux_quantum_proofs_preserve_normal_and_two_quantum_states() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
