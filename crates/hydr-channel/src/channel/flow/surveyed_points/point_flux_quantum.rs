use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{ArithmeticError, Interval, with_interval_arithmetic};

fn reciprocal<R>(
    a: &Arithmetic<'_, R>,
    numerator: &[&Tensor<R>],
    denominator: &[&Tensor<R>],
) -> HydraulicResult<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let result = with_interval_arithmetic::<R, _, _, _>(
        a.client(),
        a.dtype(),
        "optional point flux quantum proof",
        |proof| proof.positive_product_ratio_lower(numerator, denominator),
    );
    match result {
        Ok(lower) => Ok(Some(lower)),
        Err(ArithmeticError::Range { .. }) => Ok(None),
        Err(error) => Err(HydraulicError::Arithmetic(error)),
    }
}

fn pressure_half<R>(
    a: &Arithmetic<'_, R>,
    area_upper: &Tensor<R>,
    depth_upper: &Tensor<R>,
    current: &Interval<R>,
    gravity: f64,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = Tensor::zeros(area_upper.shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(area_upper.shape(), a.dtype(), c.device())?;
    let eta = c.next_up(&Tensor::zeros(&[], a.dtype(), c.device())?)?;
    let four_eta = c.mul_scalar(&eta, 4.)?;
    let maximum = c.next_down(&Tensor::full_scalar(
        &[],
        a.dtype(),
        f64::INFINITY,
        c.device(),
    )?)?;
    let g = Tensor::full_scalar(&[], a.dtype(), gravity, c.device())?;
    let positive = c.mul(&c.gt(area_upper, &zero)?, &c.gt(depth_upper, &zero)?)?;
    let candidate = c.mul(&positive, &c.le(current.upper(), &four_eta)?)?;
    // Rounded native probes select optional work. Only the separate reciprocal certificate authorizes tightening.
    let safe_g = c.where_cond(&candidate, &g, &one)?;
    let safe_h = c.where_cond(&candidate, depth_upper, &one)?;
    let t = c.mul(&safe_g, &safe_h)?;
    let t_valid = c.mul(&candidate, &c.mul(&c.gt(&t, &zero)?, &c.le(&t, &maximum)?)?)?;
    let safe_a = c.where_cond(&t_valid, area_upper, &one)?;
    let safe_t = c.where_cond(&t_valid, &t, &one)?;
    let z = c.mul(&safe_a, &safe_t)?;
    let eligible = c.mul(
        &t_valid,
        &c.mul(&c.le(&z, &maximum)?, &c.le(&z, &four_eta)?)?,
    )?;
    if c.nonzero(&eligible)?.numel() == 0 {
        return Ok((eta, zero));
    }
    let eta_factor = c.where_cond(&eligible, &eta, &one)?;
    let two = c.where_cond(
        &eligible,
        &Tensor::full_scalar(&[], a.dtype(), 2., c.device())?,
        &one,
    )?;
    let g = c.where_cond(&eligible, &g, &one)?;
    let area = c.where_cond(&eligible, area_upper, &one)?;
    let depth = c.where_cond(&eligible, depth_upper, &one)?;
    let proved = match reciprocal(a, &[&eta_factor], &[&two, &g, &area, &depth])? {
        Some(lower) => c.mul(&eligible, &c.ge(&lower, &one)?)?,
        None => zero,
    };
    Ok((eta, proved))
}

fn restrict<R>(
    a: &mut Arithmetic<'_, R>,
    current: &Interval<R>,
    eta: &Tensor<R>,
    proved: &Tensor<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = Tensor::zeros(current.lower().shape(), a.dtype(), c.device())?;
    Ok(a.bounds(
        &c.where_cond(proved, &c.maximum(current.lower(), &zero)?, current.lower())?,
        &c.where_cond(proved, &c.minimum(current.upper(), eta)?, current.upper())?,
    )?)
}

pub(super) fn pressure<R>(
    a: &mut Arithmetic<'_, R>,
    area_upper: &Tensor<R>,
    depth_upper: &Tensor<R>,
    current: &Interval<R>,
    gravity: f64,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    // P<=g*A*h and eta/(2*g*A*h)>=1 establish P<=eta/2 without forming that unrepresentable half quantum.
    let (eta, proved) = pressure_half(a, area_upper, depth_upper, current, gravity)?;
    restrict(a, current, &eta, &proved)
}

pub(super) struct MomentumInputs<'a, R: Runtime<DType = DType>> {
    pub(super) area: &'a Interval<R>,
    pub(super) discharge: &'a Interval<R>,
    pub(super) depth_upper: &'a Tensor<R>,
    pub(super) pressure: &'a Interval<R>,
    pub(super) gravity: f64,
}

pub(super) fn momentum<R>(
    a: &mut Arithmetic<'_, R>,
    inputs: MomentumInputs<'_, R>,
    current: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let MomentumInputs {
        area,
        discharge,
        depth_upper,
        pressure,
        gravity,
    } = inputs;
    let c = a.client();
    let (eta, pressure_proved) = pressure_half(a, area.upper(), depth_upper, pressure, gravity)?;
    let zero = Tensor::zeros(area.lower().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(area.lower().shape(), a.dtype(), c.device())?;
    let maximum = c.next_down(&Tensor::full_scalar(
        &[],
        a.dtype(),
        f64::INFINITY,
        c.device(),
    )?)?;
    let four_eta = c.mul_scalar(&eta, 4.)?;
    let qabs = c.maximum(&c.abs(discharge.lower())?, &c.abs(discharge.upper())?)?;
    let candidate = c.mul(
        &pressure_proved,
        &c.mul(&c.gt(&qabs, &zero)?, &c.gt(area.lower(), &zero)?)?,
    )?;
    let q = c.where_cond(&candidate, &qabs, &one)?;
    let denominator = c.where_cond(&candidate, area.lower(), &one)?;
    let u = c.div(&q, &denominator)?;
    let u_valid = c.mul(&candidate, &c.mul(&c.gt(&u, &zero)?, &c.le(&u, &maximum)?)?)?;
    let q = c.where_cond(&u_valid, &qabs, &one)?;
    let u = c.where_cond(&u_valid, &u, &one)?;
    let v = c.mul(&q, &u)?;
    let eligible = c.mul(
        &u_valid,
        &c.mul(&c.le(&v, &maximum)?, &c.le(&v, &four_eta)?)?,
    )?;
    if c.nonzero(&eligible)?.numel() == 0 {
        let proved = c.mul(&pressure_proved, &c.eq(&qabs, &zero)?)?;
        return restrict(a, current, &eta, &proved);
    }
    let area_factor = c.where_cond(&eligible, area.lower(), &one)?;
    let eta_factor = c.where_cond(&eligible, &eta, &one)?;
    let two = c.where_cond(
        &eligible,
        &Tensor::full_scalar(&[], a.dtype(), 2., c.device())?,
        &one,
    )?;
    let q = c.where_cond(&eligible, &qabs, &one)?;
    let advection = match reciprocal(a, &[&area_factor, &eta_factor], &[&two, &q, &q])? {
        Some(lower) => c.maximum(
            &c.eq(&qabs, &zero)?,
            &c.mul(&eligible, &c.ge(&lower, &one)?)?,
        )?,
        None => c.eq(&qabs, &zero)?,
    };
    // Both nonnegative terms are at most eta/2. Their complete physical sum is at most eta.
    let proved = c.mul(&pressure_proved, &advection)?;
    restrict(a, current, &eta, &proved)
}

#[cfg(test)]
mod tests {
    use super::super::{
        point_flux_quantum,
        source_fixture_support::{self as frozen, Result, hydraulic},
        wet_depth::fixture as support,
    };
    use super::*;
    use crate::channel::flow::surveyed::error::checked;

    fn run<R>(client: &R::Client) -> Result
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
        let (pressure, momentum, original_pressure, original_momentum) = hydraulic(
            checked::<R, _>(client, "mixed exact rectangle quantum bounds", |a| {
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
            }),
        )?;
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
}
